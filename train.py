''' Train the neural image codec. '''

import math, os, random, time
from pathlib import Path

import torch, torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2

from model import ImageCodec

DATA_DIR = Path('/mnt/8TB_HDD/datasets/anime_dataset/train/')
CHECKPOINT_DIR = Path('checkpoints')
LEARNING_RATE = 1e-4
LAMBDA = 0.010
MAX_STEPS = 15000
MAX_IMAGES = 20000
BATCH_SIZE = 32

def prepare_data(accelerator):
    ''' Prepare a uint8 training cache in system memory. '''
    if not DATA_DIR.exists():
        return None

    filenames = sorted(filename for filename in os.listdir(DATA_DIR) if filename.lower().endswith(('.jpg', '.png')))
    filenames = filenames[accelerator.process_index::accelerator.num_processes]
    image_limit = MAX_IMAGES // accelerator.num_processes + (accelerator.process_index < MAX_IMAGES % accelerator.num_processes)
    if len(filenames) > image_limit:
        filenames = random.sample(filenames, image_limit)
    if not filenames:
        return None

    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop((256, 256), scale=(0.45, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    cache = torch.empty((len(filenames), 3, 256, 256), dtype=torch.uint8)
    cache_size = 0

    for filename in filenames:
        try:
            image = decode_image(str(DATA_DIR / filename), mode='RGB')
            cache[cache_size].copy_(transform(image))
            cache_size += 1
        except RuntimeError:
            continue

    if cache_size == 0:
        return None

    cache = cache[:cache_size]
    return DataLoader(cache, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, drop_last=True)

def save_checkpoint(model, accelerator):
    ''' Save a checkpoint in a three-file rolling window. '''
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = list(CHECKPOINT_DIR.glob('checkpoint_*.pt'))
    checkpoint_path = None

    for index in range(1, 4):
        candidate = CHECKPOINT_DIR / f'checkpoint_{index}.pt'
        if not candidate.exists():
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        checkpoint_path = min(checkpoints, key=lambda path: path.stat().st_mtime)

    temporary_path = CHECKPOINT_DIR / '.checkpoint.tmp'
    torch.save(accelerator.unwrap_model(model).state_dict(), temporary_path)
    os.replace(temporary_path, checkpoint_path)

def main():
    ''' Train the codec until the configured step limit. '''
    torch.backends.cudnn.benchmark = True
    ddp_config = DistributedDataParallelKwargs(broadcast_buffers=False, bucket_cap_mb=10, gradient_as_bucket_view=True, static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_config])
    train_loader = prepare_data(accelerator)
    if train_loader is None:
        return

    model = ImageCodec()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=True)
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        print(f'Training started | batch size {BATCH_SIZE} | {MAX_STEPS:,} batches')

    model.train()
    step = 0
    last_log = time.monotonic()
    last_save = last_log

    while step < MAX_STEPS:
        for batch in train_loader:
            batch = batch.to(accelerator.device, non_blocking=True).float().mul_(1.0 / 255.0)
            optimizer.zero_grad()
            reconstruction, bpp = model(batch)
            mse = F.mse_loss(reconstruction, batch)
            loss = bpp + LAMBDA * (255.0 ** 2) * mse
            accelerator.backward(loss)
            optimizer.step()
            step += 1

            if accelerator.is_main_process:
                current_time = time.monotonic()
                if current_time - last_log >= 20:
                    loss_value, bpp_value, mse_value = torch.stack([loss.detach(), bpp.detach(), mse.detach()]).tolist()
                    psnr = -10.0 * math.log10(max(mse_value, 1e-12))
                    print(f'step {step:7d} | loss {loss_value:.4f} | bpp {bpp_value:.4f} | psnr {psnr:.2f} dB')
                    last_log = current_time

                    if current_time - last_save >= 300:
                        save_checkpoint(model, accelerator)
                        last_save = current_time

            if step >= MAX_STEPS:
                break

    if accelerator.is_main_process:
        save_checkpoint(model, accelerator)

if __name__ == '__main__':
    main()
