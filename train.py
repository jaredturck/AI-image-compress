''' Train the neural image codec. '''

import math, os, time
from pathlib import Path

import torch, torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

from model import ImageCodec

DATA_DIR = Path('/mnt/8TB_HDD/datasets/anime_dataset/train/')
CHECKPOINT_DIR = Path('checkpoints')
LEARNING_RATE = 1e-4
LAMBDA = 0.010
MAX_STEPS = 15000
BATCH_SIZE = 32

class ImageDataset(Dataset):
    ''' Load and transform training images. '''

    def __init__(self, paths, transform):
        ''' Store image paths and the training transform. '''
        self.paths = paths
        self.transform = transform

    def __len__(self):
        ''' Return the number of training images. '''
        return len(self.paths)

    def __getitem__(self, index):
        ''' Load and transform one training image. '''
        image = decode_image(str(self.paths[index]), mode='RGB')
        return self.transform(image)

def prepare_data():
    ''' Prepare the training image loader. '''
    if not DATA_DIR.exists():
        return None

    image_paths = [DATA_DIR / filename for filename in os.listdir(DATA_DIR) if filename.lower().endswith(('.jpg', '.png'))]
    if not image_paths:
        return None

    transform = v2.Compose([
        v2.RandomResizedCrop((256, 256), scale=(0.45, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    dataset = ImageDataset(image_paths, transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=4)

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
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    ddp_config = DistributedDataParallelKwargs(broadcast_buffers=False, bucket_cap_mb=10, gradient_as_bucket_view=True, static_graph=True)
    accelerator = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=[ddp_config])
    train_loader = prepare_data()
    if train_loader is None:
        return

    model = ImageCodec()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=True)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    if accelerator.is_main_process:
        print(f'Training started | batch size {BATCH_SIZE} | {MAX_STEPS:,} batches')

    model.train()
    step = 0
    last_log = time.monotonic()
    last_save = last_log

    while step < MAX_STEPS:
        for batch in train_loader:
            batch = batch.float().mul_(1.0 / 255.0)
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
