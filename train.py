''' Train the neural image codec. '''

import math, os, queue, random, threading, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import torch, torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torchvision.io import decode_image, decode_jpeg, read_file
from torchvision.transforms import v2
from tqdm import tqdm

from model import ImageCodec

DATA_DIR = Path('/mnt/8TB_HDD/datasets/anime_dataset/train/')
CHECKPOINT_DIR = Path('checkpoints')
LEARNING_RATE = 1e-4
LAMBDA = 0.010
MAX_STEPS = 15000
MAX_IMAGES = 20000
BATCH_SIZE = 32

def read_images(filenames, encoded_queue):
    ''' Continuously read encoded image files into the preparation queue. '''
    for filename in filenames:
        try:
            encoded = read_file(DATA_DIR / filename)
        except (OSError, RuntimeError):
            encoded = None
        encoded_queue.put((filename, encoded))
    encoded_queue.put(None)

def prepare_cpu_image(encoded, transform):
    ''' Decode and prepare one non-JPEG image on the CPU. '''
    try:
        return transform(decode_image(encoded, mode='RGB'))
    except RuntimeError:
        return None

def prepare_jpegs(items, transform, device):
    ''' Decode and prepare a JPEG batch on the local GPU. '''
    encoded = [item[1] for item in items]
    try:
        images = decode_jpeg(encoded, mode='RGB', device=device)
        return torch.stack([transform(image) for image in images]).cpu()
    except RuntimeError:
        prepared = []
        for image_data in encoded:
            try:
                image = decode_jpeg(image_data, mode='RGB', device=device)
                prepared.append(transform(image))
            except RuntimeError:
                continue
        if not prepared:
            return None
        return torch.stack(prepared).cpu()

def store_prepared(cache, cache_size, prepared):
    ''' Copy prepared images into the contiguous RAM cache. '''
    if prepared is None:
        return cache_size
    if prepared.ndim == 3:
        cache[cache_size].copy_(prepared)
        return cache_size + 1
    count = prepared.shape[0]
    cache[cache_size:cache_size + count].copy_(prepared)
    return cache_size + count

def prepare_data(accelerator):
    ''' Prepare a uint8 training cache with overlapped I/O and preprocessing. '''
    if not DATA_DIR.exists():
        return None

    filenames = [filename for filename in os.listdir(DATA_DIR) if filename.lower().endswith(('.jpg', '.png'))]
    filenames = filenames[accelerator.process_index::accelerator.num_processes]
    image_limit = MAX_IMAGES // accelerator.num_processes + (accelerator.process_index < MAX_IMAGES % accelerator.num_processes)
    if len(filenames) > image_limit:
        selected = set(random.sample(range(len(filenames)), image_limit))
        filenames = [filename for index, filename in enumerate(filenames) if index in selected]
    if not filenames:
        return None

    jpeg_transform = v2.Compose([
        v2.RandomResizedCrop((256, 256), scale=(0.45, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    cpu_transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop((256, 256), scale=(0.45, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    cache = torch.empty((len(filenames), 3, 256, 256), dtype=torch.uint8)
    cache_size = 0
    encoded_queue = queue.Queue(maxsize=512)
    reader = threading.Thread(target=read_images, args=(filenames, encoded_queue), daemon=True)
    reader.start()

    jpeg_items = []
    cpu_futures = set()
    progress = tqdm(total=len(filenames), desc='Preparing images', unit='image', disable=not accelerator.is_main_process)

    with ThreadPoolExecutor(max_workers=4) as cpu_pool:
        while True:
            item = encoded_queue.get()
            if item is None:
                break

            filename, encoded = item
            if encoded is None:
                progress.update(1)
                continue

            if filename.lower().endswith('.jpg'):
                jpeg_items.append(item)
                if len(jpeg_items) >= 64:
                    prepared = prepare_jpegs(jpeg_items, jpeg_transform, accelerator.device)
                    cache_size = store_prepared(cache, cache_size, prepared)
                    progress.update(len(jpeg_items))
                    jpeg_items.clear()
            else:
                cpu_futures.add(cpu_pool.submit(prepare_cpu_image, encoded, cpu_transform))
                if len(cpu_futures) >= 64:
                    completed, cpu_futures = wait(cpu_futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        cache_size = store_prepared(cache, cache_size, future.result())
                        progress.update(1)

        if jpeg_items:
            prepared = prepare_jpegs(jpeg_items, jpeg_transform, accelerator.device)
            cache_size = store_prepared(cache, cache_size, prepared)
            progress.update(len(jpeg_items))

        while cpu_futures:
            completed, cpu_futures = wait(cpu_futures, return_when=FIRST_COMPLETED)
            for future in completed:
                cache_size = store_prepared(cache, cache_size, future.result())
                progress.update(1)

    reader.join()
    progress.close()
    if cache_size == 0:
        return None

    cache = cache[:cache_size]
    return DataLoader(cache, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, drop_last=True)

def load_checkpoint(model):
    ''' Load the newest checkpoint weights when available. '''
    checkpoints = list(CHECKPOINT_DIR.glob('checkpoint_*.pt'))
    if not checkpoints:
        return None

    checkpoint_path = max(checkpoints, key=lambda path: path.stat().st_mtime)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    return checkpoint_path

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
    checkpoint_path = load_checkpoint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=True)
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        weights = checkpoint_path.name if checkpoint_path else 'random'
        print(f'Training started | batch size {BATCH_SIZE} | {MAX_STEPS:,} batches | weights {weights}')

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
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
