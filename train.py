''' Train the neural image codec. '''

import os, queue, threading, time
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

# accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py

DATA_DIR = Path('/mnt/8TB_HDD/datasets/anime_dataset/train/')
CHECKPOINT_DIR = Path('checkpoints')
LEARNING_RATE = 1e-4
LAMBDA = 0.010
MAX_STEPS = 15000
MAX_IMAGES = 100_000
BATCH_SIZE = 128
STOP_CHECK_INTERVAL = 25
STOP_MIN_DELTA = 0.01
STOP_PATIENCE = 5
STOP_MIN_STEPS = 500
IMAGE_SIZE = 256
JPEG_BATCH_SIZE = 64
MAX_JPEG_FUTURES = 4
MAX_CPU_FUTURES = 64
CPU_WORKERS = 6

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

def store_completed_futures(cache, cache_size, futures, progress):
    ''' Store completed preparation jobs and update progress. '''
    completed, _ = wait(futures, return_when=FIRST_COMPLETED)

    for future in completed:
        item_count = futures.pop(future)
        cache_size = store_prepared(cache, cache_size, future.result())
        progress.update(item_count)

    return cache_size

def prepare_data(accelerator):
    ''' Prepare a uint8 training cache with ordered I/O and overlapped preprocessing. '''
    if not DATA_DIR.exists():
        return None

    filenames = sorted(filename for filename in os.listdir(DATA_DIR) if filename.lower().endswith(('.jpg', '.jpeg', '.png')))
    filenames = filenames[:MAX_IMAGES]
    filenames = filenames[accelerator.process_index::accelerator.num_processes]

    if not filenames:
        return None

    jpeg_transform = v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
    cpu_transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
    ])
    cache = torch.empty((len(filenames), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    cache_size = 0
    encoded_queue = queue.Queue(maxsize=512)
    reader = threading.Thread(target=read_images, args=(filenames, encoded_queue), daemon=True)
    reader.start()

    jpeg_items = []
    jpeg_futures = {}
    cpu_futures = {}
    progress = tqdm(total=len(filenames), desc='Preparing images', unit='image', disable=not accelerator.is_main_process)

    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as cpu_pool, ThreadPoolExecutor(max_workers=1) as jpeg_pool:
        while True:
            item = encoded_queue.get()
            if item is None:
                break

            filename, encoded = item
            if encoded is None:
                progress.update(1)
                continue

            if filename.lower().endswith(('.jpg', '.jpeg')):
                jpeg_items.append(item)

                if len(jpeg_items) >= JPEG_BATCH_SIZE:
                    future = jpeg_pool.submit(prepare_jpegs, jpeg_items, jpeg_transform, accelerator.device)
                    jpeg_futures[future] = len(jpeg_items)
                    jpeg_items = []
            else:
                future = cpu_pool.submit(prepare_cpu_image, encoded, cpu_transform)
                cpu_futures[future] = 1

            if len(cpu_futures) >= MAX_CPU_FUTURES:
                cache_size = store_completed_futures(cache, cache_size, cpu_futures, progress)

            if len(jpeg_futures) >= MAX_JPEG_FUTURES:
                cache_size = store_completed_futures(cache, cache_size, jpeg_futures, progress)

        if jpeg_items:
            future = jpeg_pool.submit(prepare_jpegs, jpeg_items, jpeg_transform, accelerator.device)
            jpeg_futures[future] = len(jpeg_items)

        while cpu_futures:
            cache_size = store_completed_futures(cache, cache_size, cpu_futures, progress)

        while jpeg_futures:
            cache_size = store_completed_futures(cache, cache_size, jpeg_futures, progress)

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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    return checkpoint_path

def save_checkpoint(model, accelerator, loss, dataset_images, step):
    ''' Save a checkpoint and training metadata in a three-file rolling window. '''
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

    metadata = {
        'loss': loss,
        'dataset_images': dataset_images,
        'images_seen': step * BATCH_SIZE * accelerator.num_processes,
        'steps': step,
    }
    temporary_path = CHECKPOINT_DIR / '.checkpoint.tmp'
    torch.save({'model': accelerator.unwrap_model(model).state_dict(), 'metadata': metadata}, temporary_path)
    os.replace(temporary_path, checkpoint_path)

def main():
    ''' Train the codec until convergence or the configured step limit. '''
    torch.backends.cudnn.benchmark = True
    ddp_config = DistributedDataParallelKwargs(broadcast_buffers=False, bucket_cap_mb=10, gradient_as_bucket_view=True, static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_config])
    gpu_count = torch.cuda.device_count()

    if accelerator.is_main_process and gpu_count > 1 and accelerator.num_processes == 1:
        print(f'Warning: {gpu_count} CUDA GPUs detected but only one training process is active. Did you forget to run with Accelerate?')

    train_loader = prepare_data(accelerator)
    if train_loader is None:
        return

    dataset_images = torch.tensor(len(train_loader.dataset), device=accelerator.device)
    dataset_images = int(accelerator.reduce(dataset_images, reduction='sum').item())

    model = ImageCodec()
    checkpoint_path = load_checkpoint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=True)
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        weights = checkpoint_path.name if checkpoint_path else 'random'
        print(f'Training started | batch size {BATCH_SIZE} | weights {weights}')

    model.train()
    step = 0
    epoch = 1
    loss_count = 0
    stop_checks = 0
    best_average_loss = None
    latest_average_loss = None
    should_stop = False
    loss_sum = torch.zeros((), device=accelerator.device)
    last_log = time.monotonic()
    last_save = last_log

    while step < MAX_STEPS and not should_stop:
        for batch_number, batch in enumerate(train_loader, 1):
            batch = batch.to(accelerator.device, non_blocking=True).float().mul_(1.0 / 255.0)
            optimizer.zero_grad()
            reconstruction, bpp = model(batch)
            mse = F.mse_loss(reconstruction, batch)
            loss = bpp + LAMBDA * (255.0 ** 2) * mse
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum.add_(loss.detach())
            loss_count += 1
            step += 1

            if loss_count >= STOP_CHECK_INTERVAL:
                average_loss = accelerator.reduce(loss_sum / loss_count, reduction='mean')
                average_loss_value = average_loss.item()
                latest_average_loss = average_loss_value
                loss_sum.zero_()
                loss_count = 0

                if step < STOP_MIN_STEPS:
                    best_average_loss = average_loss_value
                    stop_checks = 0
                elif best_average_loss is None:
                    best_average_loss = average_loss_value
                elif best_average_loss - average_loss_value >= STOP_MIN_DELTA:
                    best_average_loss = average_loss_value
                    stop_checks = 0
                else:
                    stop_checks += 1

                if stop_checks >= STOP_PATIENCE:
                    should_stop = True
                    if accelerator.is_main_process:
                        print(f'Stopping early | average loss {average_loss_value:.4f} | no improvement of {STOP_MIN_DELTA:.4f} for {STOP_PATIENCE} checks')

            if accelerator.is_main_process:
                current_time = time.monotonic()

                if current_time - last_log >= 20:
                    print(f'Epoch {epoch}, batch {batch_number} of {len(train_loader)}, loss={loss.item():.4f}')
                    last_log = current_time

                    if current_time - last_save >= 300:
                        save_checkpoint(model, accelerator, latest_average_loss, dataset_images, step)
                        last_save = current_time

            if step >= MAX_STEPS or should_stop:
                break

        epoch += 1

    if accelerator.is_main_process:
        save_checkpoint(model, accelerator, latest_average_loss, dataset_images, step)

if __name__ == '__main__':
    main()
