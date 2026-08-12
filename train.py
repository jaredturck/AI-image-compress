''' Train the neural image codec. '''

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.utils import save_image

from model import ImageCodec

DATA_DIR = Path('/path/to/your/images')
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_PATH = CHECKPOINT_DIR / 'latest.pt'

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 6
VALIDATION_SPLIT = 0.02

LEARNING_RATE = 1e-4
AUX_LEARNING_RATE = 1e-3
LAMBDA = 0.010
MAX_STEPS = 15000
GRAD_CLIP = 1.0

LOG_EVERY = 20
PREVIEW_EVERY = 250
SAVE_EVERY = 500
VALIDATION_BATCHES = 8
SEED = 42

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

class ImageDataset(Dataset):
    ''' Load training images and apply a transform. '''

    def __init__(self, paths, transform):
        ''' Store image paths and the dataset transform. '''
        self.paths = paths
        self.transform = transform

    def __len__(self):
        ''' Return the number of images in the dataset. '''
        return len(self.paths)

    def __getitem__(self, index):
        ''' Load and transform one image. '''
        image = decode_image(str(self.paths[index]), mode='RGB')
        return self.transform(image)

def find_images():
    ''' Find supported images under the training directory. '''
    image_paths = []

    for path in DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    return sorted(image_paths)

def train_transform():
    ''' Build the training image augmentation pipeline. '''
    return v2.Compose([
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.45, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])

def validation_transform():
    ''' Build the validation image transform pipeline. '''
    return v2.Compose([
        v2.Resize(int(IMAGE_SIZE * 1.125), antialias=True),
        v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToDtype(torch.float32, scale=True)
    ])

def build_dataloaders():
    ''' Build the training and validation data loaders. '''
    image_paths = find_images()
    if len(image_paths) < 2:
        return None, None, 0

    validation_count = max(1, int(len(image_paths) * VALIDATION_SPLIT))
    train_count = len(image_paths) - validation_count
    generator = torch.Generator().manual_seed(SEED)
    train_paths, validation_paths = random_split(image_paths, [train_count, validation_count], generator=generator)

    train_dataset = ImageDataset(list(train_paths), train_transform())
    validation_dataset = ImageDataset(list(validation_paths), validation_transform())
    loader_options = {
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': True,
        'persistent_workers': NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)
    return train_loader, validation_loader, len(image_paths)

def count_parameters(model):
    ''' Count trainable and non-trainable model parameters. '''
    return sum(parameter.numel() for parameter in model.parameters())

def evaluate(model, validation_loader, accelerator):
    ''' Evaluate bitrate and reconstruction error on validation images. '''
    model.eval()
    total_bpp = torch.zeros((), device=accelerator.device)
    total_mse = torch.zeros((), device=accelerator.device)
    count = 0

    with torch.no_grad():
        for images in validation_loader:
            _, bpp, mse = model(images)
            total_bpp += bpp
            total_mse += mse
            count += 1

            if count >= VALIDATION_BATCHES:
                break

    metrics = torch.stack([total_bpp / count, total_mse / count])
    metrics = accelerator.reduce(metrics, reduction='mean')
    model.train()
    return metrics[0].item(), metrics[1].item()

def save_checkpoint(model, accelerator, step):
    ''' Save the latest training checkpoint. '''
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    checkpoint = {
        'model': unwrapped.state_dict(),
        'step': step,
        'parameters': count_parameters(unwrapped),
    }
    accelerator.save(checkpoint, CHECKPOINT_PATH)

def save_preview(images, reconstruction, step, accelerator):
    ''' Save a grid of source and reconstructed images. '''
    if not accelerator.is_main_process:
        return

    preview_dir = CHECKPOINT_DIR / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = torch.cat([images[:4].float(), reconstruction[:4].float()], dim=0)
    save_image(preview, preview_dir / f'step_{step:07d}.png', nrow=4)

def main():
    ''' Train the codec until the configured step limit. '''
    accelerator = Accelerator()
    set_seed(SEED)

    if not DATA_DIR.exists():
        if accelerator.is_main_process:
            print(f'Set DATA_DIR at the top of train.py first. Current value: {DATA_DIR}')
        return

    train_loader, validation_loader, image_count = build_dataloaders()
    if train_loader is None:
        if accelerator.is_main_process:
            print('The image folder needs at least two JPG, PNG, or WEBP images.')
        return

    model = ImageCodec()
    main_parameters = [parameter for name, parameter in model.named_parameters() if not name.endswith('.quantiles')]
    aux_parameters = [parameter for name, parameter in model.named_parameters() if name.endswith('.quantiles')]
    optimizer = torch.optim.Adam(main_parameters, lr=LEARNING_RATE)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=AUX_LEARNING_RATE)
    model, optimizer, aux_optimizer, train_loader, validation_loader = accelerator.prepare(
        model, optimizer, aux_optimizer, train_loader, validation_loader
    )
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]

    if accelerator.is_main_process:
        parameters = count_parameters(accelerator.unwrap_model(model))
        print(f'images: {image_count:,}')
        print(f'parameters: {parameters / 1e6:.2f}M')
        print(f'devices: {accelerator.num_processes}')
        print(f'mixed precision: {accelerator.mixed_precision}')

    model.train()
    step = 0

    while step < MAX_STEPS:
        for images in train_loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, bpp, mse = model(images)
            loss = bpp + LAMBDA * (255.0 ** 2) * mse
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(optimizer_parameters, GRAD_CLIP)
            optimizer.step()

            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = accelerator.unwrap_model(model).aux_loss()
            accelerator.backward(aux_loss)
            aux_optimizer.step()
            step += 1

            if step % LOG_EVERY == 0:
                metrics = torch.stack([loss.detach(), bpp.detach(), mse.detach()])
                metrics = accelerator.reduce(metrics, reduction='mean')
                psnr = -10.0 * math.log10(max(metrics[2].item(), 1e-12))

                if accelerator.is_main_process:
                    print(f'step {step:7d} | loss {metrics[0].item():.4f} | bpp {metrics[1].item():.4f} | psnr {psnr:.2f} dB')

            if step % PREVIEW_EVERY == 0:
                save_preview(images, reconstruction, step, accelerator)

            if step % SAVE_EVERY == 0:
                validation_bpp, validation_mse = evaluate(model, validation_loader, accelerator)
                validation_psnr = -10.0 * math.log10(max(validation_mse, 1e-12))

                if accelerator.is_main_process:
                    print(f'validation | bpp {validation_bpp:.4f} | psnr {validation_psnr:.2f} dB')

                save_checkpoint(model, accelerator, step)

            if step >= MAX_STEPS:
                break

    save_checkpoint(model, accelerator, step)

if __name__ == '__main__':
    main()
