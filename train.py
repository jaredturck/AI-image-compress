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
CHECKPOINT_PATH = Path('checkpoints/latest.pt')
LEARNING_RATE = 1e-4
LAMBDA = 0.010
MAX_STEPS = 15000

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

            if count >= 8:
                break

    metrics = torch.stack([total_bpp / count, total_mse / count])
    metrics = accelerator.reduce(metrics, reduction='mean')
    model.train()
    return metrics[0].item(), metrics[1].item()

def save_checkpoint(model, accelerator):
    ''' Save the latest training checkpoint. '''
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    accelerator.save(accelerator.unwrap_model(model).state_dict(), CHECKPOINT_PATH)

def save_preview(images, reconstruction, step, accelerator):
    ''' Save a grid of source and reconstructed images. '''
    if not accelerator.is_main_process:
        return

    preview_dir = CHECKPOINT_PATH.parent / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = torch.cat([images[:4].float(), reconstruction[:4].float()], dim=0)
    save_image(preview, preview_dir / f'step_{step:07d}.png', nrow=4)

def main():
    ''' Train the codec until the configured step limit. '''
    accelerator = Accelerator()
    set_seed(42)

    if not DATA_DIR.exists():
        if accelerator.is_main_process:
            print(f'Set DATA_DIR at the top of train.py first. Current value: {DATA_DIR}')
        return

    image_paths = []
    for path in DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            image_paths.append(path)
    image_paths.sort()

    if len(image_paths) < 2:
        if accelerator.is_main_process:
            print('The image folder needs at least two JPG, PNG, or WEBP images.')
        return

    validation_count = max(1, int(len(image_paths) * 0.02))
    train_paths, validation_paths = random_split(image_paths, [len(image_paths) - validation_count, validation_count],
        generator=torch.Generator().manual_seed(42))
    train_dataset = ImageDataset(list(train_paths), v2.Compose([
        v2.RandomResizedCrop((256, 256), scale=(0.45, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ]))
    validation_dataset = ImageDataset(list(validation_paths), v2.Compose([
        v2.Resize(288, antialias=True),
        v2.CenterCrop((256, 256)),
        v2.ToDtype(torch.float32, scale=True)
    ]))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    validation_loader = DataLoader(validation_dataset, batch_size=8, num_workers=6, pin_memory=True)

    model = ImageCodec()
    main_parameters = [parameter for name, parameter in model.named_parameters() if not name.endswith('.quantiles')]
    aux_parameters = [parameter for name, parameter in model.named_parameters() if name.endswith('.quantiles')]
    optimizer = torch.optim.Adam(main_parameters, lr=LEARNING_RATE)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
    model, optimizer, aux_optimizer, train_loader, validation_loader = accelerator.prepare(
        model, optimizer, aux_optimizer, train_loader, validation_loader)

    if accelerator.is_main_process:
        parameters = sum(parameter.numel() for parameter in accelerator.unwrap_model(model).parameters())
        print(f'images: {len(image_paths):,}')
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
            accelerator.clip_grad_norm_(main_parameters, 1.0)
            optimizer.step()

            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = accelerator.unwrap_model(model).aux_loss()
            accelerator.backward(aux_loss)
            aux_optimizer.step()
            step += 1

            if step % 20 == 0:
                metrics = accelerator.reduce(torch.stack([loss.detach(), bpp.detach(), mse.detach()]), reduction='mean')
                psnr = -10.0 * math.log10(max(metrics[2].item(), 1e-12))

                if accelerator.is_main_process:
                    print(f'step {step:7d} | loss {metrics[0].item():.4f} | bpp {metrics[1].item():.4f} | psnr {psnr:.2f} dB')

            if step % 250 == 0:
                save_preview(images, reconstruction, step, accelerator)

            if step % 500 == 0:
                validation_bpp, validation_mse = evaluate(model, validation_loader, accelerator)
                validation_psnr = -10.0 * math.log10(max(validation_mse, 1e-12))

                if accelerator.is_main_process:
                    print(f'validation | bpp {validation_bpp:.4f} | psnr {validation_psnr:.2f} dB')

                save_checkpoint(model, accelerator)

            if step >= MAX_STEPS:
                break

    save_checkpoint(model, accelerator)

if __name__ == '__main__':
    main()
