''' Train the neural image codec. '''

import math, os, time
from pathlib import Path

import torch, torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

from model import ImageCodec

DATA_DIR = Path('/path/to/your/images')
CHECKPOINT_PATH = Path('checkpoints/latest.pt')
LEARNING_RATE = 1e-4
LAMBDA = 0.010
MAX_STEPS = 15000
BATCH_SIZE = 8

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
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])
    dataset = ImageDataset(image_paths, transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)

def save_checkpoint(model, accelerator):
    ''' Save the latest training checkpoint. '''
    model = accelerator.unwrap_model(model)
    model.update(force=True, update_quantiles=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        accelerator.save(model.state_dict(), CHECKPOINT_PATH)

    accelerator.wait_for_everyone()

def main():
    ''' Train the codec until the configured step limit. '''
    accelerator = Accelerator()
    train_loader = prepare_data()
    if train_loader is None:
        return

    model = ImageCodec()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    if accelerator.is_main_process:
        print(f'Training started | batch size 8 | {MAX_STEPS:,} batches')

    model.train()
    step = 0
    last_log = time.monotonic()
    last_save = last_log

    while step < MAX_STEPS:
        for batch in train_loader:
            optimizer.zero_grad()
            reconstruction, bpp = model(batch)
            mse = F.mse_loss(reconstruction.float(), batch.float())
            loss = bpp + LAMBDA * (255.0 ** 2) * mse
            accelerator.backward(loss)
            optimizer.step()
            step += 1

            current_time = time.monotonic()

            if current_time - last_log >= 20:
                if accelerator.is_main_process:
                    psnr = -10.0 * math.log10(max(mse.item(), 1e-12))
                    print(f'step {step:7d} | loss {loss.item():.4f} | bpp {bpp.item():.4f} | psnr {psnr:.2f} dB')

                last_log = current_time

                if current_time - last_save >= 300:
                    save_checkpoint(model, accelerator)
                    last_save = current_time

            if step >= MAX_STEPS:
                break

    save_checkpoint(model, accelerator)

if __name__ == '__main__':
    main()
