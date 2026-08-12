from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import decode_image
from torchvision.transforms import v2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = decode_image(str(self.paths[index]), mode="RGB")
        return self.transform(image)


def find_images(data_dir):
    paths = []
    for path in Path(data_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def train_transform(image_size):
    return v2.Compose([
        v2.RandomResizedCrop(
            (image_size, image_size),
            scale=(0.45, 1.0),
            ratio=(1.0, 1.0),
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])


def validation_transform(image_size):
    resize_size = int(image_size * 1.125)
    return v2.Compose([
        v2.Resize(resize_size, antialias=True),
        v2.CenterCrop((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])


def build_dataloaders(data_dir, image_size, batch_size, num_workers, validation_split, seed):
    paths = find_images(data_dir)
    if len(paths) < 2:
        return None, None, 0

    validation_count = max(1, int(len(paths) * validation_split))
    train_count = len(paths) - validation_count
    generator = torch.Generator().manual_seed(seed)
    train_paths, validation_paths = random_split(paths, [train_count, validation_count], generator=generator)

    train_dataset = ImageDataset(list(train_paths), train_transform(image_size))
    validation_dataset = ImageDataset(list(validation_paths), validation_transform(image_size))

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)
    return train_loader, validation_loader, len(paths)
