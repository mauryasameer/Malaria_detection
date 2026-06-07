# malaria_detection/dataset.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGE_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_datasets(data_dir: str, val_split: float = 0.2, seed: int = 42):
    transform = build_transforms()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset, full_dataset.classes


def build_dataloaders(train_dataset, val_dataset, batch_size: int = 32, num_workers: int = 2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
