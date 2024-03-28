import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale

#TODO: Augmentations
def make_public_dataloader(batch_size, num_workers=4):

    transform = Compose([
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.Places365(root='data', split='train-standard', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

# custom dataloader with no labels
def make_custom_dataloader(path, batch_size, num_workers=4):

    transform = Compose([
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader