from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter

#TODO: Augmentations
def make_public_dataloader(batch_size, num_workers=4, aug = False):
    if aug:
        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = Compose([
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = datasets.Places365(root='data', split='train-standard', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

# custom dataloader with no labels
def make_custom_dataloader(path, batch_size, num_workers=4, aug=False):

    if aug:
        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = Compose([
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader