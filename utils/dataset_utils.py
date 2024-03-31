from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation

def make_public_dataloader(batch_size, num_workers=4, aug=False):
    if aug:
        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = Compose([
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = datasets.Places365(root='data', split='train-standard', download=True, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dataloader, test_dataloader

def make_custom_dataloader(path, batch_size, num_workers=4, aug=False):

    if aug:
        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = Compose([
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = datasets.ImageFolder(path, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dataloader, test_dataloader