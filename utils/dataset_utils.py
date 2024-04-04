from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from utils.image_process import aug_transform, basic_transform

def make_public_dataloader(batch_size, num_workers=4, aug=False):

    dataset = datasets.Places365(root='data', split='train-standard', download=True, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if aug:
        train_dataset.dataset.transform = aug_transform
    else:
        train_dataset.dataset.transform = basic_transform

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader

def make_custom_dataloader(path, batch_size, num_workers=4, aug=False):
    
    dataset = datasets.ImageFolder(path)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    if aug:
        train_dataset.dataset.transform = aug_transform
    else:
        train_dataset.dataset.transform = basic_transform
    test_dataset.dataset.transform = basic_transform
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, test_dataloader