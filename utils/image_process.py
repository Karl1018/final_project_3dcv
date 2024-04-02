import torch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation

aug_transform = Compose([
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

basic_transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def postprocess(tensor):

    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).type(torch.uint8)
    
    return tensor
