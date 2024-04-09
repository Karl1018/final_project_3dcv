import torch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, Grayscale, Resize
from skimage import color
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


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

transform_grayscale = Compose([

    Grayscale(num_output_channels=3),  # converts to 3-channel grayscale
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    Resize((256, 256))
])

def postprocess(tensor):

    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).type(torch.uint8)
    
    return tensor

def rgb_to_lab(images: torch.Tensor):
    
    # Ensure images tensor is in the range [0, 1]
    images = (images + 1) / 2
    if images.dim() == 3:
        images = images.unsqueeze(0)
    # Convert from torch Tensor to numpy array, and reshape to (H,W,C) for skimage
    images_np = images.permute(0, 2, 3, 1).cpu().detach().numpy()

    # Prepare array to hold lab_images
    lab_images = np.zeros(images_np.shape)

    # Iterate over the images and convert each image to Lab space
    for i in range(images_np.shape[0]):
        lab_images[i] = color.rgb2lab(images_np[i])

    # Convert back to torch Tensor from numpy array, and reshaping to (B,C,H,W)
    lab_images = torch.from_numpy(lab_images).permute(0, 3, 1, 2).float()

    # Normalize L channel to [0, 1]
    lab_images[:, 0, :, :] = lab_images[:, 0, :, :] / 100.0

    # Normalize a and b channels to [-1, 1]
    lab_images[:, 1:, :, :] = (lab_images[:, 1:, :, :] ) / 127.0
    # Return the images in Lab space
    return lab_images


def lab_to_rgb(images):
    images = denormalize_lab(images)
    if images.dim() == 3:
        images = images.unsqueeze(0)
    lab_batch_np = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    # Initialize an empty array for RGB images 
    rgb_batch_np = np.zeros_like(lab_batch_np, dtype=np.float32)
    # Convert each image in the batch from L*a*b* to RGB 
    for i in range(lab_batch_np.shape[0]): 
        rgb_batch_np[i] = color.lab2rgb(lab_batch_np[i])
    # Convert to uint8 
    rgb_batch_np = (rgb_batch_np * 255).astype(np.uint8) 
    # permute the dimensions to (B,C,H,W)
    # rgb_batch_np = torch.from_numpy(rgb_batch_np).permute(0, 3, 1, 2).float()
    return rgb_batch_np


def denormalize_lab(images):
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Denormalize L channel to [0, 100]
    images[:, 0, :, :] = images[:, 0, :, :] * 100.0

    # Denormalize a and b channels to [-127, 127]
    images[:, 1:, :, :] = images[:, 1:, :, :] *  127.0
    return images


def test():
    input_image = Image.open('real_0.png')
    input_image = basic_transform(input_image)
    input_image = rgb_to_lab(input_image)
    TF.to_pil_image(input_image.squeeze(0)).save('lab.png')
    input_image = lab_to_rgb(input_image)
    TF.to_pil_image(input_image.squeeze(0)).save('rgb.png')

if __name__ == "__main__":
    test()