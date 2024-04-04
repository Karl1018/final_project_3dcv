# Initialization
import torch
import torch.nn as nn
import tqdm
import time
import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from model.diffusion.network import UNet
from model.diffusion.diffusion import GaussianDiffusion
from utils.dataset_utils import make_custom_dataloader
from utils.image_process import rgb_to_lab, lab_to_rgb, denormalize_lab

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set the device for training, use MPS if available else use CUDA or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(train_dataloader, test_dataloader, resume, epochs, interval):

    # Load the denoising model (UNet)
    denoise_model = UNet().to(device)
    # Initialize the Gaussian Diffusion model
    model = GaussianDiffusion(denoise_model, betas = torch.linspace(start=1e-4, end=2e-2, steps=1000)).to(device)

    if resume:
        try:
            checkpoint = torch.load(resume)
        except FileNotFoundError:
            print('Invalid checkpoint path')
            exit(1)  
        denoise_model.load_state_dict(checkpoint['denoise_model']) # load the parameters
        model.load_state_dict(checkpoint['model'])
        print(f'Resuming training from {resume}')
    else:
        print('Starting training from scratch')

    # Set the loss function and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Print the training information
    print('Training using device:', device)
    print('Training data size:', len(train_dataloader.dataset))
    print('Training batch size:', train_dataloader.batch_size)

    # Create snapshot directory
    os.makedirs('snapshot/checkpoint', exist_ok=True)
    os.makedirs('snapshot/train', exist_ok=True)
    os.makedirs('snapshot/test', exist_ok=True)

    # Overwrite log file
    with open('snapshot/log.txt', 'w') as f:
        f.write('')

    # Save sample images
    sample = next(iter(train_dataloader))
    real_image = sample[0].to(device)
    real_image = rgb_to_lab(real_image)
    # fake_image = model(real_image)
    TF.to_pil_image(real_image.cpu()[0]).save(f'snapshot/real_0.png')

    # Record the start time
    start_time = time.time()

    # Start the training loop
    for epoch in range(epochs):
        train_dataloader = tqdm.tqdm(train_dataloader)
        
        # Initialize the running loss
        running_loss = 0.0

        for i, (images, _) in enumerate(train_dataloader):
            train_dataloader.set_description(f'Epoch [{epoch+1}/{epochs}]')
            # Convert images to Lab and split the channels
            targets_ori = images.to(device)  # Real, colored images
  
            targets = rgb_to_lab(targets_ori)
            ab_channels = targets[:, 1:, :, :]  # ab channels as targets
            l_channel = targets[:, 0, :, :]  
            # Assume data is in the form (inputs, targets)
            inputs = targets.to(device)
            # Clear the gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(inputs)

            outputs = outputs.to(device)
            ab_channels = ab_channels.to(device)

            # Calculate the loss
            loss = criterion(outputs, ab_channels)
            # Backward propagation
            loss.backward()
            # Update the weights
            optimizer.step()

            # Add up the loss
            running_loss += loss.item()


            # Logging
            if i % interval == 0:
                t = time.time() - start_time
                content = f'Time: {int(t//3600)}h {int(t%3600//60)}m, ' f'Epoch [{epoch+1}/{epochs}], 'f'Loss: {loss}'
                print(content)
                with open('snapshot/log.txt', 'a') as f:
                    f.write(content + '\n')
                
            if i % interval == 0:
                # Save generated images
                l_channel = l_channel.to(device).unsqueeze(1) 
                outputs_lab = torch.cat((l_channel, outputs), dim=1)  # Combine L and ab channels

                outputs_lab = denormalize_lab(outputs_lab)
                targets = denormalize_lab(targets)

                outputs_lab = lab_to_rgb(outputs_lab)
                targets = lab_to_rgb(targets)
                
                TF.to_pil_image(outputs_lab[0]).save(f'snapshot/train/generated_{epoch+1}.png')
                TF.to_pil_image(targets[0]).save(f'snapshot/train/real_{epoch+1}.png')
        # Test
        test_dataloader = tqdm.tqdm(test_dataloader)
        for i, (images, _) in enumerate(test_dataloader):
            test_dataloader.set_description(f'Testing Epoch [{epoch+1}/{epochs}]')
            targets_ori = images.to(device) 
            targets = rgb_to_lab(targets_ori)
            ab_channels = targets[:, 1:, :, :] 
            l_channel = targets[:, 0, :, :]  
            # Assume data is in the form (inputs, targets)
            inputs = targets.to(device)
            ab_channels = ab_channels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, ab_channels)
        
        content = f'Test: ' f'Epoch [{epoch+1}/{epochs}], 'f'Loss: {loss.item()}\n\n'
        print(content)
        with open('snapshot/log.txt', 'a') as f:
            f.write(content + '\n')

        # Save generated images
        l_channel = l_channel.to(device).unsqueeze(1)
        outputs_lab = torch.cat((l_channel, outputs), dim=1)
        outputs_lab = denormalize_lab(outputs_lab)
        targets = denormalize_lab(targets)
        outputs_lab = lab_to_rgb(outputs_lab)
        targets = lab_to_rgb(targets)
        TF.to_pil_image(outputs_lab[0]).save(f'snapshot/test/generated_{epoch+1}.png')
        TF.to_pil_image(targets[0]).save(f'snapshot/test/real_{epoch+1}.png')

        checkpoint = {
        'denoise_model': denoise_model.state_dict(),
        'model': model.state_dict()
        }
        torch.save(checkpoint, f'snapshot/checkpoint/epoch_{epoch + 1}.pth')


    print('Finished Training')


