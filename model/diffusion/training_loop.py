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
from utils.image_process import rgb_to_lab, lab_to_rgb, postprocess

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
    model = GaussianDiffusion(denoise_model, betas = torch.linspace(start=1e-4, end=2e-2, steps=1000, device=device)).to(device)

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
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=0.002)

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
        denoise_model.train()
        for i, (images, _) in enumerate(train_dataloader):
            train_dataloader.set_description(f'Epoch [{epoch+1}/{epochs}]')
            # Convert images to Lab and split the channels
            targets_ori = images.to(device)  # Real, colored images
  
            inputs = rgb_to_lab(targets_ori).to(device)
            ab_channels = inputs[:, 1:, :, :].to(device) 
            l_channel = inputs[:, 0, :, :].to(device)    
           
            # Clear the gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(ab_channels, l_channel)

            outputs = outputs.to(device)

            # Calculate the loss
            loss = criterion(outputs, ab_channels)
            # Backward propagation
            loss.backward()
            # Update the weights
            optimizer.step()

            # Add up the loss
            running_loss += loss.item()


            # Logging
            if i % interval == 0 and i != 0:
                t = time.time() - start_time
                content = f'Time: {int(t//3600)}h {int(t%3600//60)}m, ' f'Epoch [{epoch+1}/{epochs}], 'f'Loss: {loss}'
                print(content)
                with open('snapshot/log.txt', 'a') as f:
                    f.write(content + '\n')
                
            if i % interval == 0 and i != 0:
                # Save generated images
                l_channel = l_channel.unsqueeze(1) 
                outputs_lab = torch.cat((l_channel, outputs), dim=1)  # Combine L and ab channels

                outputs_lab = lab_to_rgb(outputs_lab)
        
                TF.to_pil_image(outputs_lab[0]).save(f'snapshot/train/generated_{epoch+1}.png')
                TF.to_pil_image(postprocess(targets_ori[0])).save(f'snapshot/train/real_{epoch+1}.png')
        # Test
        running_loss = 0.0
        denoise_model.eval()
        test_dataloader = tqdm.tqdm(test_dataloader)
        for i, (images, _) in enumerate(test_dataloader):
            test_dataloader.set_description(f'Testing Epoch [{epoch+1}/{epochs}]')
            targets_ori = images.to(device) 
            inputs_test = rgb_to_lab(targets_ori)
            ab_channels = inputs_test[:, 1:, :, :].to(device)
            l_channel = inputs_test[:, 0, :, :].to(device)  
            with torch.no_grad():
                # Generate noisy versions for inference
                output_test = model(ab_channels, l_channel) 
                loss = criterion(output_test, ab_channels)
                running_loss += loss.item()
        
        content = f'Test: ' f'Epoch [{epoch+1}/{epochs}], 'f'Loss: {running_loss/len(test_dataloader)}\n\n'
        print(content)
        with open('snapshot/log.txt', 'a') as f:
            f.write(content + '\n')

        # Save generated images
        l_channel = l_channel.unsqueeze(1)
        outputs_lab = torch.cat((l_channel, output_test), dim=1)
        outputs_rgb = lab_to_rgb(outputs_lab)
 
        TF.to_pil_image(outputs_rgb[0]).save(f'snapshot/test/generated_{epoch+1}.png')
        TF.to_pil_image(postprocess(targets_ori[0])).save(f'snapshot/test/real_{epoch+1}.png')

        checkpoint = {
        'denoise_model': denoise_model.state_dict(),
        'model': model.state_dict()
        }
        torch.save(checkpoint, f'snapshot/checkpoint/epoch_{epoch + 1}.pth')


    print('Finished Training')


