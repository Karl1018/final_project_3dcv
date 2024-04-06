import torch
import numpy as np
from torch import optim
import torchvision.transforms.functional as TF
from skimage.color import lab2rgb
from skimage import color
import tqdm
import os
import sys
import time

import network
import loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.image_process import postprocess  

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = network.ColorizationNet().to(device)
criterion = loss.ColorizationLoss()

# Hyperparameters
lr = 0.001

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=lr)

def print_tensor_stats(tensor, name):
    print(f"{name}:")
    print(f" - Shape: {tensor.shape}")  
    print(f" - Max: {tensor.max().item()}")
    print(f" - Min: {tensor.min().item()}")
    print(f" - Mean: {tensor.mean().item()}")
    print(f" - Std: {tensor.std().item()}")

def merge_l_ab(l_channel, ab_channels):
    """
    Merges L channel with AB channels to form a LAB image and converts it to RGB.
    
    Parameters:
        l_channel: PyTorch tensor, shape: [N, 1, H, W], the L channel of LAB space.
        ab_channels: PyTorch tensor, shape: [N, 2, H, W], the AB channels predicted by the model.
        
    Returns:
        rgb_images: PyTorch tensor, shape: [N, 3, H, W], the converted RGB images.
    """
    l_channel_scaled = l_channel * 100
    ab_channels_scaled = ab_channels * 127 
    
    lab_images = torch.cat([l_channel_scaled, ab_channels_scaled], dim=1)
    lab_images_np = lab_images.permute(0, 2, 3, 1).cpu().detach().numpy()
    
    # Convert LAB to RGB using skimage
    rgb_images_np = [lab2rgb(lab_img) for lab_img in lab_images_np]
    rgb_images = torch.from_numpy(np.stack(rgb_images_np, axis=0)).float().permute(0, 3, 1, 2).to(ab_channels.device)
    
    return rgb_images

def extract_ab_channels(real_images):
    """
    Converts real RGB image batches to AB channels in Lab space.
    
    parameter:
        real_images: PyTorch tensor, shape: [N, 3, H, W],
    return:
        real_images_ab_tensor: PyTorch tensor, shape:[N, 2, H, W],
    """
    real_images_np = real_images.permute(0, 2, 3, 1).cpu().numpy()
    real_images_lab = color.rgb2lab(real_images_np)
    real_images_ab = real_images_lab[:, :, :, 1:3]
    real_images_ab_tensor = torch.from_numpy(real_images_ab).float().to(real_images.device)
    real_images_ab_tensor = real_images_ab_tensor.permute(0, 3, 1, 2)

    return real_images_ab_tensor

def train(train_dataloader, test_dataloader, resume, epochs, interval):
    if resume:
        try:
            checkpoint = torch.load(resume)
        except FileNotFoundError:
            print('Invalid checkpoint path')
            exit(1)
        model.load_state_dict(checkpoint['model'])
        print(f'Resuming training from {resume}')
    else:
        print('Starting training from scratch')

    print('Starting training...')
    print('Training on device:', device)
    print('Training data size:', len(train_dataloader.dataset))
    print('Training batch size:', train_dataloader.batch_size)

    # Create snapshot directory
    os.makedirs('snapshot/checkpoint', exist_ok=True)
    os.makedirs('snapshot/train', exist_ok=True)
    os.makedirs('snapshot/test', exist_ok=True)

    # Overwrite log file
    with open('snapshot/log.txt', 'w') as f:
        f.write('')
    
    # Save initial model state
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'snapshot/checkpoint/checkpoint_0.pth')

    start_time = time.time()

    for epoch in range(epochs):     
        # Train
        model.train()
        running_loss = 0.0
        for i, (images, _) in enumerate(tqdm.tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
            
            # Convert images to grayscale for CNN input
            real_images = images.to(device)  
            grayscale_images = TF.rgb_to_grayscale(real_images)
            grayscale_images = (grayscale_images + 1.0) / 2.0
            
            outputs_ab = model(grayscale_images)
            real_images_ab = extract_ab_channels(real_images) / 128.0
            
            loss_CNN = criterion(outputs_ab, real_images_ab)

            optimizer.zero_grad()
            loss_CNN.backward()
            optimizer.step()

            running_loss += loss_CNN.item()

            # Logging
            if i % interval == 0:
                t = time.time() - start_time
                content = f'Time: {int(t//3600)}h {int(t%3600//60)}m, ' f'Epoch [{epoch+1}/{epochs}],' f'Loss: {loss_CNN.item():.4f}'
                print(content)
                with open('snapshot/log.txt', 'a') as f:
                    f.write(content + '\n')

            if i % interval == 0:
                
                print_tensor_stats((grayscale_images), "Grayscale Images")
                print_tensor_stats((real_images), "Real Images")
                print_tensor_stats((real_images_ab), "Real Images_ab")
                print_tensor_stats((real_images_ab * 127), "Real Images_ab_without_normalized")
                print_tensor_stats((outputs_ab), "Model Outputs_ab")
                print_tensor_stats((outputs_ab * 127), "Model Outputs_ab_without_normalized")
                
                # convert output_ab to rgb image
                l_channel = grayscale_images.expand(-1, 3, -1, -1)  # Duplicate the grayscale channel to match LAB dimension
                outputs = merge_l_ab(l_channel[:, :1, :, :], outputs_ab)  # Use only the first channel for L
                print_tensor_stats((outputs), "Model RGB Outputs")

                # save sample image
                TF.to_pil_image(postprocess(outputs[0]).cpu()).save(f'snapshot/train/generated_{epoch}_{i}.png')
                TF.to_pil_image(postprocess(real_images[0]).cpu()).save(f'snapshot/train/real_{epoch}_{i}.png')
                TF.to_pil_image(postprocess(grayscale_images[0]).cpu()).save(f'snapshot/train/grayscale_{epoch}_{i}.png')
        
        # Test
        test_dataloader = tqdm.tqdm(test_dataloader)
        for i, (images, _) in enumerate(test_dataloader):
            test_dataloader.set_description(f'Testing Epoch [{epoch+1}/{epochs}]')
            real_images = images.to(device)
            grayscale_images = TF.rgb_to_grayscale(real_images)
            grayscale_images = (grayscale_images + 1.0) / 2.0
            
            with torch.no_grad():
                outputs_ab = model(grayscale_images)
                real_images_ab = extract_ab_channels(real_images) / 128.0
                
                loss_CNN = criterion(outputs_ab, real_images_ab)
                
        t = time.time() - start_time
        content = f'Test: ' f'Epoch [{epoch+1}/{epochs}], ' f'Loss: {loss_CNN.item():.4f}'
        print(content)
        with open('snapshot/log.txt', 'a') as f:
            f.write(content + '\n')
        
        # convert output_ab to rgb image
        l_channel = grayscale_images.expand(-1, 3, -1, -1)  # Duplicate the grayscale channel to match LAB dimension
        outputs = merge_l_ab(l_channel[:, :1, :, :], outputs_ab)  # Use only the first channel for L
            
        # save sample image
        TF.to_pil_image(postprocess(outputs[0]).cpu()).save(f'snapshot/test/generated_{epoch}_{i}.png')
        TF.to_pil_image(postprocess(real_images[0]).cpu()).save(f'snapshot/test/real_{epoch}_{i}.png')
        TF.to_pil_image(postprocess(grayscale_images[0]).cpu()).save(f'snapshot/test/grayscale_{epoch}_{i}.png')
            
        avg_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        torch.save(model.state_dict(), f'snapshot/checkpoint/checkpoint_{epoch+1}.pth')