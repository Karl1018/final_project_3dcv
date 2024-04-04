import torch
from torch import optim
from torch import nn
import torchvision.transforms.functional as TF
import os
import time
import tqdm
import random
import numpy as np

from . import network
from . import loss
from .config import *
from utils.image_process import postprocess

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

setup_seed(17)

generator = network.Generator().to(device)
discriminator = network.Discriminator().to(device)

# Hyperparameters
lr = LR
beta1 = BETA1

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

def train(train_dataloader, test_dataloader, resume, epochs, interval):
    if resume:
        try:
            checkpoint = torch.load(resume)
        except FileNotFoundError:
            print('Invalid checkpoint path')
            exit(1)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        print(f'Resuming training from {resume}')
    else:
        print('Starting training from scratch')

    print('Training use device:', device)
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
    generator_state = generator.state_dict()
    discriminator_state = discriminator.state_dict()

    models_state_dict = {
        'generator': generator_state,
        'discriminator': discriminator_state
    }

    torch.save(models_state_dict, f'snapshot/checkpoint/checkpoint_0.pth')

    # Save sample images
    sample = next(iter(train_dataloader))
    real_image = sample[0].to(device)
    grayscale_image = TF.rgb_to_grayscale(real_image)
    print(grayscale_image.shape)
    fake_image = generator(grayscale_image)

    TF.to_pil_image(fake_image.cpu()[0]).save(f'snapshot/generated_0.png')
    TF.to_pil_image(real_image.cpu()[0]).save(f'snapshot/real_0.png')
    TF.to_pil_image(grayscale_image.cpu()[0]).save(f'snapshot/grayscale_0.png')

    start_time = time.time()

    for epoch in range(epochs):
        # Train
        sum_loss_D = 0
        sum_loss_G = 0
        train_dataloader = tqdm.tqdm(train_dataloader)
        for i, (images, _) in enumerate(train_dataloader):
            train_dataloader.set_description(f'Epoch [{epoch+1}/{epochs}]')

            # Convert images to grayscale for generator input
            real_images = images.to(device)  # Real, colored images
            grayscale_images = TF.rgb_to_grayscale(real_images)

            # Training the discriminator with real and fake images
            optimizer_D.zero_grad()
            fake_images = generator(grayscale_images)
            loss_D = loss.discriminator_loss(discriminator, real_images, fake_images)
            loss_D.backward()
            optimizer_D.step()

            # Training the generator
            optimizer_G.zero_grad()
            fake_images = generator(grayscale_images)
            loss_G = loss.generator_loss(discriminator, fake_images, real_images)
            loss_G.backward()
            optimizer_G.step()

            sum_loss_D += loss_D.item()
            sum_loss_G += loss_G.item()
            
            # Logging
            if i % interval == 0:
                t = time.time() - start_time
                content = f'Time: {int(t//3600)}h {int(t%3600//60)}m, ' f'Epoch [{epoch+1}/{epochs}], 'f'Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}'
                print(content)
                with open('snapshot/log.txt', 'a') as f:
                    f.write(content + '\n')
                
            if i % interval == 0:
                # Save generated images
                TF.to_pil_image(postprocess(fake_images[0]).cpu()).save(f'snapshot/train/generated_{epoch+1}.png')
                TF.to_pil_image(postprocess(real_images[0]).cpu()).save(f'snapshot/train/real_{epoch+1}.png')
                TF.to_pil_image(postprocess(grayscale_images[0]).cpu()).save(f'snapshot/train/grayscale_{epoch+1}.png')

        loss_D = sum_loss_D / len(train_dataloader)
        loss_G = sum_loss_G / len(train_dataloader)
        content = f'Epoch [{epoch+1}/{epochs}] finished, 'f'Average Loss_D: {loss_D}, Average Loss_G: {loss_G}\n'
        print(content)
        with open('snapshot/log.txt', 'a') as f:
            f.write(content + '\n')
        # Test
        test_dataloader = tqdm.tqdm(test_dataloader)
        for i, (images, _) in enumerate(test_dataloader):
            test_dataloader.set_description(f'Testing Epoch [{epoch+1}/{epochs}]')
            real_images = images.to(device)
            grayscale_images = TF.rgb_to_grayscale(real_images)
            sum_loss_D = 0
            sum_loss_G = 0
            with torch.no_grad():
                fake_images = generator(grayscale_images)
                loss_D = loss.discriminator_loss(discriminator, real_images, fake_images)
                loss_G = loss.generator_loss(discriminator, fake_images, real_images)
                sum_loss_D += loss_D.item()
                sum_loss_G += loss_G.item()
        loss_D = sum_loss_D / len(test_dataloader)
        loss_G = sum_loss_G / len(test_dataloader)

        t = time.time() - start_time
        content = f'Test: ' f'Epoch [{epoch+1}/{epochs}], 'f'Average Loss_D: {loss_D.item()}, Average Loss_G: {loss_G.item()}\n'
        content += "--------------------------------------------\n"
        print(content)
        with open('snapshot/log.txt', 'a') as f:
            f.write(content + '\n')
        
        # Save generated images
        TF.to_pil_image(postprocess(fake_images[0]).cpu()).save(f'snapshot/test/generated_{epoch}.png')
        TF.to_pil_image(postprocess(real_images[0]).cpu()).save(f'snapshot/test/real_{epoch}.png')
        TF.to_pil_image(postprocess(grayscale_images[0]).cpu()).save(f'snapshot/test/grayscale_{epoch}.png')

        # save model state every epoch
        generator_state = generator.state_dict()
        discriminator_state = discriminator.state_dict()

        models_state_dict = {
            'generator': generator_state,
            'discriminator': discriminator_state
        }

        torch.save(models_state_dict, f'snapshot/checkpoint/checkpoint_{epoch+1}.pth')