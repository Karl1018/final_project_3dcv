import torch
from torch import optim
from torch import nn
import torchvision.transforms.functional as TF
import tqdm
import os
import time

from utils import transforms
from . import network
from . import loss

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


generator = network.Generator().to(device)
discriminator = network.Discriminator().to(device)

# Hyperparameters
lr = 0.0002
beta1 = 0.5

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

def train(dataloader, resume, epochs, interval):
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
    print('Training data size:', len(dataloader.dataset))
    print('Training batch size:', dataloader.batch_size)

    # Generate sample images and record the initial model state
    os.makedirs('snapshot/checkpoint', exist_ok=True)
    generator_state = generator.state_dict()
    discriminator_state = discriminator.state_dict()

    models_state_dict = {
        'generator': generator_state,
        'discriminator': discriminator_state
    }

    torch.save(models_state_dict, f'snapshot/checkpoint/checkpoint_0.pth')

    sample = next(iter(dataloader))
    real_image = sample[0].to(device)
    grayscale_image = TF.rgb_to_grayscale(real_image)
    fake_image = generator(grayscale_image)

    TF.to_pil_image(fake_image.cpu()[0]).save(f'snapshot/generated_0.png')
    TF.to_pil_image(real_image.cpu()[0]).save(f'snapshot/real_0.png')
    TF.to_pil_image(grayscale_image.cpu()[0]).save(f'snapshot/grayscale_0.png')

    start_time = time.time()

    for epoch in range(epochs):
        dataloader = tqdm.tqdm(dataloader)
        for i, (images, _) in enumerate(dataloader):
            dataloader.set_description(f'Epoch [{epoch+1}/{epochs}]')

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
            
            # Logging
            if i % interval == 0:
                t = time.time() - start_time
                content = f'Time: {t//3600}h {t%3600//60}m, ' f'Epoch [{epoch+1}/{epochs}], 'f'Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}'
                print(content)
                with open('snapshot/log.txt', 'w') as f:
                    f.write(content)
                
            if i % interval == 0:
                generator_state = generator.state_dict()
                discriminator_state = discriminator.state_dict()

                models_state_dict = {
                    'generator': generator_state,
                    'discriminator': discriminator_state
                }

                torch.save(models_state_dict, f'snapshot/checkpoint/checkpoint_{epoch}.pth')
                # Save generated images
                TF.to_pil_image(fake_images[0].cpu()).save(f'snapshot/generated_{epoch}.png')
                TF.to_pil_image(real_images[0].cpu()).save(f'snapshot/real_{epoch}.png')
                TF.to_pil_image(grayscale_images[0].cpu()).save(f'snapshot/grayscale_{epoch}.png')