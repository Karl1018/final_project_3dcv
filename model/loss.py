import torch
from torch import nn

# Initialize loss functions
criterionGAN = nn.BCELoss()  # Adversarial loss for both generator and discriminator
criterionL1 = nn.SmoothL1Loss()    # Content loss (smooth L1 loss)

# Adversarial loss for the discriminator
def discriminator_loss(discriminator, real_images, fake_images):
    # Real images
    real_target = torch.ones_like(discriminator(real_images), device=real_images.device)  # Target label: 1
    loss_real = criterionGAN(discriminator(real_images), real_target)
    
    # Fake images
    fake_target = torch.zeros_like(discriminator(fake_images.detach()), device=fake_images.device)  # Target label: 0
    loss_fake = criterionGAN(discriminator(fake_images.detach()), fake_target)
    
    # Total discriminator loss
    loss_D = (loss_real + loss_fake) * 0.5
    return loss_D

# Generator loss
def generator_loss(discriminator, fake_images, real_images, lambda_L1=100.0):
    # Adversarial loss: how well can it fool the discriminator?
    target = torch.ones_like(discriminator(fake_images), device=fake_images.device)  # Target label: 1 (as if fake are real)
    loss_GAN = criterionGAN(discriminator(fake_images), target)
    
    # Content (L1) loss: how close is the generated image to the real one
    loss_L1 = criterionL1(fake_images, real_images)
    
    # Total generator loss
    loss_G = loss_GAN + lambda_L1 * loss_L1  # The 100 is a hyperparameter that balances GAN and L1 loss
    return loss_G
