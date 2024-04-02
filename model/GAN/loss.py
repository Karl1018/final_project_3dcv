import torch
from torch import nn
from torchvision.models import vgg16

# Initialize loss functions
criterionGAN = nn.BCELoss()
criterionL1 = nn.SmoothL1Loss()

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
def generator_loss(discriminator, fake_images, real_images, lambda_L1=100.0, lambda_perceptual=0.1):
    # Adversarial loss: how well can it fool the discriminator?
    target = torch.ones_like(discriminator(fake_images), device=fake_images.device)  # Target label: 1 (as if fake are real)
    loss_GAN = criterionGAN(discriminator(fake_images), target)
    
    # Content (L1) loss: how close is the generated image to the real one
    loss_L1 = criterionL1(fake_images, real_images)

    # Perceptual loss
    loss_perceptual = PerceptualLoss(fake_images, real_images, device=fake_images.device)
    
    # Total generator loss
    loss_G = loss_GAN + lambda_L1 * loss_L1 + lambda_perceptual * loss_perceptual
    return loss_G

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load a pre-trained VGG-16 model and configure it for feature extraction
        self.vgg = vgg16(pretrained=True).features
        # Freeze the model to prevent any further training or changes to the weights
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated_image, target_image):
        # Normalize the images to fit VGG's expected input distribution
        generated_image = self._preprocess(generated_image)
        target_image = self._preprocess(target_image)
        # Extract features from both images
        features_generated = self.vgg(generated_image)
        features_target = self.vgg(target_image)
        # Calculate the perceptual loss
        loss = torch.mean((features_generated - features_target) ** 2)
        return loss

    def _preprocess(self, image):
        # Normalize using VGG's expected mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (image - mean) / std
