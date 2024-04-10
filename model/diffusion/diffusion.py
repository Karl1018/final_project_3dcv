import torch
import torch.nn as nn
from . import network
from utils.image_process import lab_to_rgb, denormalize_lab
import torchvision.transforms.functional as TF

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_model, betas):
        super(GaussianDiffusion, self).__init__()
        self.denoise_model = denoise_model
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.noise_scale_factors = torch.sqrt(self.betas / (1 - self.alphas_bar))

    def noise_generation(self, x, t):
        noise = torch.randn_like(x, device=x.device)
        noise_scale_factors = self.noise_scale_factors.to(x.device)
        noise = noise * noise_scale_factors[t]
        return noise

    def forward(self, x_ab, x_l):
        t = torch.randint(len(self.betas), size=(1,)).to(x_ab.device) # select timestep t
        noise = self.noise_generation(x_ab, t)
        x_noisy = torch.cat((x_l.unsqueeze(1), x_ab + noise), dim=1)
        x_recon = self.denoise_model(x_noisy)
        return x_recon
    
    def reverse_diffusion(self, x):
        l_channel = x[:, 0, :, :].unsqueeze(1).to(x.device)
        ab_channel = torch.randn((l_channel.shape[0], 2, l_channel.shape[2], l_channel.shape[3])).to(x.device)
        x = torch.cat((l_channel, ab_channel), dim=1)
        return torch.cat((l_channel, self.denoise_model(x)), dim=1)
    

def test():
    device = torch.device("mps")
    print(f"Using device {device}")
    x = torch.randn((1, 3, 256, 256), device=device)
    denoise_model = network.UNet().to(device)
    diffusion_model = GaussianDiffusion(denoise_model, betas=torch.linspace(0.1, 0.2, 2)).to(device)
    fake = diffusion_model(x)
    assert fake.shape == torch.Size([1, 2, 256, 256]), "Expected output shape [1, 2, 256, 256], but got " + str(fake.shape)
    print("diffusion test passed")

if __name__ == "__main__":
    test()
