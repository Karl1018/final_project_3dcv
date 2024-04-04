import torch
import torch.nn as nn
from . import network

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_model, betas):
        super(GaussianDiffusion, self).__init__()
        self.denoise_model = denoise_model
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.noise_scale_factors = torch.sqrt(
            self.betas[1:] / self.alphas_bar[:-1]
        )

    def noise_generation(self, x, t):
        beta_t = self.betas.to(x.device)[t]
        alpha_bar_prev_t = torch.cat([torch.tensor([1.]).to(x.device), self.alphas_bar[:-1].to(x.device)])[t]
        noise = torch.randn_like(x) * torch.sqrt(beta_t / alpha_bar_prev_t)
        return noise

    def forward(self, x):
        t = torch.randint(len(self.betas), size=(1,)).to(x.device) # select timestep t
        noise = self.noise_generation(x, t)
        x_noisy = x + noise
        x_recon = self.denoise_model(x_noisy)
        return x_recon

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
