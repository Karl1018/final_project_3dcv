# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _downsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

def _upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.down2 = _downsampling_block(64, 128)
        self.down3 = _downsampling_block(128, 256)
        self.down4 = _downsampling_block(256, 512)

        # Decoder with skip connections
        self.up1 = _upsampling_block(512, 256)
        self.up2 = _upsampling_block(256*2, 128)  # *2 due to skip connection
        self.up3 = _upsampling_block(128*2, 64)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))
        return u4

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsampling
        self.conv2 = _downsampling_block(64, 128)
        self.conv3 = _downsampling_block(128, 256)
        self.conv4 = _downsampling_block(256, 512)
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Use Sigmoid for binary classification
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    x = torch.randn((1, 1, 256, 256), device=device)
    gen = Generator()
    gen.to(device)
    fake = gen(x)
    assert fake.shape == torch.Size([1, 3, 256, 256])
    print("Generator test passed")

    disc = Discriminator()
    disc.to(device)
    print(disc(fake))
if __name__ == "__main__":
    test()