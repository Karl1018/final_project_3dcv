import numpy as np
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=0.0):
        super(UNetBlock, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
            
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.dropout:
            self.do = nn.Dropout2d(self.dropout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        print("input: ", x.shape)
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.do(x)
        x = self.relu(x)
        print("output: ", x.shape, "\n\n")
        return x

class Generator(nn.Module):
    # GAN generator model based on U-Net architecture
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            UNetBlock(1, 64, down=True, stride=1, padding='same'),
            UNetBlock(64, 64, down=True),
            UNetBlock(64, 128, down=True),
            UNetBlock(128, 256, down=True),
            UNetBlock(256, 512, down=True),
            UNetBlock(512, 512, down=True),
            UNetBlock(512, 512, down=True),
            UNetBlock(512, 512, down=True),

        )

        self.decoder = nn.Sequential(
            UNetBlock(512, 512, down=False),
            UNetBlock(512, 512, down=False),
            UNetBlock(512, 512, down=False),
            UNetBlock(512, 256, down=False),
            UNetBlock(256, 128, down=False),
            UNetBlock(128, 64, down=False),
            UNetBlock(64, 64, down=False),
        )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        # skip connection
        # Encoder
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x.clone())
        # Decoder
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat((x, encoder_features[-(idx + 2)]), 1)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    # GAN discriminator model based on PatchGAN architecture
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
def test():
    x = torch.randn((1, 1, 256, 256))
    gen = Generator()
    disc = Discriminator()
    assert gen(x).shape == x.shape

if __name__ == "__main__":
    test()