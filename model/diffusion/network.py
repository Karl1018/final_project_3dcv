import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), # 64
            nn.ReLU(True),
            nn.MaxPool2d(2), # 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)  # 16
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), # 32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), # 64
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e = self.encoder(x)
        m = self.middle(e)
        return self.decoder(m)

def test():
    device = torch.device("mps")
    print(f"Using device {device}")
    x = torch.randn((1, 1, 256, 256), device=device)
    unet = UNet()
    unet.to(device)
    fake = unet(x)
    assert fake.shape == torch.Size([1, 3, 256, 256])
    print("unet test passed")

if __name__ == "__main__":
    test()