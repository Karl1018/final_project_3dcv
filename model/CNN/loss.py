import torch.nn as nn

# Define your loss function
class ColorizationLoss(nn.Module):
    def __init__(self):
        super(ColorizationLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)