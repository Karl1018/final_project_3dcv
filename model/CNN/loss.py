import torch.nn as nn

criterion = nn.MSELoss()

def loss_CNN(generate_image, real_image):
    loss = criterion(generate_image, real_image)
    return loss