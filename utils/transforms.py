from torchvision import transforms

grayscale_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize grayscale images
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize color images
])