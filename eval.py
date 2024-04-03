import torch
import lpips
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    try:
        checkpoint = torch.load(model_path)
    except FileNotFoundError:
        print('Invalid checkpoint path')
        exit(1)
    # TODO: Change the model architecture
    from model.GAN import network
    generator = network.Generator()
    discriminator = network.Discriminator()
    generator.load_state_dict(checkpoint['generator'])
    return generator

def load_image(image_path, resize_to=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def evaluate(colorized_image_path, original_image_path):
    # Load LPIPS model for perceptual similarity
    lpips_model = lpips.LPIPS(net='alex')

    # Load images
    colorized_image = load_image(colorized_image_path)
    original_image = load_image(original_image_path)

    # Move to GPU if available
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
        colorized_image = colorized_image.cuda()
        original_image = original_image.cuda()

    # Calculate LPIPS
    lpips_score = lpips_model(colorized_image, original_image)
    print(f"LPIPS: {lpips_score.item()}")

    # Move tensors to CPU and convert for PSNR and SSIM
    colorized_image_np = colorized_image.squeeze().cpu().permute(1, 2, 0).numpy()
    original_image_np = original_image.squeeze().cpu().permute(1, 2, 0).numpy()

    # Calculate PSNR
    psnr = compare_psnr(original_image_np, colorized_image_np, data_range=1)
    print(f"PSNR: {psnr}")

    print(f'Colorized Image Shape: {colorized_image_np.shape}')
    print(f'Original Image Shape: {original_image_np.shape}')
    print(f'Color Image dtype: {colorized_image_np.dtype}')
    print(f'Origin Image dtype: {original_image_np.dtype}')
    print(f'Min pixel value of C: {colorized_image_np.min()}')
    print(f'Max pixel value of C: {colorized_image_np.max()}')
    print(f'Min pixel value of O: {original_image_np.min()}')
    print(f'Max pixel value of O: {original_image_np.max()}')

    # Calculate SSIM
    ssim = compare_ssim(original_image_np, colorized_image_np, multichannel=True, data_range=1.0, channel_axis=2)
    print(f"SSIM: {ssim}")

if __name__ == "__main__":
    colorized_image_path = 'origin.jpg'  # Update this path
    original_image_path = 'reconstruction.jpg'  # Update this path
    evaluate(colorized_image_path, original_image_path)
