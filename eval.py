import torch
import lpips
import tqdm
import torchvision.transforms.functional as TF
import os
import sys

from pathlib import Path
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms
from PIL import Image
from utils.image_process import lab_to_rgb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.CNN.training_loop import merge_l_ab  
from utils.image_process import postprocess, transform_grayscale, rgb_to_lab  

def load_model(model_path, model_type):
    """
    Load the model.
    
    parameter:
        model_path: the path of trained model,
        model_type: the model using to evaluate
    return:
        model
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print('Invalid checkpoint path')
        exit(1)
    
    # upload model
    if model_type == 'cnn':
        from model.CNN import network
        model = network.ColorizationNet()
        model.load_state_dict(checkpoint)
    elif model_type == 'gan':
        from model.GAN import network
        model = network.Generator()
        model.load_state_dict(checkpoint['generator'])
    elif model_type == 'diffusion':
        from model.diffusion import network
        from model.diffusion import diffusion
        model = network.UNet()
        denoise_model = network.UNet().to(device)
        diffusion_model = diffusion.GaussianDiffusion(denoise_model, betas=torch.linspace(start=1e-4, end=2e-2, steps=1000)).to(device)
        denoise_model.load_state_dict(checkpoint['denoise_model'])
        diffusion_model.load_state_dict(checkpoint['model'])
        return diffusion_model

    model = model.to(device)
    model.eval()
    return model

def load_image(image_path, resize_to=(256, 256)):
    """
    Load the image and resize.
    
    parameter:
        image_path
    return:
        image
    """
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def evaluate(colorized_image_path, original_image_path):
    """
    Evaluate the score of generated images from our model.
    
    parameter:
        colorized_image_path: the path of generated images,
        original_image_path: the path of real images,
    return:
        LPIPS score
        PSNR score
        SSIM score
    """
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
    # print(f"LPIPS: {lpips_score.item()}")

    # Move tensors to CPU and convert for PSNR and SSIM
    colorized_image_np = colorized_image.squeeze().cpu().permute(1, 2, 0).numpy()
    original_image_np = original_image.squeeze().cpu().permute(1, 2, 0).numpy()

    # Calculate PSNR
    psnr_score = compare_psnr(original_image_np, colorized_image_np, data_range=1)
    # print(f"PSNR: {psnr_score}")

    # Calculate SSIM
    ssim_score = compare_ssim(original_image_np, colorized_image_np, multichannel=True, data_range=1.0, channel_axis=2)
    # print(f"SSIM: {ssim_score}")

    return lpips_score, psnr_score, ssim_score

def evaluate_folder(images_folder, model_type):
    """
    Evaluate the score of generated images from our model.
    
    parameter:
        images_folder,
        model_type: model using to evaluate
    """
    folder_path = Path(images_folder)

    generated_images = sorted(folder_path.glob('generated_*.png'))

    # initial all score
    total_lpips = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for generated_image_path in generated_images:
        # Extract epoch and i from the generated image filename for matching real images
        suffix = generated_image_path.name.split('generated_')[-1]
        real_image_path = folder_path / f'real_{suffix}'
        
        if real_image_path.exists():
            lpips_score, psnr_score, ssim_score = evaluate(str(generated_image_path), str(real_image_path))
            lpips_score = lpips_score.detach().item()
            # print(lpips_score)
            # print(psnr_score)
            # print(ssim_score)
            total_lpips += lpips_score
            total_psnr += psnr_score
            total_ssim += ssim_score
            count += 1
        else:
            print(f"Missing real image for {generated_image_path}")

    if count > 0:
        avg_lpips = total_lpips / count
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"Evaluation score of {model_type}")
        print(f"Average LPIPS: {avg_lpips}")
        print(f"Average PSNR: {avg_psnr}")
        print(f"Average SSIM: {avg_ssim}")
    else:
        print("No valid image pairs found for evaluation.")

def get_score_from_CNN(test_dataset_folder_path):
    
    model = load_model(model_path= r'D:\23ws\CV\3dcv-final\final_project_3dcv\experimental results\CNN1\checkpoint\checkpoint_20.pth', model_type='cnn')
    folder = Path(test_dataset_folder_path)
    os.makedirs(r'eval', exist_ok=True)

    i = 0
    # get the output from cnn
    for img_path in folder.glob('*.png'):
        # print(img_path)

        # images = Image.open(img_path).convert('RGB')
        images = Image.open(img_path)
        images.save(f'eval/real_{i}.png')

        real_images = transform_to_tensor(images).to(device)        
        grayscale_images = TF.rgb_to_grayscale(real_images)
        grayscale_images = (grayscale_images + 1.0) / 2.0
        if grayscale_images.dim() == 3:
            grayscale_images = grayscale_images.unsqueeze(0)
            # print('get grayscale_images')
            
        with torch.no_grad():
            outputs_ab = model(grayscale_images)
        
        # convert output_ab to rgb image
        l_channel = grayscale_images.expand(-1, 3, -1, -1)  # Duplicate the grayscale channel to match LAB dimension
        outputs = merge_l_ab(l_channel[:, :1, :, :], outputs_ab)  # Use only the first channel for L
        # save sample image
        TF.to_pil_image(postprocess(outputs[0]).cpu()).save(f'eval/generated_{i}.png')

        i = i + 1
    
    images_folder_path = r'D:\23ws\CV\3dcv-final\final_project_3dcv\eval'
    evaluate_folder(images_folder_path, model_type='cnn')

def get_score_from_GAN(test_dataset_folder_path):
    
    model = load_model(model_path= r'D:\23ws\CV\3dcv-final\final_project_3dcv\experimental results\GAN2\checkpoint\checkpoint_20.pth', model_type='gan')
    folder = Path(test_dataset_folder_path)
    os.makedirs(r'eval', exist_ok=True)
    
    i = 0
    # get the output from gan
    for img_path in folder.glob('*.png'):
        # print(img_path)

        # images = Image.open(img_path).convert('RGB')
        images = Image.open(img_path)
        images.save(f'eval/real_{i}.png')

        real_images = transform_to_tensor(images).to(device)        
        grayscale_images = TF.rgb_to_grayscale(real_images)
        if grayscale_images.dim() == 3:
            grayscale_images = grayscale_images.unsqueeze(0)
            # print('get grayscale_images')
            
        with torch.no_grad():
            generated_image = model(grayscale_images)
        
        # save sample image
        TF.to_pil_image(postprocess(generated_image[0]).cpu()).save(f'eval/generated_{i}.png')

        i = i + 1
    
    images_folder_path = r'D:\23ws\CV\3dcv-final\final_project_3dcv\eval'
    evaluate_folder(images_folder_path, model_type='gan')

def get_score_from_diffusion(test_dataset_folder_path):
    model = load_model(model_path= r'D:\23ws\CV\3dcv-final\final_project_3dcv\experimental results\diffusion2\checkpoint\epoch_18.pth', model_type='diffusion')
    folder = Path(test_dataset_folder_path)
    os.makedirs(r'eval', exist_ok=True)
    i = 0
    # get the output from diffusion
    for img_path in folder.glob('*.png'):
        # print(img_path)

        # images = Image.open(img_path).convert('RGB')
        images = Image.open(img_path)
        images.save(f'eval/real_{i}.png')

        grayscale_images = transform_grayscale(images).to(device)

        if grayscale_images.dim() == 3:
            grayscale_images = grayscale_images.unsqueeze(0)
            # print('get grayscale_images')
        input_image = rgb_to_lab(grayscale_images).to(device)
        output = model.reverse_diffusion(input_image)
        generated_image = lab_to_rgb(output)   
        # save sample image
        TF.to_pil_image((generated_image[0])).save(f'eval/generated_{i}.png')

        i = i + 1
    
    images_folder_path = r'D:\23ws\CV\3dcv-final\final_project_3dcv\eval'
    evaluate_folder(images_folder_path, model_type='diffusion')

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # test dataset
    test_dataset_folder_path = r'D:\23ws\CV\3dcv-final\final_project_3dcv\dataset\validation\00000'

    get_score_from_CNN(test_dataset_folder_path)
    get_score_from_GAN(test_dataset_folder_path)
    get_score_from_diffusion(test_dataset_folder_path)
