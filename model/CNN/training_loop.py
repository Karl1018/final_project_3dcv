import torch
from torch import optim
import torchvision.transforms.functional as TF
import tqdm
import os
import time

from . import network
from . import loss
from utils.image_process import postprocess  

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = network.ColorizationNet().to(device)

# Hyperparameters
lr = 0.001

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(train_dataloader, test_dataloader, epochs, save_interval, log_interval):
    print('Starting training...')
    print('Training on device:', device)
    print('Training data size:', len(train_dataloader.dataset))
    print('Training batch size:', train_dataloader.batch_size)

    # Create snapshot directory
    os.makedirs('snapshot/checkpoint', exist_ok=True)
    os.makedirs('snapshot/train', exist_ok=True)
    os.makedirs('snapshot/test', exist_ok=True)

    # Overwrite log file
    with open('snapshot/log.txt', 'w') as f:
        f.write('')
    
    # Save initial model state
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'snapshot/checkpoint/checkpoint_0.pth')

    start_time = time.time()

    for epoch in range(epochs):     
        # Train
        model.train()
        running_loss = 0.0
        for i, (images, _) in enumerate(tqdm.tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
            
            # Convert images to grayscale for CNN input
            real_images = images.to(device)  
            grayscale_images = TF.rgb_to_grayscale(real_images)

            output = model(grayscale_images)
            loss_CNN = loss.loss_CNN(output, real_images)

            optimizer.zero_grad()
            loss_CNN.backward()
            optimizer.step()

            running_loss += loss_CNN.item()

            # Logging
            if i % log_interval == 0:
                t = time.time() - start_time
                content = f'Time: {int(t//3600)}h {int(t%3600//60)}m, ' f'Epoch [{epoch+1}/{epochs}],' f'Loss: {loss_CNN.item():.4f}'
                print(content)
                with open('snapshot/log.txt', 'a') as f:
                    f.write(content + '\n')

            if i % save_interval == 0:
                # save sample image
                TF.to_pil_image(postprocess(output[0]).cpu()).save(f'snapshot/train/generated_{epoch}_{i}.png')
                TF.to_pil_image(postprocess(real_images[0]).cpu()).save(f'snapshot/train/real_{epoch}_{i}.png')
                TF.to_pil_image(postprocess(grayscale_images[0]).cpu()).save(f'snapshot/train/grayscale_{epoch}_{i}.png')
        
        # Test
        test_dataloader = tqdm.tqdm(test_dataloader)
        for i, (images, _) in enumerate(test_dataloader):
            test_dataloader.set_description(f'Testing Epoch [{epoch+1}/{epochs}]')
            real_images = images.to(device)
            grayscale_images = TF.rgb_to_grayscale(real_images)
            
            with torch.no_grad():
                output = model(grayscale_images)
                loss_CNN = loss.loss_CNN(output, real_images)
                
        t = time.time() - start_time
        content = f'Test: ' f'Epoch [{epoch+1}/{epochs}], ' f'Loss: {loss_CNN.item():.4f}'
        print(content)
        with open('snapshot/log.txt', 'a') as f:
            f.write(content + '\n')
        
        # save sample image
            TF.to_pil_image(postprocess(output[0]).cpu()).save(f'snapshot/test/generated_{epoch}_{i}.png')
            TF.to_pil_image(postprocess(real_images[0]).cpu()).save(f'snapshot/test/real_{epoch}_{i}.png')
            TF.to_pil_image(postprocess(grayscale_images[0]).cpu()).save(f'snapshot/test/grayscale_{epoch}_{i}.png')
            
        avg_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        torch.save(model.state_dict(), f'snapshot/checkpoint/checkpoint_{epoch+1}.pth')
