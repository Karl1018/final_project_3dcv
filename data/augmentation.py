from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image, UnidentifiedImageError
import os
import numpy as np

class ImageAugmentor:
    def __init__(self, input_folder, output_folder, total_aug_per_image=20):
        self.input_folder = input_folder
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder
        self.total_aug_per_image = total_aug_per_image
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    @staticmethod
    def is_valid_image(file_path):
        try:
            Image.open(file_path)
            return True
        except UnidentifiedImageError:
            return False

    def augment_folder(self):
        for filename in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, filename)
            
            # Skip if the file is not a valid image or not a common image file type
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')) or not self.is_valid_image(file_path):
                print(f"Skipping non-image file: {file_path}")
                continue
            self.augment_image(file_path, filename)

    def augment_image(self, image_path, image_name):
        img = load_img(image_path)  # Load image
        x = img_to_array(img)  # Convert image to numpy array
        x = np.expand_dims(x, axis=0)  # Add an extra dimension
        
        # Generate and save augmented images to the target folder
        i = 0
        for batch in self.datagen.flow(x, batch_size=1, save_to_dir=self.output_folder, save_prefix=image_name, save_format='png'):
            i += 1
            if i > self.total_aug_per_image:  
                break
        print(f"Finish Generating augmented image for {image_name}") 
imageAugmentor = ImageAugmentor("data/afhq/cat", "data/afhq/cat_augmentation", 20)
imageAugmentor.augment_folder()
