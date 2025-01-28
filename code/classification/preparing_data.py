import os
import cv2
import numpy as np
from PIL import Image

# Directory containing the images with circles on coins
input_dir = 'dataset/processed_images'
output_dir = 'dataset/preprocessed_images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all .jpg and .JPG files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.JPG')]

# Define target size for resizing
target_size = (224, 224)

# Loop over all image files
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    
    # Read the image using OpenCV (as BGR)
    image = cv2.imread(image_path)
    
    # Resize the image to 224x224 pixels
    resized_image = cv2.resize(image, target_size)
    
    # Normalize the pixel values to [0, 1]
    normalized_image = resized_image / 255.0  # Now the values are in the range [0, 1]
    
    # Convert the normalized image back to uint8 format (range [0, 255])
    uint8_image = (normalized_image * 255).astype(np.uint8)
    
    # Convert to RGB (if it's BGR from OpenCV)
    rgb_image = cv2.cvtColor(uint8_image, cv2.COLOR_BGR2RGB)
    
    # Convert to a PIL image if you want to save it using Pillow
    pil_image = Image.fromarray(rgb_image)
    
    # Save the preprocessed image
    output_image_path = os.path.join(output_dir, image_file)
    pil_image.save(output_image_path)

    print(f"Processed and saved: {image_file}")

print("Preprocessing completed.")
