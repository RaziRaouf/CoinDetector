import os
import cv2
import numpy as np
from pathlib import Path
from code.model.model import model_pipeline
import shutil

def process_and_save_images(image_dir, output_dir, display=False, details=False):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.JPG'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Run the model pipeline to process the image
        merged_circles_image, number_of_coin = model_pipeline(image_path, display=display, details=details)
        
        # Save the processed image with circles
        output_image_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_image_path, merged_circles_image)  # Save the image with drawn circles

        print(f"Processed and saved: {image_file}, Total Coins: {number_of_coin}")

# Example Usage:
image_dir = "dataset/images"  # Input folder with your images
output_dir = "dataset/processed_images"  # Folder where processed images will be saved

# Call the function to process and save images
process_and_save_images(image_dir, output_dir, display=False, details=False)
