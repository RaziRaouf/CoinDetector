# preprocess.py
import cv2

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load image")
        return None
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return grayscale_image
