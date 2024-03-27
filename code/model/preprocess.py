# preprocess.py
import cv2

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load image")
        return None
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return grayscale_image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    
    return blurred_image

def apply_median_blur(image, kernel_size=5):
    blurred_image = cv2.medianBlur(image, kernel_size)
    
    return blurred_image

def apply_histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    
    return equalized_image
