# segmentation.py
import cv2

def apply_otsu_threshold(image):
    # Apply Otsu's thresholding
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return segmented_image

def apply_adaptive_threshold(image, block_size=11, c=2):
    # Apply adaptive thresholding
    segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    
    return segmented_image

def apply_erosion(image, kernel_size=3, iterations=1):
    # Define the structuring element for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply erosion
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    
    return eroded_image

def apply_dilation(image, kernel_size=3, iterations=1):
    # Define the structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    
    return dilated_image
