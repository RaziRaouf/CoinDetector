# segmentation.py
import cv2
import numpy as np
from postprocessing import *

def apply_otsu_threshold(image):
    # Apply Otsu's thresholding
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return segmented_image


def color_based_segmentation(image):
    # Convert image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for euro coins
    color_ranges = {
        "1_cent": [(0, 100, 100), (20, 255, 255)],    # Reddish color
        "2_cent": [(0, 100, 100), (20, 255, 255)],    # Reddish color
        "5_cent": [(0, 100, 100), (20, 255, 255)],    # Reddish color
        "10_cent": [(0, 100, 100), (20, 255, 255)],   # Reddish color
        "20_cent": [(0, 100, 100), (20, 255, 255)],   # Reddish color
        "50_cent": [(0, 100, 100), (20, 255, 255)],   # Reddish color
        "1_euro": [(20, 100, 100), (35, 255, 255)],  # Gold color
        "2_euro": [(20, 100, 100), (35, 255, 255)]   # Gold color
    }
    
    # Initialize segmented image
    segmented_image = np.zeros_like(image)
    
    # Apply color-based segmentation for each coin denomination
    for coin, (lower, upper) in color_ranges.items():
        print("Segmenting", coin, "with HSV range:", lower, "-", upper)
        # Threshold the HSV image to get only pixels within the specified color range
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        
        # Apply the mask to the original image
        coin_segment = cv2.bitwise_and(image, image, mask=mask)
        
        # Add the segmented coin to the segmented image
        segmented_image = cv2.add(segmented_image, coin_segment)
    
    return segmented_image


def apply_adaptive_threshold(image, block_size=21, c=7, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    
    # Apply adaptive thresholding
    segmented_image = cv2.adaptiveThreshold(image, 255, method, cv2.THRESH_BINARY, block_size, c)
        
    return segmented_image

def edge_based_segmentation(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(grayscale_image, 30, 100)  # Adjust thresholds as needed
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a black background image
    segmented_image = np.zeros_like(image)
    
    # Draw contours on the segmented image
    cv2.drawContours(segmented_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)  # Fills contours with white
    
    return segmented_image
