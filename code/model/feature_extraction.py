# feature_extraction.py
import cv2
import numpy as np

def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1, threshold2)
    
    return edges

def find_contours(image):
    # Find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def display_contours(grayscale_image, contours):
    # Draw contours on the original grayscale image
    image_with_contours = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    
    return image_with_contours

