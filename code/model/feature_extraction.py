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

def apply_hough_circle_detection(image, contours, hierarchy, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0):
    # Initialize empty list to store detected circles
    all_circles = []

    # Iterate through contours and hierarchy
    for i, contour in enumerate(contours):
        # Access the hierarchy information for the current contour
        current_hierarchy = hierarchy[0][i]

        # Check if the contour is an external contour and not a child contour
        if current_hierarchy[3] == -1:
            # Compute bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)

            # Extract ROI (region of interest) from grayscale image
            roi = image[y:y+h, x:x+w]

            # Apply Hough Circle Transform to detect circles within ROI
            circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

            # If circles are found, append them to the list of detected circles
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0]:
                    # Calculate center coordinates relative to the original image
                    center = (circle[0] + x, circle[1] + y)
                    radius = circle[2]
                    all_circles.append((center, radius))

    # Draw the detected circles on the original image
    image_with_circles = np.copy(image)
    for circle in all_circles:
        center, radius = circle
        cv2.circle(image_with_circles, center, radius, (255, 255, 255), 2)

    return image_with_circles
