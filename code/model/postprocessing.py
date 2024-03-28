# postprocessing.py
import cv2

def apply_erosion(image, kernel_size=3, iterations=1):
    """
    Apply erosion to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times erosion is applied.

    Returns:
        numpy.ndarray: Eroded image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image

def apply_dilation(image, kernel_size=3, iterations=1):
    """
    Apply dilation to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times dilation is applied.

    Returns:
        numpy.ndarray: Dilated image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

# Add more post-processing functions as needed
