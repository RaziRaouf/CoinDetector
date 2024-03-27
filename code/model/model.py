# model.py
import cv2
import matplotlib.pyplot as plt
from preprocess import *
from segmentation import *
from feature_extraction import *

def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\40.jpg"
    original_image = cv2.imread(image_path)
    grayscale_image = convert_to_grayscale(image_path)
    blurred_image = apply_gaussian_blur(grayscale_image)
    segmented_image = apply_otsu_threshold(blurred_image)
    dilation_image = apply_dilation(segmented_image)
    erosion_image = apply_erosion(dilation_image)
    cannied_image = apply_canny_edge_detection(erosion_image)
    contours, hierarchy = find_contours(cannied_image)
    image_with_contours = display_contours(grayscale_image, contours)
    image_with_circles = apply_hough_circle_detection(cannied_image, contours, hierarchy)

    # Display all six images side by side
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))

    # Plot the original image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Plot the grayscale image
    axes[0, 1].imshow(grayscale_image, cmap='gray')
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].axis('off')

    # Plot the blurred image
    axes[0, 2].imshow(blurred_image, cmap='gray')
    axes[0, 2].set_title('Blurred Image')
    axes[0, 2].axis('off')

    # Plot the otsu segmented image
    axes[1, 0].imshow(segmented_image, cmap='gray')
    axes[1, 0].set_title('Otsu Segmented Image')
    axes[1, 0].axis('off')

    # Plot the dilation image
    axes[1, 1].imshow(cannied_image, cmap='gray')
    axes[1, 1].set_title('Canny Image')
    axes[1, 1].axis('off')

    # contour image
    axes[1, 2].imshow(image_with_contours, cmap='gray')
    axes[1, 2].set_title('Contour Image')
    axes[1, 2].axis('off')

    # Plot the circles image
    axes[2, 0].imshow(image_with_circles, cmap='gray')  # Corrected: Use 'image_with_circles'
    axes[2, 0].set_title('Circles Image')
    axes[2, 0].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
