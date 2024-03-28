# model.py
import cv2
import matplotlib.pyplot as plt
from preprocess import *
from segmentation import *
from feature_extraction import *
from postprocessing import *


def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\2.jpg"
    original_image = cv2.imread(image_path)

    # Preprocess the image
    grayscale_image = convert_to_grayscale(image_path)
    blurred_image = apply_gaussian_blur(grayscale_image)
    blurred_image = apply_median_blur(blurred_image)

    # Segment the image using color-based and Otsu thresholding
    color_segmented_image = color_based_segmentation(original_image)
    otsu_segmented_image = apply_otsu_threshold(blurred_image)
    
    # Resize the color-based segmented image to match the dimensions of the Otsu segmented image and combine them
    color_segmented_image_gray = cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2GRAY)

    #combined_segmentation = np.bitwise_or(resized_color_segmented_image_gray, resized_otsu_segmented_image)
    combined_segmentation = cv2.bitwise_or(color_segmented_image_gray, otsu_segmented_image)

    #apply post processing morphological operations
    dilation_image = apply_dilation(combined_segmentation)
    erosion_image = apply_erosion(dilation_image)

    # Apply Canny edge detection, the contour detection using OpenCV
    cannied_image = apply_canny_edge_detection(erosion_image)
    contours, hierarchy = find_contours(cannied_image)
    image_with_contours = display_contours(grayscale_image, contours)

    # Apply Hough Circle Detection with OpenCV and skimage
    image_with_circles = apply_hough_circle_detection(cannied_image, contours, hierarchy)
    image_with_circles_skimage = apply_skimage_hough_circle_detection(cannied_image)


    # Display all six images side by side
    fig, axes = plt.subplots(4, 4, figsize=(18, 10))

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
    axes[1, 0].imshow(otsu_segmented_image, cmap='gray')
    axes[1, 0].set_title('otsu based Segmentation')
    axes[1, 0].axis('off')

    # Plot the color-based segmented image  
    axes[1, 1].imshow(color_segmented_image, cmap='gray')
    axes[1, 1].set_title('Color-based Segmentation')
    axes[1, 1].axis('off')

    # Plot the combined image
    axes[1, 2].imshow(combined_segmentation, cmap='gray')
    axes[1, 2].set_title('Otsu Segmented Image')
    axes[1, 2].axis('off')

    # Plot the canny image
    axes[1, 3].imshow(cannied_image, cmap='gray')
    axes[1, 3].set_title('Canny Image')
    axes[1, 3].axis('off')


    # contour image
    axes[2, 0].imshow(image_with_contours, cmap='gray')
    axes[2, 0].set_title('Contour Image')
    axes[2, 0].axis('off')

    # Plot the circles image
    axes[2, 1].imshow(image_with_circles, cmap='gray')  # Corrected: Use 'image_with_circles'
    axes[2, 1].set_title('Circles Image')
    axes[2, 1].axis('off')

    # Plot the circles image using skimage
    axes[2, 2].imshow(image_with_circles_skimage, cmap='gray')
    axes[2, 2].set_title('Circles Image ')
    axes[2, 2].axis('off')



    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
