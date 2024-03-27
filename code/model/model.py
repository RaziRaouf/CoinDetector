# model.py
import cv2
import matplotlib.pyplot as plt
from preprocess import *
from segmentation import *

def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\9.jpg"
    original_image = cv2.imread(image_path)
    grayscale_image = convert_to_grayscale(image_path)
    blurred_image = apply_gaussian_blur(grayscale_image)
    segmented_image = apply_otsu_threshold(blurred_image)
    dilation_image = apply_dilation(segmented_image)
    erosion_image = apply_erosion(dilation_image)

    # Display all six images side by side
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
    axes[1, 1].imshow(dilation_image, cmap='gray')
    axes[1, 1].set_title('Dilated Image')
    axes[1, 1].axis('off')

    # Plot the erosion image
    axes[1, 2].imshow(erosion_image, cmap='gray')
    axes[1, 2].set_title('Eroded Image')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
