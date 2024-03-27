# model.py
import cv2
import matplotlib.pyplot as plt
from preprocess import *

def main():
    image_path = "F:\France\paris_cite\S2\image\projet\CoinDetector\dataset\images\9.jpg"
    original_image = cv2.imread(image_path)
    grayscale_image = convert_to_grayscale(image_path)
    apply_histogram_equalization = apply_gaussian_blur(grayscale_image)
    blurred_image = apply_median_blur(apply_histogram_equalization)


    # Display all three images side by side
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the grayscale image
    axes[1].imshow(grayscale_image, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')

    # Plot the blurred image
    axes[2].imshow(apply_histogram_equalization, cmap='gray')
    axes[2].set_title('histogram Equalized image')
    axes[2].axis('off')

    # Plot the blurred image
    axes[3].imshow(blurred_image, cmap='gray')
    axes[3].set_title('Blurred image')
    axes[3].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
