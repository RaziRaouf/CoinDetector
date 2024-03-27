# model.py
import cv2
from preprocess import convert_to_grayscale

def main():
    image_path = "F:\France\paris_cite\S2\image\projet\CoinDetector\dataset\images\.png"
    grayscale_image = convert_to_grayscale(image_path)

    # Display the original and grayscale images
    if grayscale_image is not None:
        original_image = cv2.imread(image_path)
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Grayscale Image", grayscale_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
