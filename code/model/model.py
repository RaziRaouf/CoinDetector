import cv2
import matplotlib.pyplot as plt
from .preprocess import *
from .segmentation import *
from .feature_extraction import *
from .postprocessing import *

def model_pipeline(image_path, display=False, details=False):
    """
    Cette fonction est le pipeline principal du modèle. Elle prend en entrée un chemin d'image et un booléen pour l'affichage.
    Elle effectue les étapes suivantes :
    1. Charge l'image à partir du chemin donné.
    2. Pré-traite l'image en utilisant le flou gaussien, la correction gamma, la conversion en niveaux de gris, le flou médian et l'égalisation adaptative de l'histogramme.
    3. segmente l'image pretraitée en utilisant le seuillage d'Otsu, multi-Otsu, basé sur la couleur, et adaptatif.
    4. Applique la détection de bord de Canny aux 4 images segmentées et les post-traite en utilisant le flou gaussien et la fermeture.
    5. A partir des images 4 images cannées, trouve les contours en utilisant openCV et transforme les contours circulaires en cercles.
    6. Affiche les contours et les cercles des 4 images cannées coté à coté.
    7. Fusionne et post-traite les cercles de toutes les images de Canny.
    8. Si l'affichage est True, elle affiche toutes les images et les tracés côte à côte.
    9. Si l'affichage est False, elle renvoie les cercles fusionnés et le nombre de pièces (cercles).
    """

    original_image = cv2.imread(image_path)    
    if original_image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Pretraitements de l'image
    preprocessed_image = apply_gaussian_blur(original_image.copy())
    preprocessed_image1 = apply_gamma_correction(preprocessed_image.copy())
    preprocessed_image2 = convert_to_grayscale(preprocessed_image1.copy())
    preprocessed_image3 = apply_median_blur(preprocessed_image2.copy())
    preprocessed_image4 = apply_adaptive_histogram_equalization(preprocessed_image3.copy())

    # Segmentation de l'image pretraitee avec 4 methodes differentes
    otsu_segmented_image = apply_otsu_threshold(preprocessed_image4.copy())
    multi_otsu_segmented_image, hist, peaks = multi_otsu_thresholding(preprocessed_image4.copy())
    color_segmented_image = color_based_segmentation(preprocessed_image.copy())
    adaptive_segmented_image = apply_adaptive_threshold(preprocessed_image3.copy())

    # Application de la detection de bord de Canny aux images segmentees et post-traitement avec flou gaussien et fermeture
    cannied_image = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(otsu_segmented_image.copy())))
    cannied_image1 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(multi_otsu_segmented_image.copy())))
    cannied_image2 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(color_segmented_image.copy())))
    cannied_image3 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(adaptive_segmented_image.copy())))

    # Trouver les contours des images cannées et les transformer les contours circulaires en cercles
    circles, contours, hierarchy = find_contours_circles(cannied_image)
    image_with_contours = display_contours(preprocessed_image4, contours)
    image_with_circles, number_of_coins = display_circles(preprocessed_image4, circles)

    circles1, contours1, hierarchy1 = find_contours_circles(cannied_image1)
    image_with_contours1 = display_contours(preprocessed_image4, contours1)
    image_with_circles1, number_of_coins1 = display_circles(preprocessed_image4, circles1)

    circles2, contours2, hierarchy2 = find_contours_circles(cannied_image2)
    image_with_contours2 = display_contours(preprocessed_image4, contours2)
    image_with_circles2, number_of_coins2 = display_circles(preprocessed_image4, circles2)

    circles3, contours3, hierarchy3 = find_contours_circles(cannied_image3)
    image_with_contours3 = display_contours(preprocessed_image4, contours3)
    image_with_circles3, number_of_coins3 = display_circles(preprocessed_image4, circles3)

    # Fusionner et post-traiter les cercles de toutes les images de Canny
    merged_circles = merge_and_postprocess_circles([circles, circles1, circles2, circles3])
    image_with_merged_circles, number_of_coin = display_circles(preprocessed_image4, merged_circles)

    if display==True:
        if details==True:
    # Afficher les images et les tracés côte à côte
            fig, axes = plt.subplots(4, 5, figsize=(18, 18))

            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image', color='blue', fontsize=10)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(otsu_segmented_image, cmap='gray')
            axes[0, 1].set_title('Otsu Segmented Image', color='blue', fontsize=10)
            axes[0, 1].axis('off')

            axes[0, 2].imshow(multi_otsu_segmented_image, cmap='gray')
            axes[0, 2].set_title('Multi-Otsu Segmented Image', color='blue', fontsize=10)
            axes[0, 2].axis('off')

            axes[0, 3].imshow(color_segmented_image, cmap='gray')
            axes[0, 3].set_title('Color-based Segmented Image', color='blue', fontsize=10)
            axes[0, 3].axis('off')

            axes[0, 4].imshow(adaptive_segmented_image, cmap='gray')
            axes[0, 4].set_title('Adaptive Segmented Image', color='blue', fontsize=10)
            axes[0, 4].axis('off')

            axes[1, 0].imshow(cv2.cvtColor(preprocessed_image1, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Preprocessed Image', color='blue', fontsize=10)
            axes[1, 0].axis('off')

            axes[1, 1].imshow(cannied_image, cmap='gray')
            axes[1, 1].set_title('Otsu Cannied Image', color='blue', fontsize=10)
            axes[1, 1].axis('off')

            axes[1, 2].imshow(cannied_image1, cmap='gray')
            axes[1, 2].set_title('Multi-Otsu Cannied Image', color='blue', fontsize=10)
            axes[1, 2].axis('off')

            axes[1, 3].imshow(cannied_image2, cmap='gray')
            axes[1, 3].set_title('Color-based Cannied Image', color='blue', fontsize=10)
            axes[1, 3].axis('off')

            axes[1, 4].imshow(cannied_image3, cmap='gray')
            axes[1, 4].set_title('Adaptive Cannied Image', color='blue', fontsize=10)
            axes[1, 4].axis('off')

            axes[2, 0].plot(hist)
            axes[2, 0].set_title('Histogram with Peaks', color='blue', fontsize=10)
            for peak in peaks:
                axes[2, 0].axvline(x=peak, color='r', linestyle='--')
            axes[2, 0].axis('off')

            axes[2, 1].imshow(image_with_contours, cmap='gray')
            axes[2, 1].set_title('Contours from Otsu Cannied', color='blue', fontsize=10)
            axes[2, 1].axis('off')

            axes[2, 2].imshow(image_with_contours1, cmap='gray')
            axes[2, 2].set_title('Contours from Multi-Otsu Cannied', color='blue', fontsize=10)
            axes[2, 2].axis('off')

            axes[2, 3].imshow(image_with_contours2, cmap='gray')
            axes[2, 3].set_title('Contours from Color-based Cannied', color='blue', fontsize=10)
            axes[2, 3].axis('off')

            axes[2, 4].imshow(image_with_contours3, cmap='gray')
            axes[2, 4].set_title('Contours from Adaptive Cannied', color='blue', fontsize=10)
            axes[2, 4].axis('off')

            axes[3, 0].imshow(image_with_merged_circles, cmap='gray')
            axes[3, 0].text(10, 30, 'Total Coins: ' + str(number_of_coin), color='red', fontsize=12)
            axes[3, 0].set_title('Merged Contours', color='blue', fontsize=10)
            axes[3, 0].axis('off')

            axes[3, 1].imshow(image_with_circles, cmap='gray')
            axes[3, 1].text(10, 30, 'Total Coins: ' + str(number_of_coins), color='red', fontsize=12)
            axes[3, 1].set_title('Circles from Otsu Cannied', color='blue', fontsize=10)
            axes[3, 1].axis('off')

            axes[3, 2].imshow(image_with_circles1, cmap='gray')
            axes[3, 2].text(10, 30, 'Total Coins: ' + str(number_of_coins1), color='red', fontsize=12)
            axes[3, 2].set_title('circles from Multi-Otsu Cannied', color='blue', fontsize=10)
            axes[3, 2].axis('off')

            axes[3, 3].imshow(image_with_circles2, cmap='gray')
            axes[3, 3].text(10, 30, 'Total Coins: ' + str(number_of_coins2), color='red', fontsize=12)
            axes[3, 3].set_title('circles from Color-based Cannied', color='blue', fontsize=10)
            axes[3, 3].axis('off')

            axes[3, 4].imshow(image_with_circles3, cmap='gray')
            axes[3, 4].text(10, 30, 'Total Coins: ' + str(number_of_coins3), color='red', fontsize=12)
            axes[3, 4].set_title('circles from Adaptive Cannied', color='blue', fontsize=10)
            axes[3, 4].axis('off')


            plt.tight_layout()
            plt.show()
        elif details == False:
            # Afficher l'image originale et limage avec les cercles fusionnés
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image', color='blue', fontsize=10)
            axes[0].axis('off')

            axes[1].imshow(image_with_merged_circles, cmap='gray')
            axes[1].text(10, 30, 'Total Coins: ' + str(number_of_coin), color='red', fontsize=12)
            axes[1].set_title('Merged Circles', color='blue', fontsize=10)
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

    elif display==False:
        return merged_circles, number_of_coin

def main():
    image_path = "dataset/images/0.jpg"
    #model_pipeline(image_path, display=True)

    #image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\10.jpg"
    model_pipeline(image_path, display=True, details=True)
"""
    merged_circles, number_of_coin = model_pipeline(image_path)
    print("Merged Circles:", merged_circles)
    print("Total Coins Detected:", number_of_coin)
"""

if __name__ == "__main__":
    main()