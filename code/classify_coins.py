import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from code.model.model import model_pipeline

# 1. Load your trained CNN model
model = load_model('code/coin_detection_model.h5')  
class_labels = {0: '0.01', 1: '0.02', 2: '0.05', 3: '0.10', 4: '0.20' , 5:'0.50', 6:'1', 7:'2'} 

# 2. Run detection pipeline
def process_image(image_path):
    # Get merged circles from your existing pipeline
    merged_circles, _ = model_pipeline(image_path, display=False)
    
    # Load original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    return original_image, merged_circles

# 3. Classify and annotate
def classify_and_annotate(image, circles):
    annotated_image = image.copy()
    
    for circle in circles:
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Extract ROI (ensure it's within image bounds)
        y_min = max(0, y - r)
        y_max = min(image.shape[0], y + r)
        x_min = max(0, x - r)
        x_max = min(image.shape[1], x + r)
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            continue  # Skip invalid regions
            
        # Preprocess for CNN (match your training setup)
        resized = cv2.resize(roi, (224, 224))  # Use your model's input size
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Predict class
        pred = model.predict(input_tensor)
        label = class_labels[np.argmax(pred)]
        confidence = np.max(pred)
        
        # Draw annotations
        cv2.circle(annotated_image, (x, y), r, (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f})"
        cv2.putText(annotated_image, text, (x - r, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)
    
    return annotated_image

# 4. Main execution flow
def main():
    image_path = "dataset/images/7.jpg"  # Your test image
    
    # Step 1: Detect coins using your pipeline
    original_image, circles = process_image(image_path)
    
    # Step 2: Classify and annotate
    annotated_image = classify_and_annotate(original_image, circles)
    
    # Step 3: Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(original_image), plt.title("Original Image")
    plt.subplot(122), plt.imshow(annotated_image), plt.title("Annotated Prediction")
    plt.show()

if __name__ == "__main__":
    main()