import os
import cv2
import json
import math

# Paths
images_dir = "dataset/processed_images"  # Directory containing images
json_dir = "dataset/labels"              # Directory containing JSON files
output_dir = "dataset/cropped_coins"     # Directory to save cropped coins

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image in the images directory
for image_file in os.listdir(images_dir):
    if image_file.endswith((".jpg", ".png")):  # Adjust extensions if needed
        # Match the JSON file
        json_file = os.path.join(json_dir, f"{os.path.splitext(image_file)[0]}.json")
        if not os.path.exists(json_file):
            print(f"JSON file not found for {image_file}, skipping.")
            continue

        # Load the image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_file}")
            continue

        # Load the bounding box data from the JSON file
        with open(json_file, "r") as f:
            data = json.load(f)

        # Check if the "shapes" key exists
        if "shapes" not in data:
            print(f"No 'shapes' key found in {json_file}, skipping.")
            continue

        # Process each coin in the "shapes" key
        for i, shape in enumerate(data["shapes"]):
            if "points" not in shape:
                print(f"Missing 'points' in shape {i} of {json_file}, skipping.")
                continue

            # Extract the center and perimeter points
            (x_center, y_center), (x_perimeter, y_perimeter) = shape["points"]

            # Calculate the radius
            radius = int(math.sqrt((x_perimeter - x_center)**2 + (y_perimeter - y_center)**2))

            # Calculate the bounding box coordinates
            x_min = int(x_center - radius)
            y_min = int(y_center - radius)
            x_max = int(x_center + radius)
            y_max = int(y_center + radius)

            # Validate coordinates against image dimensions
            height, width, _ = image.shape
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))

            # Skip invalid bounding boxes
            if x_min >= x_max or y_min >= y_max:
                print(f"Invalid bounding box for coin {i} in {image_file}, skipping.")
                continue

            # Crop the coin region
            cropped_coin = image[y_min:y_max, x_min:x_max]

            # Skip empty regions
            if cropped_coin is None or cropped_coin.size == 0:
                print(f"Cropped region is empty for coin {i} in {image_file}, skipping.")
                continue

            # Get the label and ensure it is in a valid format
            label = shape.get("label", "unknown")  # Use label for naming

            # Create subfolder based on label if it doesn't exist
            label_folder = os.path.join(output_dir, str(label))  # Convert label to string
            os.makedirs(label_folder, exist_ok=True)

            # Define the output path for the cropped coin
            output_path = os.path.join(label_folder, f"{os.path.splitext(image_file)[0]}_coin_{i}_{label}.jpg")

            # Save the cropped image
            cv2.imwrite(output_path, cropped_coin)
            print(f"Cropped coin saved: {output_path}")
