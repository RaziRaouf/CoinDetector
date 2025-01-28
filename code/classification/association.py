import os
import shutil

# Define the directory containing cropped coins
input_dir = "dataset/cropped_coins"  # Replace with your actual folder path
output_dir = "dataset/cropped_coins"  # Define the base output folder
os.makedirs(output_dir, exist_ok=True)

# Create subfolders for each coin value
coin_values = ["0.01", "0.02", "0.05", "0.10", "0.20", "0.50", "1", "2"]
for value in coin_values:
    os.makedirs(os.path.join(output_dir, value), exist_ok=True)

# Move files to respective subfolders
for file_name in os.listdir(input_dir):
    if file_name.endswith((".jpg", ".JPG")):  # Process image files only
        # Extract the coin value from the file name
        parts = file_name.split("_")
        coin_value = parts[-1].split(".")[0]  # Last part contains the value (e.g., "2" or "0.50")

        # Check if the coin value matches one of the predefined values (handle floating-point numbers)
        if coin_value.replace('.', '', 1).isdigit() and coin_value in coin_values:
            # Define source and destination paths
            src_path = os.path.join(input_dir, file_name)
            dest_path = os.path.join(output_dir, coin_value, file_name)

            # Move the file
            shutil.move(src_path, dest_path)
            print(f"Moved: {file_name} -> {coin_value}/")
        else:
            print(f"Warning: Coin value '{coin_value}' not recognized in file {file_name}. Skipping...")
