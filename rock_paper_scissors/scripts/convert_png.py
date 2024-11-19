import os
import cv2

# Images folder path
IMAGES_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = os.path.join(IMAGES_PATH[:-7], 'images/')

# Output folder path (create if not exists)
OUTPUT_PATH = os.path.join(IMAGES_PATH, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Supported image formats
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Process each file in the folder
for filename in os.listdir(IMAGES_PATH):
    print(IMAGES_PATH)
    print(filename)
    if filename.lower().endswith(SUPPORTED_FORMATS):  # Check file format
        input_path = os.path.join(IMAGES_PATH, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"  # Ensure .png extension
        output_path = os.path.join(OUTPUT_PATH, output_filename)
        
        # Read and save the image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is not None:  # Check if the image is loaded successfully
            cv2.imwrite(output_path, img)
            print(f"Processed: {filename} -> {output_path}")
        else:
            print(f"Failed to process: {filename}")
