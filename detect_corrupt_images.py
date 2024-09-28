import os
from PIL import Image

# Define your image directory (train/test)
image_dir = 'data/test'

# Function to detect and remove corrupted images
def detect_and_remove_corrupted_images(image_dir):
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                try:
                    img = Image.open(image_path)
                    img.verify()  # Check if it's a valid image
                except (IOError, SyntaxError) as e:
                    print(f'Corrupted image detected: {image_path}. Deleting it.')
                    os.remove(image_path)  # Optionally delete the corrupt file

# Run the function
if __name__ == "__main__":
    detect_and_remove_corrupted_images(image_dir)
