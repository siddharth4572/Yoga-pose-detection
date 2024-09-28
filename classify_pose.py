import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained model
model = load_model('yoga_pose_model.h5')

# Define a function to classify an image
def classify_pose(image_path):
    # Load the image
    img = load_img(image_path, target_size=(224, 224))  # Resize the image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the pixel values

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest score

    return predicted_class, img  # Return predicted class and original image

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path of the image to classify: ").strip()  # Get image path from user and strip spaces
    class_index, img = classify_pose(image_path)

    # Optional: You can map the class index to actual pose names if you have a list
    class_labels = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']  # Replace with your actual class names
    predicted_pose = class_labels[class_index]

    # Display the image with the predicted pose
    plt.imshow(img)  # Show the image
    plt.title(f'Predicted Pose: {predicted_pose}')  # Title with predicted pose name
    plt.axis('off')  # Turn off axis

    # Add text to the image at a specified position
    plt.text(10, 20, f'Pose: {predicted_pose}', fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))

    plt.show()  # Display the image
