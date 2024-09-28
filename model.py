import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes):
    # Load the MobileNetV2 model without the top layer
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top of MobileNetV2
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Adjust for your 5 classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Example usage
    model = build_model(num_classes=5)
    model.summary()
