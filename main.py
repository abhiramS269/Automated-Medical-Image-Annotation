import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Define parameters
IMG_SIZE = (224, 224)  # VGG16 input size

# Load pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze the convolutional base

# Add custom layers on top
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output: damage or no damage
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array, img

# Highlight the exact damaged area
def highlight_damage_area(img, prediction):
    if prediction[0] > 0.5:  # If damage detected
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Apply a binary threshold to identify potential damaged areas
        _, thresholded = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

        # Find contours of the potential damage area
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image
        img_with_highlighted_damage = np.array(img).copy()

        # Draw rectangles around the detected contours (damaged spots)
        for contour in contours:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a thin rectangle around the damaged area (colored red)
            cv2.rectangle(img_with_highlighted_damage, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Use 1 for thin lines

        return img_with_highlighted_damage
    else:
        return np.array(img)
# Predict and visualize damage in X-ray
def predict_and_visualize():
    # Open file dialog to select an X-ray image
    Tk().withdraw()  # Close the root Tkinter window
    img_path = askopenfilename(title="Select X-ray Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if img_path:  # Proceed if a file was selected
        img_array, img = load_and_preprocess_image(img_path)
        prediction = model.predict(img_array)

        if prediction[0] > 0.5:
            print("Prediction: Damage detected")
            img_with_highlighted_damage = highlight_damage_area(img, prediction)

            plt.figure(figsize=(8, 8))
            plt.imshow(img_with_highlighted_damage)
            plt.title("Detected Damaged Area")
            plt.axis('off')
            plt.show()
        else:
            print("Prediction: Normal X-ray")
    else:
        print("No file selected.")

# Run the function to predict and visualize
predict_and_visualize()
