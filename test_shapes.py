"""Test geometric shape recognition from image using trained CNN model"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import os

# Configuration
IMG_SIZE = 64
CLASSES = ["circle", "rectangle", "triangle"]

# Load model
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "shapes_cnn_model.h5")
    model = load_model(model_path)
    print("=" * 60)
    print("GEOMETRIC SHAPE RECOGNITION")
    print("=" * 60)
    print("Model loaded successfully.\n")
except IOError:
    print("Error: Cannot find shapes_cnn_model.h5")
    print("Please run 'python train_shapes.py' first to train the model.")
    exit()


def preprocess_shape(shape_image, img_size=IMG_SIZE):
    """Preprocess shape image for CNN"""
    try:
        # Convert to grayscale if needed
        if len(shape_image.shape) == 3:
            shape_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(shape_image, (img_size, img_size))
        
        # Normalize
        normalized = resized.astype("float32") / 255.0
        
        # Reshape for CNN
        img_array = normalized.reshape(1, img_size, img_size, 1)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def recognize_shapes_from_file():
    """Recognize shapes from image file"""
    # File dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title="Select image with shapes",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    root.destroy()
    
    if not file_path:
        print("No file selected.")
        return
    
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    # Read image
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Cannot read image.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} shape(s)\n")
    
    # Create output image
    output = image.copy()
    
    # Process each contour
    for idx, contour in enumerate(contours):
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < 500:  # Skip small noise
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract ROI
        roi = gray[y:y+h, x:x+w]
        
        # Preprocess for CNN
        preprocessed = preprocess_shape(roi)
        
        if preprocessed is not None:
            # Predict
            predictions = model.predict(preprocessed, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            shape_name = CLASSES[predicted_class]
            
            # Draw on output
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"{shape_name}: {confidence:.1f}%"
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Print prediction
            print(f"Shape {idx + 1}:")
            print(f"  Detected: {shape_name}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  All predictions:")
            for i, class_name in enumerate(CLASSES):
                print(f"    {class_name}: {predictions[0][i]*100:.2f}%")
            print()
    
    # Display result
    cv2.imshow("Original", image)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Shape Recognition", output)
    
    print("=" * 60)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_single_shape():
    """Recognize a single shape from clean image"""
    # File dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title="Select single shape image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    root.destroy()
    
    if not file_path:
        print("No file selected.")
        return
    
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    # Read image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Cannot read image.")
        return
    
    # Preprocess
    preprocessed = preprocess_shape(image)
    
    if preprocessed is not None:
        # Predict
        predictions = model.predict(preprocessed, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        shape_name = CLASSES[predicted_class]
        
        print(f"\n{'='*60}")
        print(f"PREDICTION: {shape_name}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"{'='*60}")
        print("\nAll predictions:")
        for i, class_name in enumerate(CLASSES):
            print(f"  {class_name}: {predictions[0][i]*100:.2f}%")
        
        # Display
        cv2.imshow("Input Image", image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Main menu
def main():
    print("\nSelect recognition mode:")
    print("1. Recognize multiple shapes in image")
    print("2. Recognize single shape")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        recognize_shapes_from_file()
    elif choice == "2":
        recognize_single_shape()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
