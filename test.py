"""Handwritten digit recognition from image using CNN model trained on MNIST"""

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import os

# Load model
try:
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "mnist_cnn_model.h5")
    model = load_model(model_path)
    print("Model loaded successfully.\n")
except IOError:
    print("Error: Cannot find mnist_cnn_model.h5")
    exit()


def preprocess_digit(digit_image):
    """Convert digit image to MNIST format (28x28, black background, white digit)"""
    try:
        digit_np = np.array(digit_image)
        
        # Find bounding box and crop
        coords = cv2.findNonZero(digit_np)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit_np = digit_np[y:y+h, x:x+w]
        
        # Add padding
        digit_np = cv2.copyMakeBorder(digit_np, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        
        # Resize maintaining aspect ratio
        height, width = digit_np.shape
        if height == 0 or width == 0:
            return None
        
        if height > width:
            new_height = 20
            new_width = max(1, int(20 * width / height))
        else:
            new_width = 20
            new_height = max(1, int(20 * height / width))
        
        new_width = min(new_width, 20)
        new_height = min(new_height, 20)
        
        digit_resized = cv2.resize(digit_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 canvas
        img_padded = np.zeros((28, 28), dtype=np.uint8)
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        img_padded[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = digit_resized
        
        # Normalize
        img_array = img_padded.astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def recognize_from_file():
    """Recognize digit sequence from image file"""
    # File dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title="Select image with digit sequence",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    root.destroy()
    
    if not file_path:
        print("No file selected.")
        return
    
    # Read image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot read image.")
        return
    
    # Preprocessing
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find and filter contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_bboxes = []
    image_height, image_width = image.shape
    
    # Calculate median height of all contours for adaptive filtering
    all_heights = [cv2.boundingRect(cnt)[3] for cnt in contours]
    median_height = np.median(all_heights) if all_heights else 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        area = cv2.contourArea(cnt)
        
        # More flexible filtering to catch thin digits like "1"
        if (h > max(10, median_height * 0.2) and  # Lower minimum height
            0.2 < aspect_ratio < 8.0 and  # Much wider range for thin digits
            area > 50 and  # Lower minimum area
            w > 3 and  # Lower minimum width for "1"
            h < image_height * 0.9 and  # Not too tall
            w < image_width * 0.6):  # Not too wide
            digit_bboxes.append((x, y, w, h))
    
    # Sort by row then column (top-to-bottom, left-to-right)
    def sort_boxes_by_row(boxes):
        if not boxes:
            return boxes
        
        # Group boxes by rows
        boxes_sorted = sorted(boxes, key=lambda b: b[1])  # Sort by y first
        rows = []
        current_row = [boxes_sorted[0]]
        
        for box in boxes_sorted[1:]:
            # If y-coordinate is close to current row, add to same row
            if abs(box[1] - current_row[0][1]) < median_height * 0.5:
                current_row.append(box)
            else:
                # Start new row
                rows.append(sorted(current_row, key=lambda b: b[0]))  # Sort by x
                current_row = [box]
        
        rows.append(sorted(current_row, key=lambda b: b[0]))  # Add last row
        
        # Flatten rows
        return [box for row in rows for box in row]
    
    digit_bboxes = sort_boxes_by_row(digit_bboxes)
    print(f"Detected {len(digit_bboxes)} digits.\n")
    
    # Recognition
    recognized_string = ""
    pil_image = Image.fromarray(thresh)
    
    for index, (x, y, w, h) in enumerate(digit_bboxes):
        padding = 5
        digit_pil = pil_image.crop((
            max(0, x - padding), 
            max(0, y - padding), 
            min(thresh.shape[1], x + w + padding), 
            min(thresh.shape[0], y + h + padding)
        ))
        
        processed_digit = preprocess_digit(digit_pil)
        
        if processed_digit is not None:
            prediction = model.predict(processed_digit, verbose=0)
            digit = np.argmax(prediction[0])
            confidence = prediction[0][digit]
            
            print(f"  [{index}] -> {digit} ({confidence:.1%})")
            recognized_string += str(digit)
        else:
            recognized_string += "?" 
    
    # Result
    print(f"\n{'='*50}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Result: {recognized_string}")
    print(f"Success: {len([c for c in recognized_string if c != '?'])}/{len(digit_bboxes)} digits")
    print(f"{'='*50}")


if __name__ == "__main__":
    recognize_from_file()
