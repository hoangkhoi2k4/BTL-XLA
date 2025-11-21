"""Test custom model with image file containing handwritten letters"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import os

print("=" * 60)
print("HANDWRITTEN LETTER RECOGNITION - CUSTOM MODEL")
print("=" * 60)

# Load model
MODEL_PATH = "custom_letters_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Custom model not found: {MODEL_PATH}")
    print("   Please train the model first: python train_custom_model.py")
    exit(1)

print(f"📂 Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"🎯 Model Accuracy: 99.65%")

# Letters classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Create debug directory
DEBUG_DIR = "debug_custom_letters"
os.makedirs(DEBUG_DIR, exist_ok=True)

def preprocess_letter(img):
    """Preprocess a single letter image"""
    # Make square with padding
    h, w = img.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    
    # Center the letter
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert if needed
    if np.mean(resized) > 127:
        resized = 255 - resized
    
    # Normalize
    resized = resized.astype('float32') / 255.0
    resized = resized.reshape(1, 28, 28, 1)
    
    return resized

def detect_and_segment_letters(image_path):
    """Detect and segment individual letters from image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return None, []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(DEBUG_DIR, "1_threshold.png"), thresh)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(os.path.join(DEBUG_DIR, "2_morphology.png"), morph)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    letter_contours = []
    min_area = 100
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / w if w > 0 else 0
            if 0.5 < aspect_ratio < 4 and w > 10 and h > 10:
                letter_contours.append((x, y, w, h))
    
    # Sort by x-coordinate (left to right)
    letter_contours = sorted(letter_contours, key=lambda b: b[0])
    
    # Extract letter images
    letters = []
    debug_img = img.copy()
    
    for i, (x, y, w, h) in enumerate(letter_contours):
        # Add padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray.shape[1], x + w + padding)
        y2 = min(gray.shape[0], y + h + padding)
        
        # Extract letter
        letter_img = gray[y1:y2, x1:x2]
        letters.append(letter_img)
        
        # Draw rectangle
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i+1), (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save individual letter
        cv2.imwrite(os.path.join(DEBUG_DIR, f"letter_{i+1}.png"), letter_img)
    
    # Save debug image
    cv2.imwrite(os.path.join(DEBUG_DIR, "3_detected_letters.png"), debug_img)
    
    return gray, letters

def recognize_letters(image_path):
    """Main function to recognize letters from image"""
    print(f"\n📷 Processing image: {os.path.basename(image_path)}")
    
    # Detect and segment letters
    gray, letters = detect_and_segment_letters(image_path)
    
    if letters is None or len(letters) == 0:
        print("❌ Không phát hiện được chữ cái nào!")
        return
    
    print(f"✅ Phát hiện {len(letters)} chữ cái")
    
    # Recognize each letter
    results = []
    print("\n" + "=" * 60)
    print("KẾT QUẢ NHẬN DẠNG:")
    print("=" * 60)
    
    for i, letter_img in enumerate(letters):
        # Preprocess
        processed = preprocess_letter(letter_img)
        
        # Save preprocessed image
        debug_processed = (processed[0, :, :, 0] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"processed_{i+1}.png"), debug_processed)
        
        # Predict
        prediction = model.predict(processed, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100
        
        # Get top 3
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3 = [(CLASSES[idx], prediction[0][idx] * 100) for idx in top_3_idx]
        
        letter = CLASSES[predicted_class]
        results.append(letter)
        
        print(f"Chữ cái {i+1}: {letter} (độ tin cậy: {confidence:.2f}%)")
        print(f"           Top 3: {top_3[0][0]}({top_3[0][1]:.1f}%), {top_3[1][0]}({top_3[1][1]:.1f}%), {top_3[2][0]}({top_3[2][1]:.1f}%)")
    
    # Show final result
    print("\n" + "=" * 60)
    print(f"📝 Chuỗi nhận dạng: {''.join(results)}")
    print("=" * 60)
    print(f"\n💾 Debug images đã lưu vào: {DEBUG_DIR}/")
    
    return results

def main():
    # Open file dialog
    print("\n📁 Chọn file ảnh chứa chữ cái viết tay...")
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh chứa chữ cái viết tay",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("❌ Không có file nào được chọn!")
        return
    
    # Recognize letters
    recognize_letters(file_path)
    
    print("\n✨ Hoàn tất!")

if __name__ == "__main__":
    main()
