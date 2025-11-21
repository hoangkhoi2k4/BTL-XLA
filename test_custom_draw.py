"""Interactive handwritten letter recognition with custom trained model"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

print("=" * 60)
print("INTERACTIVE LETTER RECOGNITION - CUSTOM MODEL")
print("=" * 60)

# Load custom model
MODEL_PATH = "custom_letters_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Custom model not found: {MODEL_PATH}")
    print("   Please train the model first: python train_custom_model.py")
    exit(1)

print(f"📂 Loading custom model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Letters classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Drawing settings
drawing = False
canvas = np.ones((400, 400), dtype=np.uint8) * 255
last_point = None

def preprocess_for_model(img):
    """Preprocess canvas for model prediction"""
    # Find bounding box of drawn content
    coords = cv2.findNonZero(255 - img)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    # Crop
    cropped = img[y1:y2, x1:x2]
    
    # Make square
    h, w = cropped.shape
    size = max(h, w)
    square = np.ones((size, size), dtype=np.uint8) * 255
    
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert (model expects white on black)
    resized = 255 - resized
    
    # Normalize
    resized = resized.astype('float32') / 255.0
    resized = resized.reshape(1, 28, 28, 1)
    
    return resized

def predict_letter():
    """Predict letter from canvas"""
    processed = preprocess_for_model(canvas)
    
    if processed is None:
        return None, []
    
    # Predict
    prediction = model.predict(processed, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    
    # Get top 5
    top_5_idx = np.argsort(prediction[0])[-5:][::-1]
    top_5 = [(CLASSES[idx], prediction[0][idx] * 100) for idx in top_5_idx]
    
    return CLASSES[predicted_class], top_5

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for drawing"""
    global drawing, last_point, canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_point:
            cv2.line(canvas, last_point, (x, y), 0, 12)
            last_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None

def main():
    global canvas
    
    print("\n" + "=" * 60)
    print("HƯỚNG DẪN:")
    print("=" * 60)
    print("• Vẽ chữ cái (A-Z) trên canvas")
    print("• Nhấn SPACE để nhận dạng")
    print("• Nhấn C để xóa canvas")
    print("• Nhấn Backspace để xóa chữ cuối")
    print("• Nhấn ESC hoặc Q để thoát")
    print("=" * 60)
    print(f"\n🎯 Model Accuracy: 99.65% (Custom trained)")
    print("=" * 60)
    
    # Create window
    cv2.namedWindow('Draw Letter Here')
    cv2.setMouseCallback('Draw Letter Here', mouse_callback)
    
    recognized_text = ""
    
    while True:
        # Create display
        display = canvas.copy()
        
        # Add instructions
        cv2.putText(display, "Draw letter (A-Z)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
        cv2.putText(display, "SPACE: Recognize | C: Clear | ESC: Exit", (10, 380),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
        
        # Show recognized text
        if recognized_text:
            cv2.putText(display, f"Text: {recognized_text}", (10, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Draw Letter Here', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Recognize (SPACE)
        if key == ord(' '):
            letter, top_5 = predict_letter()
            
            if letter:
                recognized_text += letter
                print(f"\n✅ Nhận dạng: {letter}")
                print("   Top 5 predictions:")
                for i, (l, conf) in enumerate(top_5, 1):
                    print(f"      {i}. {l}: {conf:.2f}%")
                print(f"\n📝 Chuỗi hiện tại: {recognized_text}")
                
                # Clear canvas for next letter
                canvas = np.ones((400, 400), dtype=np.uint8) * 255
            else:
                print("❌ Không phát hiện chữ cái! Vui lòng vẽ rõ hơn.")
        
        # Clear canvas (C)
        elif key == ord('c') or key == ord('C'):
            canvas = np.ones((400, 400), dtype=np.uint8) * 255
            print("🗑️ Canvas đã được xóa")
        
        # Backspace - remove last character
        elif key == 8 and recognized_text:
            recognized_text = recognized_text[:-1]
            print(f"⬅️ Xóa chữ cuối. Chuỗi: {recognized_text}")
        
        # Exit (ESC or Q)
        elif key == 27 or key == ord('q') or key == ord('Q'):
            break
    
    cv2.destroyAllWindows()
    
    if recognized_text:
        print("\n" + "=" * 60)
        print(f"🎯 CHUỖI CUỐI CÙNG: {recognized_text}")
        print("=" * 60)
    
    print("\n✨ Cảm ơn bạn đã sử dụng!")

if __name__ == "__main__":
    main()
