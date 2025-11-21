"""Tool to collect handwriting data for training custom model"""

import numpy as np
import cv2
import os
import json
from datetime import datetime

print("=" * 60)
print("HANDWRITING DATA COLLECTION TOOL")
print("=" * 60)

# Settings
CANVAS_SIZE = 400
DATASET_DIR = "custom_handwriting_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Letters to collect
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load existing progress
PROGRESS_FILE = os.path.join(DATASET_DIR, "progress.json")
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
else:
    progress = {letter: {"train": 0, "test": 0} for letter in LETTERS}

def save_progress():
    """Save collection progress"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_statistics():
    """Get collection statistics"""
    total_train = sum(p["train"] for p in progress.values())
    total_test = sum(p["test"] for p in progress.values())
    completed_letters = [l for l in LETTERS if progress[l]["train"] >= 50]
    return total_train, total_test, completed_letters

# Drawing state
drawing = False
canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
last_point = None
current_letter_idx = 0
current_letter = LETTERS[current_letter_idx]
samples_target = 50  # Target samples per letter for training
test_samples_target = 10  # Target samples for test set

def preprocess_for_save(img):
    """Preprocess drawn image for saving"""
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
    
    # Resize to 64x64 (higher res for better quality)
    resized = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
    
    return resized

def save_sample(is_test=False):
    """Save current canvas as a sample"""
    global canvas
    
    processed = preprocess_for_save(canvas)
    
    if processed is None:
        print("❌ Canvas trống! Vui lòng vẽ chữ cái.")
        return False
    
    # Determine save directory and count
    if is_test:
        save_dir = os.path.join(TEST_DIR, current_letter)
        count = progress[current_letter]["test"]
        dataset_type = "test"
    else:
        save_dir = os.path.join(TRAIN_DIR, current_letter)
        count = progress[current_letter]["train"]
        dataset_type = "train"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{current_letter}_{count+1}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, processed)
    
    # Update progress
    progress[current_letter][dataset_type] += 1
    save_progress()
    
    print(f"✅ Đã lưu: {dataset_type}/{current_letter}/{filename}")
    return True

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

def draw_ui(display, letter, train_count, test_count):
    """Draw UI elements on display"""
    # Header
    cv2.rectangle(display, (0, 0), (CANVAS_SIZE, 60), (240, 240, 240), -1)
    cv2.putText(display, f"Viet chu: {letter}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display, f"Train: {train_count}/{samples_target} | Test: {test_count}/{test_samples_target}", 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Footer
    cv2.rectangle(display, (0, CANVAS_SIZE - 80), (CANVAS_SIZE, CANVAS_SIZE), (240, 240, 240), -1)
    cv2.putText(display, "SPACE: Luu (Train) | T: Luu (Test)", (10, CANVAS_SIZE - 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(display, "C: Xoa | N: Chu tiep | P: Chu truoc", (10, CANVAS_SIZE - 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(display, "S: Thong ke | ESC: Thoat", (10, CANVAS_SIZE - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return display

def show_statistics():
    """Show collection statistics"""
    total_train, total_test, completed = get_statistics()
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ THU THẬP DỮ LIỆU")
    print("=" * 60)
    print(f"Tổng mẫu train: {total_train}")
    print(f"Tổng mẫu test: {total_test}")
    print(f"Chữ đã hoàn thành (≥50 train): {len(completed)}/26")
    if completed:
        print(f"Danh sách: {', '.join(completed)}")
    print("\nChi tiết từng chữ:")
    for letter in LETTERS:
        train = progress[letter]["train"]
        test = progress[letter]["test"]
        status = "✅" if train >= samples_target else "⏳"
        print(f"  {status} {letter}: Train={train}/{samples_target}, Test={test}/{test_samples_target}")
    print("=" * 60 + "\n")

def main():
    global canvas, current_letter_idx, current_letter
    
    print("\n" + "=" * 60)
    print("HƯỚNG DẪN:")
    print("=" * 60)
    print("1. Vẽ chữ cái được hiển thị trên canvas")
    print("2. Nhấn SPACE để lưu vào train set")
    print("3. Nhấn T để lưu vào test set")
    print("4. Nhấn C để xóa và vẽ lại")
    print("5. Nhấn N/P để chuyển chữ tiếp theo/trước")
    print("6. Nhấn S để xem thống kê")
    print("7. Mục tiêu: 50 mẫu train + 10 mẫu test cho mỗi chữ")
    print("=" * 60)
    
    # Show initial statistics
    show_statistics()
    
    # Create window
    cv2.namedWindow('Data Collection')
    cv2.setMouseCallback('Data Collection', mouse_callback)
    
    while True:
        # Create display with UI
        display = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
        train_count = progress[current_letter]["train"]
        test_count = progress[current_letter]["test"]
        display = draw_ui(display, current_letter, train_count, test_count)
        
        cv2.imshow('Data Collection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Save to train set (SPACE)
        if key == ord(' '):
            if save_sample(is_test=False):
                canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
                
                # Auto move to next letter if completed
                if progress[current_letter]["train"] >= samples_target:
                    print(f"🎉 Hoàn thành {samples_target} mẫu train cho chữ {current_letter}!")
                    # Find next incomplete letter
                    for i in range(26):
                        next_idx = (current_letter_idx + i + 1) % 26
                        if progress[LETTERS[next_idx]]["train"] < samples_target:
                            current_letter_idx = next_idx
                            current_letter = LETTERS[current_letter_idx]
                            print(f"➡️  Chuyển sang chữ: {current_letter}")
                            break
        
        # Save to test set (T)
        elif key == ord('t') or key == ord('T'):
            if save_sample(is_test=True):
                canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
        
        # Clear canvas (C)
        elif key == ord('c') or key == ord('C'):
            canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
            print("🗑️  Canvas đã xóa")
        
        # Next letter (N)
        elif key == ord('n') or key == ord('N'):
            current_letter_idx = (current_letter_idx + 1) % 26
            current_letter = LETTERS[current_letter_idx]
            canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
            print(f"➡️  Chuyển sang chữ: {current_letter}")
        
        # Previous letter (P)
        elif key == ord('p') or key == ord('P'):
            current_letter_idx = (current_letter_idx - 1) % 26
            current_letter = LETTERS[current_letter_idx]
            canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
            print(f"⬅️  Quay lại chữ: {current_letter}")
        
        # Show statistics (S)
        elif key == ord('s') or key == ord('S'):
            show_statistics()
        
        # Exit (ESC)
        elif key == 27:
            break
    
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ CUỐI CÙNG")
    print("=" * 60)
    total_train, total_test, completed = get_statistics()
    print(f"✅ Tổng mẫu train: {total_train}")
    print(f"✅ Tổng mẫu test: {total_test}")
    print(f"✅ Chữ hoàn thành: {len(completed)}/26")
    print("\n💾 Dataset đã lưu tại: {DATASET_DIR}/")
    print("=" * 60)
    print("\n✨ Hoàn tất! Bạn có thể chạy script train để train model.")

if __name__ == "__main__":
    main()
