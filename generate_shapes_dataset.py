"""Generate synthetic dataset for geometric shapes (circle, rectangle, triangle)"""

import numpy as np
import cv2
import os
from tqdm import tqdm

print("=" * 60)
print("GENERATING GEOMETRIC SHAPES DATASET")
print("=" * 60)

# Configuration
IMG_SIZE = 64  # Image size
NUM_SAMPLES_PER_CLASS = 2000  # Number of samples per class
DATASET_DIR = "shapes_dataset"
CLASSES = ["circle", "rectangle", "triangle"]

# Create directories
for split in ["train", "test"]:
    for shape in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, split, shape), exist_ok=True)

def generate_circle(img_size):
    """Generate random circle"""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Random parameters
    center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
    center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
    radius = np.random.randint(img_size // 6, img_size // 3)
    thickness = np.random.choice([2, 3, 4, -1])  # -1 for filled
    
    cv2.circle(img, (center_x, center_y), radius, 255, thickness)
    
    return img

def generate_rectangle(img_size):
    """Generate random rectangle"""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Random parameters
    width = np.random.randint(img_size // 4, img_size // 2)
    height = np.random.randint(img_size // 4, img_size // 2)
    x1 = np.random.randint(5, img_size - width - 5)
    y1 = np.random.randint(5, img_size - height - 5)
    thickness = np.random.choice([2, 3, 4, -1])
    
    cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), 255, thickness)
    
    return img

def generate_triangle(img_size):
    """Generate random triangle"""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Random triangle vertices
    margin = img_size // 6
    pt1 = (np.random.randint(margin, img_size - margin), 
           np.random.randint(margin, img_size - margin))
    pt2 = (np.random.randint(margin, img_size - margin), 
           np.random.randint(margin, img_size - margin))
    pt3 = (np.random.randint(margin, img_size - margin), 
           np.random.randint(margin, img_size - margin))
    
    pts = np.array([pt1, pt2, pt3], np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    thickness = np.random.choice([2, 3, 4, -1])
    
    if thickness == -1:
        cv2.fillPoly(img, [pts], 255)
    else:
        cv2.polylines(img, [pts], True, 255, thickness)
    
    return img

def add_noise(img):
    """Add random noise and transformations"""
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
    
    # Random noise
    if np.random.random() > 0.7:
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
    
    # Random blur
    if np.random.random() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

# Generate dataset
print("\nGenerating shapes dataset...")

generators = {
    "circle": generate_circle,
    "rectangle": generate_rectangle,
    "triangle": generate_triangle
}

for shape in CLASSES:
    print(f"\nGenerating {shape}s...")
    
    # Training set (80%)
    train_count = int(NUM_SAMPLES_PER_CLASS * 0.8)
    for i in tqdm(range(train_count), desc=f"Train {shape}"):
        img = generators[shape](IMG_SIZE)
        img = add_noise(img)
        filename = os.path.join(DATASET_DIR, "train", shape, f"{shape}_{i}.png")
        cv2.imwrite(filename, img)
    
    # Test set (20%)
    test_count = NUM_SAMPLES_PER_CLASS - train_count
    for i in tqdm(range(test_count), desc=f"Test {shape}"):
        img = generators[shape](IMG_SIZE)
        img = add_noise(img)
        filename = os.path.join(DATASET_DIR, "test", shape, f"{shape}_{i}.png")
        cv2.imwrite(filename, img)

print("\n" + "=" * 60)
print("Dataset generation completed!")
print(f"Location: {DATASET_DIR}/")
print(f"Classes: {CLASSES}")
print(f"Train samples: {train_count} per class")
print(f"Test samples: {test_count} per class")
print("=" * 60)
