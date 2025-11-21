"""Train CNN model for geometric shape recognition (circle, rectangle, triangle)"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=" * 60)
print("CNN MODEL TRAINING - GEOMETRIC SHAPES")
print("=" * 60)

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
DATASET_DIR = "shapes_dataset"
CLASSES = ["circle", "rectangle", "triangle"]
NUM_CLASSES = len(CLASSES)

def load_dataset(dataset_dir, img_size):
    """Load dataset from directory"""
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print("\nLoading dataset...")
    
    # Load training data
    for idx, shape in enumerate(CLASSES):
        train_path = os.path.join(dataset_dir, "train", shape)
        if not os.path.exists(train_path):
            raise ValueError(f"Training data not found: {train_path}")
        
        files = [f for f in os.listdir(train_path) if f.endswith('.png')]
        print(f"Loading {len(files)} training images for {shape}...")
        
        for file in files:
            img_path = os.path.join(train_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X_train.append(img)
                y_train.append(idx)
    
    # Load test data
    for idx, shape in enumerate(CLASSES):
        test_path = os.path.join(dataset_dir, "test", shape)
        if not os.path.exists(test_path):
            raise ValueError(f"Test data not found: {test_path}")
        
        files = [f for f in os.listdir(test_path) if f.endswith('.png')]
        print(f"Loading {len(files)} test images for {shape}...")
        
        for file in files:
            img_path = os.path.join(test_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X_test.append(img)
                y_test.append(idx)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    # Reshape
    X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
    X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    print(f"\nDataset loaded:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")
    
    return X_train, y_train, X_test, y_test

# Check if dataset exists
if not os.path.exists(DATASET_DIR):
    print(f"\nError: Dataset not found at '{DATASET_DIR}'")
    print("Please run 'python generate_shapes_dataset.py' first to create the dataset.")
    exit()

# Load data
X_train, y_train, X_test, y_test = load_dataset(DATASET_DIR, IMG_SIZE)

# Data Augmentation
print("\nSetting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest"
)
datagen.fit(X_train)

# Build CNN model
print("\nBuilding CNN model...")
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1), padding="same"),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Fully Connected
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss", 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=5, 
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "shapes_cnn_model_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Training
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60 + "\n")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# Evaluation
print("\n" + "=" * 60)
print("Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("=" * 60)

# Save final model
model.save("shapes_cnn_model.h5")
print("\nModel saved: shapes_cnn_model.h5")
print("Best model saved: shapes_cnn_model_best.h5")

# Plot training history
print("\nGenerating training plots...")
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('shapes_training_history.png')
print("Training history plot saved: shapes_training_history.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
