"""Train custom handwriting recognition model on collected data"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json

print("=" * 60)
print("TRAIN CUSTOM HANDWRITING MODEL")
print("=" * 60)

# Settings
DATASET_DIR = "custom_handwriting_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_SAVE_PATH = "custom_letters_model.h5"
PRETRAINED_MODEL = "emnist_letters_pretrained.h5"

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_dataset(data_dir):
    """Load images from directory structure"""
    images = []
    labels = []
    
    print(f"\n📂 Loading data from {data_dir}...")
    
    for label_idx, letter in enumerate(LETTERS):
        letter_dir = os.path.join(data_dir, letter)
        
        if not os.path.exists(letter_dir):
            print(f"⚠️  Warning: {letter} directory not found")
            continue
        
        files = [f for f in os.listdir(letter_dir) if f.endswith('.png')]
        
        for filename in files:
            filepath = os.path.join(letter_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize to 28x28
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Invert if needed (white on black)
                if np.mean(img) > 127:
                    img = 255 - img
                
                images.append(img)
                labels.append(label_idx)
        
        print(f"  {letter}: {len(files)} samples")
    
    if len(images) == 0:
        return None, None
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\n✅ Loaded {len(images)} images")
    return images, labels

def preprocess_data(X, y):
    """Preprocess images and labels"""
    # Reshape and normalize
    X = X.reshape(X.shape[0], 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y = to_categorical(y, 26)
    
    return X, y

def create_model(use_pretrained=True):
    """Create or load model"""
    
    if use_pretrained and os.path.exists(PRETRAINED_MODEL):
        print(f"\n🔄 Loading pretrained model: {PRETRAINED_MODEL}")
        print("   Using transfer learning from EMNIST model...")
        
        base_model = load_model(PRETRAINED_MODEL)
        
        # Freeze early layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        print("✅ Pretrained model loaded (last 4 layers trainable)")
        return base_model
    
    else:
        print("\n🏗️  Creating new model from scratch...")
        
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            
            # Dense layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(26, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ New model created")
        return model

def main():
    # Check if dataset exists
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Dataset not found at {TRAIN_DIR}")
        print("   Please run collect_handwriting_data.py first!")
        return
    
    # Load training data
    X_train, y_train = load_dataset(TRAIN_DIR)
    
    if X_train is None:
        print("❌ No training data found!")
        return
    
    # Load test data if available
    X_test, y_test = None, None
    if os.path.exists(TEST_DIR):
        X_test, y_test = load_dataset(TEST_DIR)
    
    # If no test data, split from training data
    if X_test is None or len(X_test) < 10:
        print("\n⚠️  No separate test set, splitting from training data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        print(f"✅ Split: {len(X_train)} train, {len(X_test)} test")
    
    # Preprocess
    print("\n🔧 Preprocessing data...")
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    print(f"✅ Training set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # Data Augmentation
    print("\n🔄 Setting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Create model
    use_transfer = os.path.exists(PRETRAINED_MODEL)
    model = create_model(use_pretrained=use_transfer)
    
    print("\n📊 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 60)
    print("🎯 TRAINING MODEL")
    print("=" * 60)
    print(f"Strategy: {'Transfer Learning' if use_transfer else 'From Scratch'}")
    print(f"Epochs: 100 (with early stopping)")
    print(f"Batch size: 32")
    print("-" * 60)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("📊 EVALUATION")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"✅ Test loss: {test_loss:.4f}")
    
    # Save final model
    model.save(MODEL_SAVE_PATH.replace('.h5', '_final.h5'))
    
    print("\n" + "=" * 60)
    print("💾 MODELS SAVED")
    print("=" * 60)
    print(f"   - {MODEL_SAVE_PATH} (best validation accuracy)")
    print(f"   - {MODEL_SAVE_PATH.replace('.h5', '_final.h5')} (final model)")
    
    # Save training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    history_plot = 'custom_model_training_history.png'
    plt.savefig(history_plot)
    print(f"\n📈 Training history saved: {history_plot}")
    
    print("\n" + "=" * 60)
    print("✨ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📝 Next steps:")
    print(f"   1. Check training history: {history_plot}")
    print(f"   2. Test your model with: python test_letters.py")
    print(f"   3. Or use: python draw_letters.py")
    print(f"\n💡 Update test files to use: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
