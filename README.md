# Nhận dạng chữ số viết tay - Handwritten Digit Recognition

Chương trình nhận dạng chuỗi chữ số viết tay từ ảnh sử dụng Convolutional Neural Network (CNN) và MNIST dataset.Chương trình nhận dạng chuỗi chữ số viết tay từ ảnh sử dụng Convolutional Neural Network (CNN) và MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)

![License](https://img.shields.io/badge/License-MIT-green.svg)- Python 3.8+

- TensorFlow 2.x

## Tính năng- OpenCV

- NumPy

- Nhận dạng chuỗi chữ số viết tay từ ảnh- PIL (Pillow)

- Mô hình CNN với Data Augmentation

- Độ chính xác ~99% trên MNIST test set## Cài đặt

- Tự động phát hiện và phân đoạn chữ số

- Debug mode với ảnh trung gian```bash

- Hiển thị confidence score cho mỗi dự đoán# Clone hoặc tải project

## Yêu cầu hệ thống# Tạo virtual environment

python -m venv .venv

- Python 3.8+

- TensorFlow 2.10+# Kích hoạt virtual environment

- OpenCV 4.5+.\.venv\Scripts\activate # Windows

- NumPy, Pillow, SciPysource .venv/bin/activate # Linux/Mac

## Cài đặt# Cài đặt dependencies

pip install tensorflow opencv-python pillow numpy scipy

### 1. Clone repository```

```bash## Sử dụng

git clone <repository-url>

cd demo### 1. Huấn luyện mô hình (nếu chưa có)

```

```bash

### 2. Tạo virtual environmentpython train.py

```

````bash

# WindowsMô hình sẽ được lưu vào file `mnist_cnn_model.h5`.

python -m venv .venv

.\.venv\Scripts\activate### 2. Nhận dạng chữ số từ ảnh



# Linux/Mac```bash

python3 -m venv .venvpython test.py

source .venv/bin/activate```

````

Chương trình sẽ:

### 3. Cài đặt dependencies

- Mở cửa sổ chọn file ảnh

````bash- Phân tích và nhận dạng chữ số

pip install -r requirements.txt- Hiển thị kết quả và độ tin cậy

```- Lưu ảnh debug vào thư mục `debug_images/`



## Sử dụng## Hướng dẫn chụp/vẽ ảnh đầu vào



### 1. Huấn luyện mô hình (nếu chưa có)Để đạt độ chính xác cao nhất:



```bash **NÊN:**

python train.py

```- Viết rõ ràng, nét đậm

- Nền trắng, chữ đen (hoặc ngược lại)

Quá trình training:- Khoảng cách đều giữa các chữ số

- Thời gian: ~5-10 phút (CPU) hoặc ~2-3 phút (GPU)- Chữ số chiếm ít nhất 30% chiều cao ảnh

- Mô hình được lưu: `mnist_cnn_model.h5` (~3MB)- Tránh nhiễu, vết bẩn

- Độ chính xác test set: ~99%

 **KHÔNG NÊN:**

### 2. Nhận dạng chữ số từ ảnh

- Chữ quá mờ, quá nhỏ

```bash- Các chữ số dính sát nhau

python test.py- Ảnh bị mờ, nhiễu nhiều

```- Chữ viết nghiêng quá 15 độ



Chương trình sẽ:## Kết quả

1. Mở cửa sổ chọn file ảnh

2. Phân tích và phát hiện các chữ số- **Độ chính xác trên MNIST test set**: ~99%

3. Nhận dạng từng chữ số với độ tin cậy- **Độ chính xác trên ảnh viết tay**: ~85-95% (tùy chất lượng ảnh)

4. Hiển thị kết quả và lưu ảnh debug

## Cấu trúc project

### Ví dụ output:

````

````demo/

✓ Tải mô hình 'mnist_cnn_model.h5' thành công.├── train.py              # Script huấn luyện mô hình

✓ Đã phát hiện 7 chữ số trong ảnh.├── test.py               # Script nhận dạng chữ số

├── mnist_cnn_model.h5    # Mô hình đã train

  Chữ số 0: 1 (99.8%) | Lựa chọn 2: 4 (0.1%)├── debug_images/         # Ảnh debug

  Chữ số 1: 2 (100.0%) | Lựa chọn 2: 7 (0.0%)├── HUONG_DAN.md          # Hướng dẫn chi tiết

  Chữ số 2: 3 (98.6%) | Lựa chọn 2: 5 (1.2%)└── GIẢI_PHÁP.md          # Giải pháp các vấn đề

  ...```



==================================================## 🔧 Cấu trúc mô hình

File: my_numbers.png

Kết quả: 1234567```

✓ Nhận dạng: 7/7 chữ sốModel: Sequential

Debug images: debug_images/- Conv2D (32 filters) + BatchNorm + Conv2D (32) + BatchNorm + MaxPool + Dropout

==================================================- Conv2D (64 filters) + BatchNorm + Conv2D (64) + BatchNorm + MaxPool + Dropout

```- Flatten

- Dense (256) + BatchNorm + Dropout

## Hướng dẫn tạo ảnh đầu vào- Dense (128) + BatchNorm + Dropout

- Dense (10, softmax)

Để đạt độ chính xác cao nhất:```



### **NÊN:****Tính năng:**

- Viết rõ ràng, nét đậm (độ dày ~5-10px)

- Nền trắng/đen, chữ đen/trắng (tương phản cao)- Data Augmentation (rotation, shift, shear, zoom)

- Khoảng cách đều giữa các chữ số- Batch Normalization

- Chữ số chiếm ≥30% chiều cao ảnh- Dropout regularization

- Ảnh sạch, không nhiễu- Early Stopping

- Learning Rate Scheduling

###  **TRÁNH:**

- Chữ quá mờ, quá nhỏ (<20px)##  Debug

- Các chữ số dính sát nhau

- Ảnh bị nhiễu, vết bẩn nhiềuKhi nhận dạng sai, kiểm tra thư mục `debug_images/`:

- Chữ nghiêng quá 15°

- Nền và chữ có màu gần nhau1. `0_original.png` - Ảnh gốc

2. `1_blurred.png` - Ảnh sau khi làm mịn

### Ví dụ ảnh TỐT:3. `2_threshold.png` - Ảnh sau phân ngưỡng

```4. `3_morphology.png` - Ảnh sau morphological operations

┌─────────────────────────────┐5. `4_contours.png` - Vùng phát hiện (xanh = đúng, đỏ = loại bỏ)

│                             │6. `debug_digit_X.png` - Từng chữ số 28x28 (MÔ HÌNH NHÌN THẤY)

│    1  2  3  4  5  6  7      │

│                             │## Tài liệu tham khảo

└─────────────────────────────┘

```- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

## Hiệu suất- [OpenCV Documentation](https://docs.opencv.org/)



| Metric | MNIST Test Set | Ảnh viết tay |##  Tác giả

|--------|----------------|--------------|

| Accuracy | ~99% | 85-95% |Dự án xử lý ảnh - Nhận dạng chữ số viết tay

| Precision | ~99% | 80-90% |

| Speed | ~50ms/image | ~100ms/image |## 📄 License



*Độ chính xác trên ảnh viết tay phụ thuộc vào chất lượng ảnh đầu vào*MIT License


##  Cấu trúc mô hình

```python
Model: Sequential
┌─────────────────────────────────────────┐
│ Block 1:                                │
│  - Conv2D (32, 3x3) + ReLU + BatchNorm  │
│  - Conv2D (32, 3x3) + ReLU + BatchNorm  │
│  - MaxPooling2D (2x2)                   │
│  - Dropout (0.25)                       │
├─────────────────────────────────────────┤
│ Block 2:                                │
│  - Conv2D (64, 3x3) + ReLU + BatchNorm  │
│  - Conv2D (64, 3x3) + ReLU + BatchNorm  │
│  - MaxPooling2D (2x2)                   │
│  - Dropout (0.25)                       │
├─────────────────────────────────────────┤
│ Fully Connected:                        │
│  - Flatten                              │
│  - Dense (256) + ReLU + BatchNorm       │
│  - Dropout (0.5)                        │
│  - Dense (128) + ReLU + BatchNorm       │
│  - Dropout (0.5)                        │
│  - Dense (10) + Softmax                 │
└─────────────────────────────────────────┘
````

**Kỹ thuật sử dụng:**

- Data Augmentation (rotation ±15°, shift 15%, shear 15%, zoom 15%)
- Batch Normalization (tăng tốc training)
- Dropout Regularization (chống overfitting)
- Early Stopping (dừng khi val_loss không giảm)
- Learning Rate Scheduling (giảm LR khi plateau)

## Cấu trúc project

```
demo/
├── .venv/                # Virtual environment (không commit)
├── debug_images/         # Ảnh debug (tự động tạo)
├── train.py             # Script huấn luyện mô hình
├── test.py              # Script nhận dạng chữ số
├── mnist_cnn_model.h5   # Mô hình đã train (~3MB)
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Troubleshooting

### Vấn đề: Nhận dạng sai

**Giải pháp:** Kiểm tra thư mục `debug_images/`:

1. `0_original.png` - Ảnh gốc có rõ không?
2. `2_threshold.png` - Chữ có tách rõ khỏi nền không?
3. `4_contours.png` - Vùng phát hiện có đúng không? (xanh = hợp lệ, đỏ = loại bỏ)
4. `debug_digit_X.png` - Chữ số 28x28 có rõ ràng không?

**Nếu vẫn sai:**

- Cải thiện chất lượng ảnh đầu vào
- Viết chữ rõ ràng, nét đậm hơn
- Tăng khoảng cách giữa các chữ số

### Vấn đề: Phát hiện sai số lượng chữ số

**Nguyên nhân:**

- Chữ số dính nhau → tách rời hơn
- Có vết bẩn/nhiễu → làm sạch ảnh
- Chữ quá nhỏ → viết to hơn (≥30% chiều cao ảnh)

### Vấn đề: Import errors

```bash
# Reinstall dependencies
pip uninstall tensorflow opencv-python pillow numpy scipy
pip install -r requirements.txt
```

## Tips & Tricks

### Tăng độ chính xác:

1. Viết chữ theo chuẩn MNIST (đặc biệt số 6, 8, 9)
2. Chụp ảnh có ánh sáng tốt, không bóng mờ
3. Crop ảnh để chỉ chứa vùng chữ số
4. Tăng độ tương phản nền-chữ

### Số khó nhận dạng:

- **Số 6**: Phần trên cần có đường cong rõ, vòng dưới tròn đều
- **Số 8**: Hai vòng tròn cần rõ ràng, đều nhau
- **Số 1**: Viết thẳng, không quá nghiêng
- **Số 7**: Gạch ngang rõ ràng ở đầu

## Advanced Usage

### Fine-tune với dataset riêng:

```python
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('mnist_cnn_model.h5')

# Prepare your custom data (X_custom, y_custom)
# X_custom shape: (n_samples, 28, 28, 1)
# y_custom shape: (n_samples, 10) - one-hot encoded

# Fine-tune
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_custom, y_custom, epochs=10, batch_size=32)
model.save('mnist_cnn_model_custom.h5')
```

### Batch processing:

```python
import glob
from test import recognize_from_file

# Process multiple images
for img_path in glob.glob("images/*.png"):
    print(f"Processing: {img_path}")
    # Modify test.py to accept file_path parameter
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Tài liệu tham khảo

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Dataset gốc
- [TensorFlow Documentation](https://www.tensorflow.org/) - Framework
- [OpenCV Documentation](https://docs.opencv.org/) - Image processing
- [Keras API](https://keras.io/api/) - High-level API

## Tác giả

Dự án Xử lý Ảnh - Nhận dạng Chữ số Viết tay

## License

MIT License - Free to use for educational and commercial purposes