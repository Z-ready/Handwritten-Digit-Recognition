---

## 📝 Handwritten Digit Recognition

This project implements **Handwritten Digit Recognition** using deep learning. It is based on the **MNIST dataset**, and a Convolutional Neural Network (CNN) is used for classification.

### 🚀 Features
- Uses **PyTorch** for deep learning.
- Trained on the **MNIST dataset** (28×28 grayscale images).
- Supports model evaluation and inference on custom images.
- Simple command-line interface for training and testing.

---

## 📦 Installation
Make sure you have Python installed (recommended version: **Python 3.8+**). 



---

## 📊 Dataset
This project uses the **MNIST dataset**, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The dataset will be automatically downloaded when running the training script.

---

## 🏗️ Model Architecture
The HandWrite CNN consists of:

Convolutional Layer 1: 1 → 32 filters, 3×3 kernel, ReLU activation
Convolutional Layer 2: 32 → 64 filters, 3×3 kernel, ReLU activation
Max Pooling: 2×2 (Reduces spatial size: 28×28 → 14×14)
Convolutional Layer 3: 64 → 128 filters, 3×3 kernel, ReLU activation
Convolutional Layer 4: 128 → 128 filters, 3×3 kernel, ReLU activation
Max Pooling: 2×2 (Reduces spatial size: 14×14 → 7×7)
Flatten Layer: Converts feature maps to a 1D vector
Fully Connected Layer 1: 128×7×7 → 256 neurons, ReLU activation
Dropout (0.5): Prevents overfitting
Fully Connected Layer 2: 256 → 128 neurons, ReLU activation
Output Layer: 128 → 10 (Softmax for classification of digits 0-9)
Activation Function: ReLU
Regularization: Dropout (0.5)
Output: 10 classes (digits 0-9)
---

## 🔥 Training
To train the model, run:
```bash
python train.py
```
By default, the model will train for **10 epochs** using the Adam optimizer.

---

## 🧪 Testing
To test the trained model on the test dataset:
```bash
python test.py
```

You can also predict a custom handwritten digit:
```bash
python predict.py --image path/to/your/image.png
```

---

## 🤝 Contributing
Feel free to open issues and pull requests to improve this project. If you find this useful, give it a ⭐ on GitHub!

---
