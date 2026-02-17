
# 🩸 HematoVision

## Advanced Blood Cell Classification Using Transfer Learning

### 📌 Category

Artificial Intelligence | Deep Learning | Computer Vision

---

## 📖 Project Overview

**HematoVision** is an AI-powered web application designed to classify blood cells using **Transfer Learning**. The system leverages a pre-trained **MobileNetV2 Convolutional Neural Network (CNN)** model to accurately classify microscopic blood cell images into four categories:

* Eosinophil
* Lymphocyte
* Monocyte
* Neutrophil

By utilizing transfer learning, the model benefits from previously learned image features, significantly improving accuracy while reducing training time and computational cost.

The system is integrated with a **Flask-based web application**, allowing users to upload blood cell images and receive instant predictions.

---

## 🎯 Project Objectives

By completing this project, you will:

* Understand fundamental **Deep Learning concepts**
* Learn how **Transfer Learning** improves model performance
* Perform **data preprocessing and augmentation**
* Build and evaluate a CNN-based classification model
* Deploy a trained model using **Flask**
* Create a functional AI-powered web application

---

## 🧠 Technologies Used

### Programming Language

* Python 3.x

### Deep Learning Framework

* TensorFlow / Keras

### Model Architecture

* MobileNetV2 (Pre-trained CNN)

### Web Framework

* Flask

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* Scikit-learn
* Pillow

---

## 📂 Dataset Information

* **Source:** Kaggle
* **Total Images:** ~12,500
* **Classes:** 4
* **Images per Class:** ~3,000

### Cell Types:

1. Eosinophil
2. Lymphocyte
3. Monocyte
4. Neutrophil

Dataset Link:
[https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data)

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

* Download dataset from Kaggle
* Extract images into respective class folders

### 2️⃣ Data Preprocessing

* Image resizing
* Normalization
* Label encoding
* Train-test split

### 3️⃣ Data Augmentation

* Rotation
* Zoom
* Horizontal flip
* Shear transformation

### 4️⃣ Model Building

* Load pre-trained MobileNetV2
* Freeze base layers
* Add:

  * Flatten layer
  * Dropout layer
  * Dense layer (SoftMax)
* Compile using:

  * Optimizer: Adam
  * Loss: Categorical Crossentropy

### 5️⃣ Model Training

* Train for 5 epochs
* Use EarlyStopping
* Save best model as:

  ```
  blood_cell.h5
  ```

### 6️⃣ Model Evaluation

* Accuracy
* Loss curves
* Confusion matrix
* Classification report

### 7️⃣ Application Development

* Create HTML pages:

  * home.html
  * predict.html
* Build Flask backend (app.py)
* Load saved model
* Deploy locally

---

## 🏗 Architecture Flow

```
User 
   ↓
Upload Image (Web UI)
   ↓
Flask Backend
   ↓
Image Preprocessing
   ↓
MobileNetV2 Transfer Learning Model
   ↓
Prediction
   ↓
Display Result
```

---

## 📁 Project Structure

```
HematoVision/
│
├── templates/
│   ├── home.html
│   ├── results.html
│
├── static/
│
├── blood_cell.h5
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/Kavya14345/HematoVision.git
cd HematoVision
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install tensorflow flask numpy pandas matplotlib seaborn scikit-learn pillow opencv-python
```

### Step 3: Run Application

```bash
python app.py
```

### Step 4: Open Browser

```
http://127.0.0.1:5000/
```

---

## 🧪 Model Performance

* Training Accuracy: ~95–98%
* Validation Accuracy: ~93–96%
* Loss minimized using EarlyStopping
* Transfer learning significantly reduced training time

---

## 🚀 Use Case Scenarios

### 🏥 1. Automated Clinical Diagnostics

* Faster blood cell classification
* Reduced manual workload
* High diagnostic accuracy

### 🌐 2. Telemedicine Applications

* Remote blood image upload
* Instant AI-powered analysis
* Accessible healthcare

### 🎓 3. Medical Education

* Interactive learning tool
* Morphology understanding
* Practical training support

---

## 🔮 Future Enhancements

* Add more blood cell categories
* Integrate with hospital databases
* Deploy on cloud (AWS / Azure)
* Build mobile application
* Real-time microscope camera integration

---

## 📌 Prerequisites

You should have basic knowledge of:

* Neural Networks
* Convolutional Neural Networks (CNN)
* Transfer Learning
* Overfitting & Regularization
* Deep Learning Optimizers
* Flask basics

---

## 👨‍💻 Author

Developed as part of Artificial Intelligence / Deep Learning guided project.

---

## 📜 License

This project is for educational and research purposes only.

---
