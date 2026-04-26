# 🖼️ Amazon Product Image Classifier

### Computer Vision System for Product Classification

---

## 🚀 Business Impact

This project transforms raw product images into a **data-driven classification system** that supports e-commerce operations.

* 🧠 Automatically classify product images into categories
* ⚡ Reduce manual labeling effort
* 📦 Improve catalog organization and searchability
* 🎯 Support scalable product onboarding

👉 Goal: **Use computer vision to automate product categorization in e-commerce systems**

---

## 🎯 Use Case

This system is designed for:

* 🛒 **E-commerce Platforms** → automatic product categorization
* 📦 **Operations Teams** → faster catalog management
* 📊 **Data Teams** → structured image-based datasets
* 🤖 **AI Pipelines** → downstream recommendation or search systems

### Example

Upload a product image →
👉 Model predicts category (e.g., electronics, fashion, home, etc.)

---

## 🧠 Project Overview

This project uses **deep learning (MobileNetV2)** to classify product images.

It combines:

* Pretrained convolutional neural networks
* Transfer learning
* Image preprocessing pipelines
* Streamlit interface for real-time prediction

👉 Result: **Lightweight and efficient image classification system**

---

## ⚙️ Features

* 🖼️ Image upload interface (Streamlit)
* 🤖 Deep learning model (MobileNetV2)
* ⚡ Fast inference (optimized architecture)
* 🎯 Real-time prediction
* 📦 Pretrained model integration
* 🧩 Modular structure (train + inference separation)

---

## 🧬 Model Details

* Architecture: **MobileNetV2**
* Framework: TensorFlow / Keras
* Approach: Transfer Learning
* Input: Product images
* Output: Predicted product category

👉 MobileNetV2 is chosen for:

* efficiency
* speed
* suitability for deployment

---

## 🧪 Training Pipeline

* Image dataset loading
* Preprocessing (resize, normalization)
* Data splitting (train/test)
* Model training (fine-tuning)
* Model export (`.h5`)

---

## 📊 Inference Workflow

1. Upload image
2. Preprocess input
3. Load trained model
4. Generate prediction
5. Display category result

---

## 🛠️ Installation

```bash
git clone https://github.com/Mst-KrgZ/amazon-computer-vision-classifier.git
cd amazon-computer-vision-classifier

pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

* Python 3.11
* TensorFlow / Keras
* Streamlit
* NumPy / Pandas
* Pillow

---

## 📂 Project Structure

```text
amazon-computer-vision-classifier/

├── app.py
├── requirements.txt
├── mobilenetv2_local.h5
│
├── train/
│   └── (training data & scripts)
│
├── notebooks/
│   └── CV.ipynb (optional)
```

---

## 👨‍💻 Author

**Mesut Karagöz**
Data Scientist

🔗 GitHub: https://github.com/Mst-KrgZ
🔗 LinkedIn: https://www.linkedin.com/in/mesut-karagöz-181733260/

---

⚠️ Note:
The trained model file (`mobilenetv2_local.h5`) is not included in this repository due to file size limitations.
The full working demo, including the model, is available on Hugging Face Spaces.


## ⚡ Final Note

> This project demonstrates how deep learning can be applied to automate visual classification tasks in real-world e-commerce scenarios.
