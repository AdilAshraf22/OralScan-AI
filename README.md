# OralScan AI 🔬

### AI-Powered Oral Cancer Screening Using Deep Learning and Explainable AI

OralScan AI is a deep learning-based web application designed for **preliminary oral cancer screening** from oral cavity images.

The system uses **MobileNetV2 transfer learning** to classify oral images into **Normal** and **Cancerous** categories. To make the predictions more interpretable, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is integrated to generate a visual heatmap showing the regions that influenced the model's prediction.

The trained model is integrated into a **Flask web application**, allowing users to upload an image and receive a prediction, confidence score, and Grad-CAM visualization through a simple web interface.

> **Important:** OralScan AI is intended as a preliminary screening/research tool and is **not a replacement for professional medical diagnosis**.

---

## 📌 Project Overview

Early detection of oral cancer can improve the chances of timely clinical intervention. However, access to specialized screening and medical expertise can be limited.

OralScan AI explores how deep learning and explainable AI can be combined into a lightweight web-based screening system.

The complete workflow is:

```text
User uploads oral image
        ↓
Image preprocessing
        ↓
MobileNetV2-based CNN
        ↓
Normal / Cancerous prediction
        ↓
Confidence score
        ↓
Grad-CAM explanation
        ↓
Result displayed through web interface
```

---

## ✨Key Features

- **Deep Learning Classification**
  - MobileNetV2 transfer learning for oral image classification.

- **Binary Classification**
  - Classifies images into:
    - Normal
    - Cancerous

- **Grad-CAM Explainability**
  - Generates a heatmap highlighting image regions that influenced the model's prediction.

- **Flask Web Application**
  - Simple browser-based interface for image upload and result visualization.

- **Lightweight Architecture**
  - MobileNetV2 enables relatively efficient inference without requiring specialized hardware.

- **Performance Evaluation**
  - Evaluated using accuracy, precision, recall, and F1-score.

---

## 🧠 Machine Learning Approach

### MobileNetV2

The project uses **MobileNetV2** as the backbone architecture through transfer learning.

MobileNetV2 was selected because of its lightweight architecture and suitability for image classification with comparatively limited labeled data.

The model learns visual patterns from oral images and performs binary classification between normal and cancerous categories.

### Image Preprocessing

The image processing pipeline includes:

- Image resizing
- Pixel normalization
- Data augmentation
- Random rotations
- Zoom augmentation
- Horizontal flipping

These techniques help introduce visual variation during training and improve the model's ability to handle unseen images.

---

## 🔥 Explainable AI with Grad-CAM

A major component of OralScan AI is **Grad-CAM**.

Instead of returning only a classification such as:

```text
Cancerous
```

the system also produces a heatmap indicating the regions that contributed to the model's decision.

```text
Input Image
     ↓
MobileNetV2
     ↓
Prediction
     ↓
Grad-CAM
     ↓
Heatmap Overlay
```

This provides a visual explanation of the model's prediction and makes the system more interpretable.

---

## 📊 Model Performance

The model was evaluated using standard classification metrics.

| Metric | Result |
|---|---:|
| Accuracy | **94.1%** |
| Precision | **92.5%** |
| Recall | **93.8%** |
| F1-Score | **93.1%** |

These results are reported in the associated research paper on the held-out validation set.

The project evaluates multiple metrics rather than relying only on accuracy, which is particularly important for medical image classification.

---

## 🏗️ System Architecture

```text
                    ┌───────────────────┐
                    │       User        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Web Interface   │
                    │  HTML / CSS / JS  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Flask Backend   │
                    │      app.py       │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Image Preprocessing│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    MobileNetV2    │
                    │   Trained Model   │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    Prediction     │
                    │ Normal/Cancerous  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │     Grad-CAM      │
                    │  Heatmap Output   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Prediction +       │
                    │ Confidence +       │
                    │ Heatmap            │
                    └────────────────────┘
```

---

## 🛠️ Technologies Used

### Programming & Frameworks

- Python
- Flask
- HTML
- CSS
- JavaScript

### Machine Learning

- TensorFlow
- Keras
- MobileNetV2
- Convolutional Neural Networks
- Transfer Learning
- Grad-CAM

### Image Processing

- OpenCV
- NumPy
- Image preprocessing and augmentation

### Development Tools

- Git
- GitHub
- Jupyter Notebook
- VS Code

---

## 📂 Project Structure

```text
OralScan-AI/
│
├── model/
│   ├── oral_model.h5
│   ├── train.py
│   ├── evaluate.py
│   └── class_indices.json
│
├── templates/
│   └── index.html
│
├── app.py
│
├── .gitignore
│
└── README.md
```

### Dataset

The original training dataset is **not included in this repository**.

The dataset was intentionally excluded from GitHub because of its size and because the project uses oral/medical images.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AdilAshraf22/OralScan-AI.git
```

```bash
cd OralScan-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

Install the required Python libraries used by the project:

```bash
pip install flask tensorflow keras numpy opencv-python pillow
```

### 5. Run the Application

```bash
python app.py
```

The Flask server will start locally.

Open the URL shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

## 🖥️ Application Workflow

1. Open the OralScan AI web application.
2. Upload an oral cavity image.
3. The Flask backend receives the image.
4. The image is preprocessed.
5. The MobileNetV2 model performs classification.
6. The system generates a confidence score.
7. Grad-CAM generates a heatmap.
8. The prediction and visualization are displayed to the user.

---

## 🔬 Research Paper

The OralScan AI project is also documented in an IEEE-style research paper:

### **“AI-Based Oral Cancer Detection Using Machine Learning and Pattern Analysis”**

**Authors:**

- Shri Sidhaarth
- Adil Ashraf
- Dr. Nithiya S

**Department of Computing Technologies**  
**SRM Institute of Science and Technology, Chennai, India**

The paper presents the OralScan framework, including the MobileNetV2 classification approach, Grad-CAM explainability, web-based implementation, system architecture, and experimental evaluation.

The research reports:

- 94.1% validation accuracy
- 92.5% precision
- 93.8% recall
- 93.1% F1-score

The paper also discusses future extensions including mobile deployment, larger and more diverse datasets, multi-class oral lesion classification, cloud integration, improved explainability, and clinical validation.

---

## 📈 Research Contributions

The project combines four major components:

### 1. Transfer Learning

A MobileNetV2-based classifier is used to perform oral image classification without requiring a large model to be trained from scratch.

### 2. Explainable AI

Grad-CAM is integrated directly into the prediction workflow to provide visual explanations of model decisions.

### 3. Web-Based Deployment

The trained model is integrated into a Flask web application, allowing users to interact with the model through a browser.

### 4. Multi-Metric Evaluation

The system is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

rather than relying solely on accuracy.

---

## 🔮 Future Improvements

Potential future extensions include:

- 📱 Mobile application for real-time image capture
- 📚 Larger and more diverse training datasets
- 🧠 Multi-class classification of specific oral lesions
- ☁️ Cloud-based deployment
- 🔬 Additional explainability techniques
- 🏥 Clinical validation with medical professionals
- 📊 Improved model architectures and ensemble approaches
- 📷 Real-time camera-based screening

---

## ⚠️ Medical Disclaimer

OralScan AI is an **academic/research project for preliminary screening and demonstration purposes**.

It should **not be used as a substitute for examination, diagnosis, or treatment by a qualified healthcare professional**.

A model prediction does not constitute a medical diagnosis.

---

## 👨‍💻 Authors

**Adil Ashraf**  
B.Tech Computer Science and Engineering  
SRM Institute of Science and Technology

**Shri Sidhaarth**  
Department of Computing Technologies  
SRM Institute of Science and Technology

**Dr. Nithiya S**  
Assistant Professor, Department of Computing Technologies  
SRM Institute of Science and Technology

---

## ⭐ Project Highlights

```text
MobileNetV2
     +
Transfer Learning
     +
Grad-CAM
     +
Flask
     +
Web Interface
     +
Medical Image Classification
     =
OralScan AI
```

---

## 📜 License

This project is licensed under the MIT License.

---

⭐ If you find this project interesting, feel free to explore the repository and the associated research work.
