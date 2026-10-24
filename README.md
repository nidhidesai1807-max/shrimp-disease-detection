# Shrimp Disease Detection Using Deep Learning

A computer vision project exploring deep learning techniques for classifying shrimp images into health and disease categories, with model conversion workflows for mobile deployment.

## Overview

Shrimp diseases can affect aquaculture productivity and shrimp health. This project investigates CNN-based image classification to identify five shrimp health categories from images.

The project explores transfer learning, data augmentation, cross-validation, fine-tuning, and lightweight model formats for potential Android integration.

## Disease Classes

The main five-class experiment uses the following categories:

- BlackGill
- Fungal
- Healthy
- Vibriosis
- WSSV (White Spot Syndrome Virus)

## Dataset

The main classification experiment uses 390 images organized into class-specific folders.

| Class | Number of images |
|---|---:|
| BlackGill | 46 |
| Fungal | 48 |
| Healthy | 83 |
| Vibriosis | 59 |
| WSSV | 154 |
| **Total** | **390** |

Images are resized to 224 × 224 pixels for classification. The dataset is imbalanced, so augmentation and other imbalance-handling approaches are explored.

The notebook expects a dataset directory named `segmented_dataset_refined`. The dataset itself is not included in this repository.

## Models and Techniques

### Image Classification

The notebook explores several pretrained CNN architectures:

- **MobileNet:** Main five-class classification experiment.
- **MobileNetV2:** Alternative lightweight classification experiment.
- **EfficientNetB0:** Alternative experiment exploring regularization and augmentation.

The classification pipeline uses a pretrained feature extractor, `GlobalAveragePooling2D`, Dense layers, dropout, and a Softmax output layer.

### Training Strategies

- Transfer learning and fine-tuning
- Five-fold stratified cross-validation
- Online and offline data augmentation experiments
- Class weighting in selected experiments
- Dropout and L2 regularization
- MixUp in an alternative experiment
- Early stopping and learning-rate scheduling
- Test-Time Augmentation (TTA)

### Image Segmentation

The notebook also includes a separate PyTorch U-Net model conversion workflow. It exports a trained segmentation model to ONNX, checks inference consistency, and converts weights to FP16 precision.

The notebook does not demonstrate the complete end-to-end connection between U-Net segmentation and MobileNet classification.

## Model Conversion and Deployment

Two separate model-conversion workflows are explored:

1. **U-Net:** PyTorch checkpoint (`.pth`) → ONNX → FP16 ONNX.
2. **Classifier:** Keras model (`.keras`) → TensorFlow Lite (`.tflite`) with FP16 optimization.

The classifier conversion code checks the expected input and output shapes and performs a basic inference test. These workflows prepare models for potential Android integration; a complete Android application and end-to-end deployment are not demonstrated in this notebook.

## Results

The notebook records **73.85% average cross-validation accuracy** for one MobileNet experiment using Test-Time Augmentation.

This is a cross-validation result, not an independently verified real-world test accuracy. Other model experiments in the notebook have different results and configurations.

Further evaluation on an independent test set is needed before making final performance claims.

## Repository Structure

```text
shrimp-disease-detection/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── classifier_final.ipynb
├── models/
│   └── README.md
└── results/
    └── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/shrimp-disease-detection.git
cd shrimp-disease-detection
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the classification dataset in the directory expected by the notebook:

```text
segmented_dataset_refined/
├── BlackGill/
├── Fungal/
├── Healthy/
├── Vibriosis/
└── WSSV/
```

Update the dataset path in the notebook if your directory is located elsewhere.

### 5. Run the notebook

```bash
jupyter notebook notebooks/classifier_final.ipynb
```

Run the cells in order for the experiment you want to reproduce. The notebook contains multiple alternative experiments, and their settings and outputs should be considered separately.

## Technologies Used

- Python
- TensorFlow and Keras
- PyTorch
- MobileNet, MobileNetV2, EfficientNetB0
- U-Net
- ONNX and ONNX Runtime
- TensorFlow Lite
- NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook

## Limitations and Future Improvements

- Increase the number and diversity of labelled shrimp images.
- Address class imbalance and improve weaker per-class results.
- Evaluate on a separate, independent test dataset.
- Verify reproducibility across model configurations.
- Connect segmentation and classification into a complete inference pipeline.
- Benchmark model size, latency, and accuracy on an Android device.
- Add a documented Android inference implementation.

## Disclaimer

This is an experimental computer vision project. Predictions should not be treated as a definitive veterinary or aquaculture diagnosis without expert validation.

## Author

**Nidhi Desai**

Electronics and Communication Engineering  
Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
