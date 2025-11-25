# Nematode-Classification

This repository contains two complete deep learning pipelines for classifying 19 nematode species from the I Nema dataset. Both pipelines extend the original I Nema baseline with stronger preprocessing, improved training strategies, and modern architectures.

The project includes two notebooks:

* **efficientnetb3-nema.ipynb** — TensorFlow and Keras
* **resnet101-nema.ipynb** — PyTorch

Each notebook loads the dataset, preprocesses images, trains the model, and outputs evaluation metrics, confusion matrices, and per class performance summaries.

---

## Overview

The I Nema dataset contains microscope images of 19 nematode species.
The original benchmark from the I Nema paper achieved **79 percent accuracy** using ResNet 101.

This project improves model performance using two architectures:

| Model          | Framework          | Accuracy          |
| -------------- | ------------------ | ----------------- |
| EfficientNetB3 | TensorFlow / Keras | **84.94 percent** |
| ResNet101      | PyTorch            | **80.80 percent** |

---

## Dataset

The dataset is publicly available at https://github.com/xuequanlu/I-Nema.

---

## Notebook 1: EfficientNetB3 (efficientnetb3-nema.ipynb)

### Framework

TensorFlow / Keras

### Architecture

* EfficientNetB3 with ImageNet weights, `include_top=False`, `pooling="avg"`
* Dense(256, ReLU, L2 regularization)
* Dropout layers
* Final Dense(19, softmax)

### Training Setup

* Image size: **300 × 300**
* Three stage progressive unfreezing:

  * Stage 1: freeze backbone
  * Stage 2: unfreeze last 80 layers
  * Stage 3: unfreeze last 120 layers
* Warm up + cosine decay learning rate scheduler
* Weight decay callback
* Best model tracking per stage
* Augmentation: horizontal and vertical flips

### Outputs

* complete_training_history.png
* confusion_matrix.png
* confusion_matrix_normalized.png
* classification_report.txt
* metrics_by_species.png

---

## Notebook 2: ResNet101 (resnet101-nema.ipynb)

### Framework

PyTorch

### Architecture

* ResNet 101 pretrained on ImageNet
* Linear output layer: 2048 → 19 classes

### Training Setup

* Image size: **224 × 224**
* Weighted CrossEntropyLoss
* WeightedRandomSampler for balanced batches
* Augmentation: rotation, flips, color jitter, resized crop
* ReduceLROnPlateau scheduler
* Gradient clipping and early stopping

### Outputs

* best_from_current_run.pth
* final_confusion_matrix.png
* per_class_analysis.png
* final_evaluation_report.txt

---

## Installation

### TensorFlow setup

```
pip install tensorflow scikit-learn numpy matplotlib seaborn opencv-python Pillow tqdm
```

### PyTorch setup

```
pip install torch torchvision scikit-learn numpy matplotlib seaborn
```

---

## Running the Notebooks

### Option A: Google Colab

1. Open Colab
2. Upload `efficientnetb3-nema.ipynb` or `resnet101-nema.ipynb`
3. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Install dependencies
5. Set dataset paths inside the notebook
6. Run all cells

### Option B: Local (Jupyter)

1. Install dependencies
2. Launch:

```
jupyter notebook
```

3. Open the desired notebook
4. Adjust dataset paths
5. Run all cells

---

## Results Summary

| Model          | Accuracy          | Weighted F1 |
| -------------- | ----------------- | ----------- |
| EfficientNetB3 | **84.94 percent** | 0.85        |
| ResNet101      | **80.07 percent** | 0.80        |

EfficientNetB3 achieves the highest performance and significantly exceeds the original I Nema baseline.

---

## Repository Structure

```
.
├── efficientnetb3-nema.ipynb
├── resnet101-nema.ipynb
└── README.md
```

---

## Acknowledgments

* Lu et al. (2021) I Nema Dataset
* TensorFlow, Keras, and PyTorch frameworks


