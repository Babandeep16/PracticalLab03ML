#  Practical Lab 3 – Cats vs Dogs Image Classification

This lab demonstrates binary image classification using two deep learning approaches:

1. A **Vanilla Convolutional Neural Network (CNN)** built from scratch.
2. A **Fine-Tuned VGG16** model leveraging transfer learning.

We classify a dataset of cat and dog images using TensorFlow/Keras, analyze performance, and visualize results.

---

## Project Structure

PracticalLab03/
│
├── datasets/
│ └── cats_vs_dogs_small/
│ ├── train/
│ │ ├── cats/
│ │ └── dogs/
│ ├── validation/
│ │ ├── cats/
│ │ └── dogs/
│ └── test/
│ ├── cats/
│ └── dogs/
│
├── models/ # Stores best model weights (.h5 files)
├── Lab03.ipynb # Jupyter Notebook with code and analysis
├── requirements.txt # All required dependencies
└── venv/ # Virtual environment folder




---

##  Requirements

Install required packages from `requirements.txt` using:

```bash
pip install -r requirements.txt



## Dataset

The dataset is a downsampled version from Kaggle's Dogs vs Cats, containing:

5,000 training images (2,500 cats, 2,500 dogs)

1,000 validation images (500 cats, 500 dogs)

2,000 test images (1,000 cats, 1,000 dogs)

The data is split manually into structured subfolders for training, validation, and testing.


## Key Steps

1. Dataset Loading and Preparation
Image files are programmatically copied into train, validation, and test directories.

A balanced dataset is ensured with equal number of cat and dog images in each subset.

2. Exploratory Data Analysis (EDA)
Class distributions are printed.

Sample images are visualized for both classes to inspect data quality.

3. Model 1: Vanilla CNN
A custom CNN model is built using Conv2D, MaxPooling2D, and Dense layers.

Compiled with adam optimizer and binary_crossentropy loss.

Best model saved via ModelCheckpoint.

4. Model 2: Fine-Tuned VGG16
The VGG16 base is loaded without top layers.

Custom classification head is added.

The pretrained layers are frozen initially, then optionally unfrozen for fine-tuning.

5. Training & Evaluation
Both models are trained for 10 epochs.

Evaluation is done on the test set using:

- Classification report

- Confusion matrix

- Misclassified image visualization

##  Results Comparison Table

| Metric        | Vanilla CNN | Fine-Tuned VGG16 |
|---------------|-------------|------------------|
| Accuracy      | 86%         | **95%**          |
| Precision     | 0.86        | **0.95**         |
| Recall        | 0.86        | **0.95**         |
| F1-Score      | 0.86        | **0.95**         |
| Training Time | ~8 minutes  | ~31 minutes      |

##  VGG16 significantly outperformed the vanilla CNN with better generalization and faster convergence.

## Visualizations
## Training/Validation Accuracy and Loss plots

## Confusion Matrices for both models

## Misclassified Examples (from VGG16 predictions)

These visualizations aid in model interpretability and debugging.

## Conclusion
Transfer learning with VGG16 provided substantial performance improvements over a custom CNN.

The model can be further enhanced using data augmentation, dropout, or advanced architectures like ResNet or EfficientNet.

Misclassified images showed ambiguity or poor lighting, highlighting the need for robustness in real-world deployment.


 Author
Name: Babandeep

Course: Foundations of Machine Learning Frameworks

College: Conestoga College

Lab: PracticalLab03 

