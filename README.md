# Practical Lab 3 – Cats vs Dogs Image Classification

This lab demonstrates binary image classification using two deep learning approaches:

1. A **Vanilla Convolutional Neural Network (CNN)** built from scratch.
2. A **Fine-Tuned VGG16** model leveraging transfer learning.

We classify a dataset of cat and dog images using TensorFlow/Keras, analyze performance, and visualize results.

---

## Project Structure

```
PracticalLab03/
├── .gitignore
├── Lab03.ipynb         # Jupyter Notebook with full implementation and analysis
├── requirements.txt    # All required dependencies
├── datasets/           # Contains training, validation, and test images
├── models/             # Stores best model files (CNN and VGG16)
```

---

## Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is a downsampled version from Kaggle's Dogs vs Cats, containing:

* 5,000 training images (2,500 cats, 2,500 dogs)
* 1,000 validation images (500 cats, 500 dogs)
* 2,000 test images (1,000 cats, 1,000 dogs)

The data is structured into folders for training, validation, and testing.

> **Due to GitHub file size limits, the dataset is hosted externally:**

[Download Dataset from Google Drive](https://drive.google.com/file/d/14XcmkzEvRLGSwMvsweN4SETwUWm9vWCO/view?usp=sharing)
[Download Models from Google Drive](https://drive.google.com/file/d/16GTtgB8Hpc7uTg1-mxo_ExTJShQx_Yb3/view?usp=sharing)

After downloading, unzip them and place them in the following structure:

```bash
datasets/
├── cats_vs_dogs_small/
├── train/
├── validation/
└── test/

models/
├── best_cnn.h5
└── best_vgg.h5
```

---

## Key Steps

### 1. Dataset Loading and Preparation

* Programmatically copy images into `train`, `validation`, and `test` directories.
* Ensure class balance with equal number of cat and dog images per split.

### 2. Exploratory Data Analysis (EDA)

* Print class distribution.
* Visualize random samples from each class.

### 3. Model 1: Vanilla CNN

* Build a CNN using Conv2D, MaxPooling2D, Flatten, and Dense layers.
* Compile with `adam` optimizer and `binary_crossentropy` loss.
* Save best model using `ModelCheckpoint` callback.

### 4. Model 2: Fine-Tuned VGG16

* Load pretrained VGG16 without top layers.
* Add custom classification head (Dense and Dropout).
* Freeze base layers initially, then optionally fine-tune.

### 5. Training & Evaluation

* Train both models for 10 epochs each.
* Evaluate on test set using:

  * Classification report
  * Confusion matrix
  * Misclassified image visualization

---

## Results Comparison

| Metric        | Vanilla CNN | Fine-Tuned VGG16 |
| ------------- | ----------- | ---------------- |
| Accuracy      | 86%         | **95%**          |
| Precision     | 0.86        | **0.95**         |
| Recall        | 0.86        | **0.95**         |
| F1-Score      | 0.86        | **0.95**         |
| Training Time | \~8 minutes | \~31 minutes     |

>  **VGG16 significantly outperformed the vanilla CNN**, showing better generalization and faster convergence.

---

## Visualizations

* Training/Validation Accuracy and Loss Plots
* Confusion Matrices for both models
* Misclassified Images (from VGG16)

These visualizations aid in model interpretability and debugging.

---

## Conclusion

Transfer learning using VGG16 provided substantial performance improvements over a custom CNN.

Some misclassifications were due to ambiguous or low-quality images, emphasizing the need for robust data and preprocessing.

---

## Author

**Name:** Babandeep
**Course:** Foundations of Machine Learning Frameworks
**College:** Conestoga College
**Lab:** PracticalLab03
