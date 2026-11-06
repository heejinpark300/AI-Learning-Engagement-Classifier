# AI-Learning-Engagement-Classifier
A MobileNetV2-based deep learning prototype for classifying engagement-related behaviors in learning-scene images.
# AI Learning Engagement Classifier

### A MobileNetV2-Based Deep Learning Prototype for Classifying Learning-Scene Images

**Project Type:** Deep Learning · Computer Vision · Educational AI  
**Developer:** Heejin Park, Ph.D.  
**Date:** November 2026  
**Status:** Educational Prototype

## 1. Project Overview

This project explores the use of deep learning to classify learning-scene images into two dataset-defined categories: **Engagement** and **Disengagement**.

The application uses MobileNetV2 transfer learning and provides an interactive interface through Streamlit.

The project investigates how computer vision techniques could contribute to exploratory analyses of observable behaviors in educational settings.

## 2. Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- NumPy
- scikit-learn
- Streamlit

## 3. Dataset and Preprocessing

The initial dataset contained 16,000 images.

The preprocessing workflow included:

1. Identifying and removing exact duplicate images.
2. Detecting and excluding conflicting labels.
3. Identifying near-duplicate images.
4. Splitting the cleaned dataset into training, validation, and test sets.

**Final dataset: 11,159 images**

| Dataset | Images |
|---|---:|
| Training | 8,928 |
| Validation | 1,115 |
| Test | 1,116 |

## 4. Model Development

The classifier was developed using MobileNetV2 pretrained on ImageNet, with a custom binary classification head.

- Input resolution: 224 × 224 pixels
- Output classes: Engagement / Disengagement
- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Training strategy: Transfer Learning with Early Stopping

## 5. Model Performance

The model achieved **69.53% test accuracy** on 1,116 held-out images.

| Metric | Result |
|---|---:|
| Test Accuracy | 69.53% |
| Engagement Recall | 57.95% |
| Test Samples | 1,116 |

The `results/` directory contains the confusion matrix and supporting evaluation outputs.

## 6. Streamlit Application

The Streamlit application allows users to:

- Upload an image (JPG, JPEG, or PNG).
- Classify the image as Engagement or Disengagement.
- View model-generated prediction scores.

**Application entry point:** `engagement_app.py`

## 7. Educational Significance and Limitations

This project demonstrates the technical feasibility of applying transfer learning to the classification of observable learning-scene images.

However, visual classification should not be interpreted as a validated measurement of a student's cognitive engagement, motivation, or attention.

Further research should address dataset validity, subject-independent evaluation, generalizability, and ethical considerations before classroom implementation.

## 8. Future Directions

Potential extensions include:

- More diverse and representative datasets.
- Improved classification performance and error analysis.
- Human-in-the-loop evaluation.
- Multimodal analysis combining behavioral and learning-process data.
- Privacy-aware applications in educational research.

---

**Author:** Heejin Park, Ph.D.  
**Research Interests:** AI in Education · Human–AI Interaction · AI Literacy · Educational Technology
