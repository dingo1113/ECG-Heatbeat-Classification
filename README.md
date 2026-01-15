[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/dingo1113/ECG-Heartbeat-Classification/blob/main/ecg_stft_image_detection.ipynb)

# ECG Arrhythmia Classification Using STFT and Deep Learning

This project implements an end-to-end deep learning pipeline for classifying
ECG heartbeats using time-frequency representations and convolutional neural
networks (CNNs). The system processes raw ECG signals, converts them into
spectrogram images using Short-Time Fourier Transform (STFT), and applies
CNN-based classification.

## Overview
The goal of this project is to detect and classify cardiac arrhythmias from
ECG signals using signal processing and machine learning techniques.

Key components include:
- ECG signal preprocessing and filtering
- STFT-based spectrogram generation
- Image-based CNN classification
- Model evaluation using multiple performance metrics

## Data
- ECG data sourced from the MIT-BIH Arrhythmia Database
- Tens of thousands of individual heartbeats processed and labeled
- Data split by record to prevent patient-level data leakage
- Dataset not included in this repository due to size constraints

## Signal Processing
- Applied band-pass filtering to isolate relevant ECG frequencies
- Used Short-Time Fourier Transform (STFT) to convert ECG signals into
  time-frequency spectrograms
- Spectrograms resized and normalized for CNN input

## Model
- Convolutional Neural Network (CNN) for image-based classification
- Implemented in TensorFlow
- Trained on STFT spectrogram representations of ECG signals
- Designed for extensibility to additional arrhythmia classes

## Evaluation
- Model performance evaluated using:
  - Classification accuracy
  - Validation loss
  - Area Under the ROC Curve (AUC)
- Training and validation metrics tracked across epochs

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- SciPy
- OpenCV
- Google Colab

## Repository Contents
- `ecg_stft_image_detection.ipynb` — End-to-end notebook including signal
  preprocessing, spectrogram generation, model training, and evaluation

## Notes
- Dataset files are excluded due to size and licensing constraints
- Notebook developed and executed in Google Colab
- Project focuses on educational and research applications

## Future Improvements
- Class imbalance handling using weighted loss functions
- Cross-patient generalization experiments
- Model optimization for real-time or embedded deployment

