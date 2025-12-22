# SMUL_GA03  
**Environmental Sound Classification using UrbanSound8K**

This repository contains the implementation developed for **Assignment GA3** of the *Multimedia Systems* course.  
The goal of this project is to perform **environmental sound classification** using classical machine learning techniques and audio feature extraction.

---

## Project Overview

Environmental sound classification aims to automatically recognize sound events such as *dog barking, sirens, drilling,* or *engine idling*.  
In this project, we use the **UrbanSound8K** dataset and a **supervised learning pipeline** composed of:

- Audio preprocessing and feature extraction (MFCCs)
- Classical machine learning classifiers (SVM and Random Forest)
- Hyperparameter tuning
- Cross-validation using the official dataset folds
- Quantitative evaluation with standard metrics

---
```
SMUL_GA03/
│
├── main.py # Main
│
├── config.py # Global experiment configuration and logging
│
├── data.py # Dataset loading, metadata, and audio handling
├── features.py # MFCC feature extraction
│
├── svm_model.py # Support Vector Machine + hyperparameter tuning
├── rf_model.py # Random Forest + hyperparameter tuning
│
├── evaluation.py # Evaluation metrics and prediction timing
├── plotting.py # Plots and confusion matrices
│
├── results/ # Generated CSV files and plots
│
├── requirements.txt # Python dependencies
└── README.md
```


## Methodology

### Feature Extraction
- **MFCCs (Mel-Frequency Cepstral Coefficients)**
- First-order (delta) and second-order (delta-delta) temporal derivatives
- Statistical aggregation using mean and standard deviation

### Models
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

### Hyperparameter Tuning
- Grid search performed **independently for each fold**
- Selection based on **macro F1-score**

### Evaluation Protocol
- Official **10-fold cross-validation** provided by UrbanSound8K
- Metrics:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-score (macro)

---

## Output Files

All experimental outputs are saved in the `results/` folder:

- `fold_metrics_*.csv` – performance metrics per fold
- `mean_confusion_matrix_*.csv` – averaged confusion matrices
- `*_best_params_per_fold.csv` – selected hyperparameters per fold
- `.png` files – plots and confusion matrices

---

## How to Run

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

4. Run:

   ```bash
   python main.py
