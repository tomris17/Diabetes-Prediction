# Diabetes Prediction Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains a machine learning classification project designed to predict whether a patient has diabetes based on health and lifestyle indicators[cite: 7].

---

## Dataset Notice
*Note: The dataset used in this project (`diabetes_prediction_dataset.csv`)[cite: 7] can be accessed publicly on Kaggle.*

---

## Dataset Features
The dataset includes various medical and demographic attributes:
* **age**: Patient age[cite: 7].
* **hypertension**: Presence of hypertension.
* **heart_disease**: Presence of heart disease.
* **bmi**: Body mass index.
* **HbA1c_level**: Glycated hemoglobin level.
* **blood_glucose_level**: Blood glucose level.
* **gender & smoking_history**: Categorical indicators encoded via one-hot encoding[cite: 7].
* **diabetes**: Target variable (0: No, 1: Yes)[cite: 7].

---

## Project Workflow
1. **Data Preprocessing**: Handling categorical features via `pd.get_dummies`[cite: 7] and splitting the dataset into training and testing sets[cite: 7].
2. **Model Training**: Evaluating multiple algorithms and fitting an optimal `GradientBoostingClassifier`[cite: 7].
3. **Evaluation**: Measuring performance using accuracy score (~97.26%)[cite: 9], precision, recall, and F1-score.
4. **Model Persistence**: Exporting the trained model using `joblib` into `diabetes_model.pkl`[cite: 7].
5. **Web Application**: Interactive deployment layout built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/diabetes-prediction.git](https://github.com/YOUR_USERNAME/diabetes-prediction.git)
   cd diabetes-prediction
