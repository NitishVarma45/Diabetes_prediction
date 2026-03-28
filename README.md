# Diabetes Prediction Model

A machine learning project to predict diabetes using the Pima Indians Diabetes Dataset.

## Dataset
- 768 patients, 9 features, binary outcome (diabetic / non-diabetic)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Project Workflow

### 1. EDA
- Analyzed distributions and box plots for all features
- Identified zero values, outliers, and skewness across features

### 2. Preprocessing
- **Glucose** — Removed 5 impossible zero rows
- **BloodPressure** — Group median imputation for zeros and value of 24
- **SkinThickness** — Group median imputation for zeros and value of 99
- **Insulin** — KNN imputation (n=5) for 48% missing zeros
- **BMI** — Group median imputation for zeros
- **Pregnancies, DiabetesPedigreeFunction, Age** — No action needed

### 3. Model Training
- Algorithm: **Random Forest Classifier**
- Train/Test Split: 80/20 (stratified)
- No scaling required (tree-based model)

### 4. Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.8235 |
| AUC-ROC | 0.8848 |
| Class 1 Recall | 0.74 |
| Class 1 F1 | 0.74 |

### 5. Optimizations
- Analyzed feature importance — Glucose and SkinThickness were top predictors
- Tuned decision threshold from 0.50 → **0.42** to improve diabetic recall from 0.62 → 0.74

## Requirements
```
pandas
numpy
scikit-learn
seaborn
matplotlib
```
