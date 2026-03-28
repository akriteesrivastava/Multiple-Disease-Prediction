# Multiple Disease Prediction

A machine learning project for predicting multiple diseases from tabular medical datasets.

This project currently supports:
- Diabetes Risk
- Breast Cancer
- Heart Disease

The project has two execution modes:
- CLI evaluation mode through `main.py`
- Optional Streamlit app through `streamlit_app.py`

## Project Highlights

- Uses real CSV datasets stored in the `Dataset` folder
- Trains separate disease models for diabetes, breast cancer, and heart disease
- Uses a stacked ensemble model for prediction
- Applies preprocessing for numeric and categorical features
- Prints evaluation metrics including accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix, and feature importance

## Model Used

The final prediction pipeline uses a stacked ensemble consisting of:
- Random Forest
- Calibrated SVM
- KNN
- Logistic Regression as the stacking meta-model

## Folder Structure

```text
Project_1_Multiple_disease_prediction/
├── Dataset/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── breast_cancer_wisconsin.csv
├── models/
│   ├── diabetes_risk_model_v5.joblib
│   ├── breast_cancer_model_v4.joblib
│   └── heart_disease_model_v4.joblib
├── main.py
├── model_utils.py
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## Datasets

Place the following dataset files inside the `Dataset` folder:
- `diabetes.csv`
- `heart.csv`
- `breast_cancer_wisconsin.csv`

Expected target columns:
- Diabetes: `Outcome`
- Heart Disease: `HeartDisease`
- Breast Cancer: `diagnosis`

Notes:
- `id` is dropped from breast cancer inputs
- `Unnamed: 32` is dropped from breast cancer data if present
- invalid zero values in some diabetes columns are treated as missing values before training

## Requirements

- Python 3.11 or 3.12 recommended
- Windows, Linux, or macOS

Python packages used:
- datasets
- joblib
- pandas
- scikit-learn
- streamlit

Install them with:

```powershell
pip install -r requirements.txt
```

## Recommended Setup On A New Laptop

Do not copy `.venv` from one laptop to another.

Instead:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then run the project normally.

## How To Run

### 1. Run CLI Evaluation

This is the current primary workflow.

```powershell
python main.py
```

What it does:
- checks that datasets exist
- loads or trains the required models
- prints evaluation results for all diseases

The CLI output includes:
- train/test split
- cross-validation method
- ensemble metrics
- confusion matrix
- model comparison
- top feature importance

### 2. Run Streamlit App

If you want the interactive app:

```powershell
python -m streamlit run streamlit_app.py
```

The Streamlit app allows:
- choosing a disease model
- entering feature values
- getting a predicted result and probability
- viewing training summary metrics

## Training Strategy

The current training pipeline uses:
- 70:30 train/test split
- Stratified 5-fold cross-validation
- disease-specific preprocessing
- stacking ensemble for final prediction

### Diabetes Feature Engineering

For the diabetes dataset, the project also creates engineered features such as:
- interaction features
- ratio features
- risk indicator flags
- age, BMI, and glucose bucket groups

## Current Evaluation Snapshot

From the latest verified run:

### Diabetes Risk
- Accuracy: 0.7619
- Precision: 0.6444
- Recall: 0.7160
- F1 Score: 0.6784
- ROC AUC: 0.8512

### Breast Cancer
- Accuracy: 0.9766
- Precision: 1.0000
- Recall: 0.9375
- F1 Score: 0.9677
- ROC AUC: 0.9980

### Heart Disease
- Accuracy: 0.9094
- Precision: 0.9051
- Recall: 0.9346
- F1 Score: 0.9196
- ROC AUC: 0.9531

These values may change slightly if models are retrained on another machine or with updated package versions.

## Important Notes

- This is an educational and portfolio project
- It is not a medical device
- It should not be used for real clinical decisions
- Evaluation results depend on the quality and limitations of the datasets used

## Moving The Project To Another Laptop

Recommended approach:
- copy the project folder
- keep the `Dataset` folder
- optionally keep the `models` folder for faster reuse
- create a fresh `.venv` on the new laptop
- reinstall dependencies with `requirements.txt`
- run `python main.py` to verify everything works

If the copied model files fail to load on the new laptop, delete the `models` folder and rerun:

```powershell
python main.py
```

The project will retrain the models.

## GitHub Upload Checklist

Before pushing to GitHub:
- keep `README.md`
- keep `requirements.txt`
- keep source code
- keep `Dataset` only if licensing allows public distribution
- avoid uploading `.venv`
- avoid uploading `__pycache__`
- verify commands in the README actually work

## Possible Future Improvements

- threshold tuning for diabetes prediction
- hyperparameter tuning for all base models
- ROC and confusion matrix plots
- saved evaluation reports in text or CSV format
- deployment to Streamlit Community Cloud or another hosting platform

## License / Data Usage

Before publishing publicly, check the license terms of the datasets you downloaded from Kaggle or Hugging Face. Some datasets may require attribution or may have reuse conditions.

## Author

Project by Akritee Srivastava
