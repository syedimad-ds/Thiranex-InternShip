# Task 2: Heart Disease Predictive Modeling

## Executive Summary

This project predicts the presence of heart disease from patient clinical measurements. It uses a reproducible, leak-free Scikit-Learn workflow: exact duplicate records are removed before splitting, preprocessing is learned only from training folds, and the final tuned model is evaluated once on a stratified 20% holdout set.

## Dataset and Quality Audit

The raw `heart.csv` dataset contains 1,025 rows and 14 fields (13 predictors plus binary `target`). It has no null values but does have **723 exact duplicates**. Those records are removed before the 80/20 split, leaving 302 unique patient records. Removing duplicates avoids training-test leakage and prevents artificially optimistic scores.

`target = 1` denotes heart disease presence; `target = 0` denotes no heart disease.

## Modeling Approach

- Clinical continuous fields (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are median-imputed and standardized.
- Coded clinical categories (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) are most-frequent-imputed and one-hot encoded.
- Each transform is inside a `ColumnTransformer` and `Pipeline`, so it is fitted only on the training data or CV fold.
- Logistic Regression is the baseline; Random Forest and Gradient Boosting are ensemble comparisons.
- Random Forest is tuned with 5-fold stratified `GridSearchCV`, optimizing ROC-AUC on training data only.

## Deliverables

```
Task 2/
├── data/raw/heart.csv
├── data/processed/task2_cleaned.csv
├── models/tuned_random_forest.joblib
├── notebooks/task_modeling.ipynb
├── reports/figures/                 # 300-DPI diagnostic exports
└── src/
    ├── data_cleaning.py
    ├── modeling.py
    ├── visualization.py
    └── run_modeling.py
```

The workflow exports a confusion matrix, ROC curve, precision-recall curve, and tree-based feature-importance chart. The serialized model contains both preprocessing and estimator, so it can be applied directly to new rows with the same 13 input columns.

## Run Locally

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\run_modeling.py
jupyter notebook notebooks\task_modeling.ipynb
```

## Run in Google Colab

1. Upload the `Task 2` folder to `MyDrive/Thiranex/Task 2` in Google Drive.
2. Open `notebooks/task_modeling.ipynb` in Colab.
3. Run all cells. The first cell installs dependencies, mounts Drive, and discovers the project folder automatically.

## Evaluation Notes

Accuracy alone is insufficient for this clinical classification problem. The notebook reports precision, recall, F1, ROC-AUC, and PR-AUC; review recall and false negatives alongside ROC-AUC before choosing an operational threshold. This notebook is an educational modeling exercise and is not a clinical decision-support tool.
