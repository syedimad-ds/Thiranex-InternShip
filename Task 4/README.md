# Task 4: FinTech Transaction Fraud Detection

## Executive Summary

This end-to-end FinTech project screens transaction risk in a highly imbalanced fraud dataset. It operationalizes a time-based evaluation design, feature engineering available at transaction time, class-weighted modeling, hyperparameter tuning, and model-agnostic feature importance.

## Dataset and Risk Framing

The raw dataset contains 500,000 transactions and 25 fields. Fraud accounts for 11,680 transactions (2.34%); there are no missing values or exact duplicate rows. Because this is a rare-event screening problem, PR-AUC, recall, and false negatives are evaluated alongside ROC-AUC rather than relying on accuracy.

The pipeline does not use `transaction_id` or `customer_id` as predictors. It parses the transaction timestamp and adds hour, day-of-week, weekend, and log-transformed amount/distance/gap features that are available at scoring time.

## Methodology

- Hold out the most recent 20% of transactions, preventing future transaction information from entering training.
- Apply imputation and encoding inside `Pipeline` / `ColumnTransformer` objects.
- Compare a class-balanced Logistic Regression baseline against Histogram Gradient Boosting.
- Tune the nonlinear model with stratified cross-validation on a bounded training sample, optimizing PR-AUC; refit its selected configuration on all training observations.
- Export confusion matrix, ROC curve, precision-recall curve, class distribution, and permutation importance at 300 DPI.

## Project Layout

```
Task 4/
├── data/raw/credit_card_fraud_detection_500k.csv
├── data/processed/task4_transactions_prepared.csv
├── models/tuned_fraud_detector.joblib
├── notebooks/task_analysis.ipynb
├── reports/figures/
└── src/
    ├── data_cleaning.py
    ├── modeling.py
    ├── visualization.py
    └── run_modeling.py
```

## Run Locally

```powershell
pip install -r requirements.txt
python src\run_modeling.py
jupyter notebook notebooks\task_analysis.ipynb
```

## Google Colab

Upload the `Task 4` folder to `MyDrive/Thiranex/Task 4`, open `notebooks/task_analysis.ipynb` in Colab, and run all cells. The notebook installs dependencies, mounts Drive, and discovers the folder automatically.

## Responsible Use

This is a portfolio fraud-screening exercise, not an autonomous decision system. A production workflow should set the fraud threshold according to investigation capacity, cost of false positives, and fraud-loss impact; it should also monitor drift, fairness, and real-time latency.
