# Thiranex Data Science Virtual Internship Portfolio

## Overview

This repository contains four completed, verified data-science milestones for the **Thiranex Virtual Internship**. Each task includes a Colab-ready notebook, reusable Python modules, processed data, high-resolution visualizations, and task-specific documentation.

## Projects

| Task | Focus | Dataset | Main Deliverables |
|---|---|---|---|
| [Task 1](Task%201/README.md) | Data cleaning and visualization | IBM Telco Customer Churn | Cleaned data, missingness audit, IQR treatment, 7 figures |
| [Task 2](Task%202/README.md) | Predictive modeling | Heart Disease Classification | Leak-free pipelines, tuned Random Forest, diagnostics |
| [Task 3](Task%203/README.md) | Exploratory data analysis | Superstore Sales | Hypothesis-driven business analysis and 6 figures |
| [Task 4](Task%204/README.md) | FinTech fraud detection | 500K credit-card transactions | Temporal validation, imbalanced classification, tuned model and diagnostics |

## Repository Layout

```
Thiranex/
├── Task 1/                         # Cleaning and visualization
├── Task 2/                         # Supervised machine learning
├── Task 3/                         # Exploratory data analysis
├── Task 4/                         # End-to-end FinTech fraud project
├── requirements.txt                # Shared Python dependencies
└── README.md                       # This portfolio overview
```

Each task uses the same clean structure:

```
Task X/
├── data/raw/                       # Source dataset
├── data/processed/                 # Pipeline output
├── notebooks/                      # Executed, Colab-ready notebook
├── reports/figures/                # 300-DPI visualizations
├── src/                            # Reusable Python modules
├── models/                         # Serialized model, where applicable
├── requirements.txt
└── README.md
```

## Highlights

- **Task 1:** Identified hidden whitespace missing values in `TotalCharges`, corrected its data type, applied business-aware imputation, and produced a clean `7,043 × 22` dataset with no nulls.
- **Task 2:** Removed 723 exact duplicate clinical records before the holdout split to prevent evaluation leakage. The Logistic Regression baseline reached a holdout ROC-AUC of 0.905.
- **Task 3:** Found high Technology and Office Supplies margins (17.4% and 17.2%), low Furniture margin (2.6%), West as the leading profit region, and negative average profit at discounts of 30% or greater.
- **Task 4:** Built a fraud-screening pipeline over 500,000 transactions using a chronological holdout and PR-AUC for rare-event evaluation. The final workflow includes responsible-use guidance, model persistence, and operational diagnostics.

## Verification Status

All tasks were checked before submission:

| Check | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---:|---:|---:|---:|
| Source modules compile | Passed | Passed | Passed | Passed |
| Processed dataset has no nulls | Passed | Passed | Passed | Passed |
| Notebook cell errors | 0 | 0 | 0 | 0 |
| Required figures generated | 7 | 4 | 6 | 5 |
| Saved model smoke test | N/A | Passed | N/A | Passed |

## Run Locally

From this root folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then open the task-specific notebook or run its command-line pipeline:

```powershell
python "Task 1\src\data_cleaning.py"
python "Task 2\src\run_modeling.py"
python "Task 3\src\run_analysis.py"
python "Task 4\src\run_modeling.py"
```

Task 4 uses 500,000 transactions and may take longer than the other projects.

## Google Colab

1. Upload the entire `Thiranex` folder to Google Drive.
2. Open any task notebook in Colab.
3. If prompted, place the folder at `MyDrive/Thiranex/Task X`.
4. Run all cells. Each notebook installs its dependencies, mounts Drive, and discovers its project folder automatically.

## Notes

- The raw datasets are included to make every pipeline reproducible.
- This portfolio is for educational and internship evaluation purposes.
- The Task 4 fraud model is a screening exercise, not an automated decision system.
