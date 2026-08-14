# Task 1: Data Cleaning & Visualization

## Executive Summary

This project performs a rigorous data quality audit, statistical cleaning, and visual storytelling on the **IBM Telco Customer Churn** dataset (7,043 customers × 21 features). The pipeline identifies hidden missingness in `TotalCharges`, applies mechanism-aware imputation, treats outliers via IQR clipping, and exports publication-quality figures for portfolio review.

## Dataset Description

| Attribute | Detail |
|---|---|
| **Source** | IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) |
| **Rows** | 7,043 |
| **Features** | 21 (demographics, services, billing, churn label) |
| **Target** | `Churn` (Yes / No) |
| **Known Data Issues** | `TotalCharges` stored as string with 11 whitespace blanks (tenure = 0) |

## Methodology

### 1. Auditing
- Validated shape, dtypes, explicit/hidden missing values, and duplicates
- Classified `TotalCharges` missingness as **MNAR** (determined by `tenure = 0`)

### 2. Cleaning (`src/data_cleaning.py`)
| Step | Rationale |
|---|---|
| Strip whitespace, blank → NaN | Normalize categorical strings |
| Cast `TotalCharges` to float | Fix incorrect dtype |
| Impute tenure-0 with `MonthlyCharges` | MNAR-aware business logic |
| Mode / KNN imputation | Fallback for residual gaps |
| IQR outlier clipping | Robust treatment of extreme billing values |
| `_outlier_flag` column | Traceability for downstream modeling (Task 2) |

### 3. Visualizations (`src/visualization.py`)
Exported to `reports/figures/` at **300 DPI**:
1. Missing data heatmap (before / after)
2. TotalCharges distribution shift
3. Univariate numeric histograms/KDE
4. MonthlyCharges by Contract (count + boxplot)
5. Tenure by Churn (count + boxplot)
6. Annotated correlation heatmap

## Key Insights

- **26.5% churn rate** — significant retention opportunity for the telco
- **New customers** (tenure = 0) had blank total charges; imputed using first-month billing logic
- **Tenure** shows strong inverse relationship with churn
- **Contract type** drives monthly charge variability — month-to-month plans correlate with higher churn risk
- **No duplicate records** in the raw data

## Project Structure

```
Task 1/
├── data/
│   ├── raw/task1_data.csv
│   └── processed/task1_cleaned.csv
├── notebooks/task_analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   └── visualization.py
├── reports/figures/
├── requirements.txt
└── README.md
```

## How to Run

### Local (Jupyter)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/data_cleaning.py
jupyter notebook notebooks/task_analysis.ipynb
```

### Google Colab
1. Upload the entire `Task 1` folder to Google Drive at `MyDrive/Thiranex/Task 1`
2. Open `notebooks/task_analysis.ipynb` in Colab (File → Upload notebook, or open from Drive)
3. Run all cells — Drive mounts automatically and dependencies install on first run

**Alternative:** Clone your GitHub repo to `/content/Thiranex/` in Colab.

## QA Checklist

- [x] Raw data validated (`7,043 × 21`)
- [x] Hidden missing values identified and imputed
- [x] Correct dtypes post-cleaning (`TotalCharges` → float64)
- [x] Zero unhandled nulls
- [x] Outliers treated via IQR
- [x] Cleaned CSV saved to `data/processed/task1_cleaned.csv`
- [x] 7 figures exported at 300 DPI

## Dependencies

pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter

---

**Thiranex Virtual Internship — Task 1 | Due: 16 Aug 2026**
