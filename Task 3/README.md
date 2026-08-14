# Task 3: Superstore Exploratory Data Analysis

## Executive Summary

This project turns the Superstore order-level dataset into five decision-focused analyses. It shows that Technology has the greatest profit contribution and a 17.4% margin, Furniture is materially less profitable at 2.6%, the West leads regional profit, and higher discount levels consistently destroy average profit.

## Dataset and Preparation

`data/raw/task3_data.csv` contains 10,194 order line items and 21 source columns, with no missing values or exact duplicate rows. The preparation step parses order and ship dates, validates numeric fields, and derives order year/month/quarter, shipping days, and profit margin. The prepared dataset is saved as `data/processed/task3_prepared.csv`.

## Business Questions

1. Which categories provide the strongest profit margin?
2. Do sales exhibit a repeatable monthly or quarterly pattern?
3. Which customer segment has the greatest average order value?
4. At what point does discounting turn average profit negative?
5. Which regions lead total profit and profit margin?

## Key Findings

- Technology produces the highest total profit ($146.5K) and a 17.4% profit margin.
- Office Supplies has a similarly strong 17.2% margin; Furniture trails dramatically at 2.6%.
- The West leads regional profit ($110.8K), followed by the East ($94.9K).
- Average profit is positive through 20% discount but becomes negative at 30% and above, with the steepest losses at 50% discount.
- Seasonality should be interpreted across the full 2023–2026 time series rather than from a single quarter.

## Project Structure

```
Task 3/
├── data/raw/task3_data.csv
├── data/processed/task3_prepared.csv
├── notebooks/task_analysis.ipynb
├── reports/figures/                 # 300-DPI charts
└── src/
    ├── data_cleaning.py
    ├── analysis.py
    ├── visualization.py
    └── run_analysis.py
```

## Run Locally

```powershell
pip install -r requirements.txt
python src\run_analysis.py
jupyter notebook notebooks\task_analysis.ipynb
```

## Google Colab

Upload `Task 3` to `MyDrive/Thiranex/Task 3`, open the notebook, and run all cells. The setup cell installs the dependencies, mounts Drive, and finds the task folder automatically.
