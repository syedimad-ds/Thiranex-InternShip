"""Data quality routines for the Task 2 heart-disease dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return a compact, serializable data-quality audit."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_balance": df["target"].value_counts(normalize=True).sort_index().to_dict(),
    }


def clean_heart_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate clinical records and validate the target."""
    required = {
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target",
    }
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    cleaned = df.copy()
    duplicates_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if cleaned.isna().any().any():
        raise ValueError("Unexpected missing values found in the heart dataset.")
    if not set(cleaned["target"].unique()).issubset({0, 1}):
        raise ValueError("The target must be binary, encoded as 0 and 1.")

    return cleaned, {"duplicates_removed": duplicates_removed, "rows_after_cleaning": len(cleaned)}


def run_cleaning_pipeline(raw_path: str | Path, processed_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load, audit, deduplicate, persist, and return Task 2 data."""
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(raw_path)
    audit_before = audit_dataset(raw_df)
    cleaned_df, log = clean_heart_data(raw_df)
    cleaned_df.to_csv(processed_path, index=False)
    return raw_df, cleaned_df, audit_before, log


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    _, cleaned, audit, log = run_cleaning_pipeline(
        root / "data/raw/heart.csv", root / "data/processed/task2_cleaned.csv"
    )
    print(f"Raw shape: {audit['shape']}; duplicates: {audit['duplicate_rows']}")
    print(f"Cleaning log: {log}")
    print(f"Cleaned shape: {cleaned.shape}; nulls: {cleaned.isna().sum().sum()}")
