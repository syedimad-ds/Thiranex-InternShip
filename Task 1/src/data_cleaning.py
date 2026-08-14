"""Reusable data auditing, preprocessing, and cleaning routines for Task 1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

MissingnessMechanism = Literal["MCAR", "MAR", "MNAR", "NONE"]


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return a structured audit summary for exploratory review."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    missing = df.isna().sum()
    hidden_missing = {}
    for col in object_cols:
        blank_count = df[col].astype(str).str.strip().eq("").sum()
        if blank_count:
            hidden_missing[col] = int(blank_count)

    duplicate_count = int(df.duplicated().sum())

    type_issues = []
    if "TotalCharges" in df.columns and not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        invalid = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").isna().sum()
        type_issues.append(
            {
                "column": "TotalCharges",
                "issue": "stored_as_string",
                "invalid_or_blank_values": int(invalid),
            }
        )

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": missing[missing > 0].to_dict(),
        "hidden_missing_in_strings": hidden_missing,
        "duplicate_rows": duplicate_count,
        "numeric_columns": numeric_cols,
        "categorical_columns": object_cols,
        "type_issues": type_issues,
        "summary_statistics": df.describe(include="all").T.to_dict(),
    }


def classify_missingness(df: pd.DataFrame, column: str) -> MissingnessMechanism:
    """
    Heuristic classification of missingness for a single column.

    MNAR: blanks correlate strongly with another observed field (e.g., tenure=0).
    MAR: missingness varies across groups but not deterministically.
    MCAR: missingness appears random across groups.
    """
    if column not in df.columns:
        return "NONE"

    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        missing_mask = series.isna()
    else:
        missing_mask = series.isna() | series.astype(str).str.strip().isin(["", "nan"])

    if missing_mask.sum() == 0:
        return "NONE"

    if "tenure" in df.columns:
        tenure_zero_rate = df.loc[missing_mask, "tenure"].eq(0).mean()
        overall_zero_rate = df["tenure"].eq(0).mean()
        if tenure_zero_rate > 0.9 and tenure_zero_rate > overall_zero_rate * 5:
            return "MNAR"

    if "InternetService" in df.columns:
        grouped_rate = df.groupby("InternetService")[column].apply(
            lambda s: s.isna().mean() if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.strip().eq("").mean()
        )
        if grouped_rate.max() - grouped_rate.min() > 0.05:
            return "MAR"

    return "MCAR"


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned.loc[cleaned[col].eq(""), col] = np.nan
    return cleaned


def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned


def _impute_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute TotalCharges for new customers (tenure=0) using MonthlyCharges.

    Rationale: MNAR missingness — charges are blank because billing history
    has not accumulated yet, not because values are unknown.
    """
    cleaned = df.copy()
    missing_mask = cleaned["TotalCharges"].isna()
    new_customer_mask = missing_mask & cleaned["tenure"].eq(0)

    cleaned.loc[new_customer_mask, "TotalCharges"] = cleaned.loc[
        new_customer_mask, "MonthlyCharges"
    ]

    remaining_missing = cleaned["TotalCharges"].isna()
    if remaining_missing.any():
        median_charge = cleaned["TotalCharges"].median()
        cleaned.loc[remaining_missing, "TotalCharges"] = median_charge

    return cleaned


def _impute_numeric_knn(
    df: pd.DataFrame,
    columns: list[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    cleaned = df.copy()
    if not cleaned[columns].isna().any().any():
        return cleaned

    imputer = KNNImputer(n_neighbors=n_neighbors)
    cleaned[columns] = imputer.fit_transform(cleaned[columns])
    return cleaned


def _impute_categorical_mode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if cleaned[col].isna().any():
            mode_value = cleaned[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)
    return cleaned


def _clip_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = df.copy()
    outlier_flag = pd.Series(False, index=cleaned.index, name="_outlier_flag")

    for col in columns:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        col_outliers = (cleaned[col] < lower) | (cleaned[col] > upper)
        outlier_flag |= col_outliers
        cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

    return cleaned, outlier_flag


def clean_dataset(
    df: pd.DataFrame,
    outlier_method: Literal["iqr", "zscore"] = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full cleaning pipeline and return cleaned data plus a log.
    """
    log: dict = {"steps": [], "missingness": {}}

    cleaned = _strip_string_columns(df)
    log["steps"].append("stripped_whitespace_and_blank_strings_to_nan")

    if "TotalCharges" in cleaned.columns:
        mechanism = classify_missingness(cleaned, "TotalCharges")
        log["missingness"]["TotalCharges"] = mechanism
        cleaned = _fix_total_charges(cleaned)
        cleaned = _impute_total_charges(cleaned)
        log["steps"].append(f"converted_total_charges_numeric_imputed_{mechanism.lower()}")

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "SeniorCitizen"]

    categorical_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "customerID"]

    if cleaned[categorical_cols].isna().any().any():
        cleaned = _impute_categorical_mode(cleaned, categorical_cols)
        log["steps"].append("imputed_categorical_with_mode")

    if cleaned[numeric_cols].isna().any().any():
        cleaned = _impute_numeric_knn(cleaned, numeric_cols)
        log["steps"].append("imputed_numeric_with_knn")

    outlier_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in cleaned.columns]
    if outlier_method == "iqr":
        cleaned, outlier_flag = _clip_outliers_iqr(cleaned, outlier_cols, multiplier=iqr_multiplier)
        log["steps"].append(f"clipped_outliers_iqr_{iqr_multiplier}")
    else:
        outlier_flag = pd.Series(False, index=cleaned.index, name="_outlier_flag")
        for col in outlier_cols:
            mean = cleaned[col].mean()
            std = cleaned[col].std()
            if std == 0:
                continue
            z_scores = (cleaned[col] - mean) / std
            col_outliers = z_scores.abs() > zscore_threshold
            outlier_flag |= col_outliers
            cleaned.loc[z_scores > zscore_threshold, col] = mean + zscore_threshold * std
            cleaned.loc[z_scores < -zscore_threshold, col] = mean - zscore_threshold * std
        log["steps"].append(f"clipped_outliers_zscore_{zscore_threshold}")

    cleaned["_outlier_flag"] = outlier_flag
    log["outlier_rows_clipped"] = int(outlier_flag.sum())

    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)
    cleaned["tenure"] = cleaned["tenure"].astype(int)

    assert cleaned.isna().sum().sum() == 0, "Unhandled null values remain after cleaning."
    return cleaned, log


def run_cleaning_pipeline(
    raw_path: str | Path,
    processed_path: str | Path,
    outlier_method: Literal["iqr", "zscore"] = "iqr",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load raw data, audit, clean, save, and return both frames plus logs.
    """
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    audit_before = audit_dataset(raw_df)
    cleaned_df, clean_log = clean_dataset(raw_df, outlier_method=outlier_method)
    audit_after = audit_dataset(cleaned_df)

    cleaned_df.to_csv(processed_path, index=False)

    return raw_df, cleaned_df, audit_before, {"cleaning": clean_log, "audit_after": audit_after}


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw = project_root / "data" / "raw" / "task1_data.csv"
    processed = project_root / "data" / "processed" / "task1_cleaned.csv"

    raw_df, cleaned_df, audit_before, pipeline_log = run_cleaning_pipeline(raw, processed)

    print("=== RAW AUDIT ===")
    print(f"Shape: {audit_before['shape']}")
    print(f"Duplicates: {audit_before['duplicate_rows']}")
    print(f"Hidden missing: {audit_before['hidden_missing_in_strings']}")
    print(f"Type issues: {audit_before['type_issues']}")

    print("\n=== CLEANING LOG ===")
    print(pipeline_log["cleaning"])

    print("\n=== CLEANED SUMMARY ===")
    print(cleaned_df.info())
    print("\nMissing values:", cleaned_df.isna().sum().sum())
    print(f"Saved to: {processed}")
