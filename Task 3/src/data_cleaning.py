"""Data preparation and audit routines for Superstore EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Order ID", "Order Date", "Ship Date", "Segment", "Region", "Category",
    "Sub-Category", "Sales", "Quantity", "Discount", "Profit",
}


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return dataset quality measures before preparation."""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum()[lambda values: values.gt(0)].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def prepare_superstore_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse dates, validate fields, and add documented analysis features."""
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = df.copy()
    for column in ["Order Date", "Ship Date"]:
        prepared[column] = pd.to_datetime(prepared[column], format="%m/%d/%Y", errors="raise")
    for column in ["Sales", "Quantity", "Discount", "Profit"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")

    prepared["Order Year"] = prepared["Order Date"].dt.year
    prepared["Order Month"] = prepared["Order Date"].dt.to_period("M").astype(str)
    prepared["Order Quarter"] = "Q" + prepared["Order Date"].dt.quarter.astype(str)
    prepared["Shipping Days"] = (prepared["Ship Date"] - prepared["Order Date"]).dt.days
    prepared["Profit Margin"] = prepared["Profit"].div(prepared["Sales"]).where(prepared["Sales"].ne(0))

    if prepared[["Sales", "Quantity", "Discount", "Profit"]].isna().any().any():
        raise ValueError("Unexpected missing numeric values after preparation.")
    if (prepared["Shipping Days"] < 0).any():
        raise ValueError("Ship Date cannot precede Order Date.")
    return prepared, {"new_features": ["Order Year", "Order Month", "Order Quarter", "Shipping Days", "Profit Margin"]}


def run_preparation_pipeline(raw_path: str | Path, processed_path: str | Path):
    """Read, audit, prepare, save, and return the Superstore dataset."""
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(raw_path)
    audit = audit_dataset(raw_df)
    prepared, log = prepare_superstore_data(raw_df)
    prepared.to_csv(processed_path, index=False)
    return raw_df, prepared, audit, log
