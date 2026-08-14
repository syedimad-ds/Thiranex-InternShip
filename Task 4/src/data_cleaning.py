"""Validation and feature engineering for transaction-risk modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "is_fraud"
ID_COLUMNS = ["transaction_id", "customer_id"]


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return core quality and class-imbalance statistics."""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum()[lambda values: values.gt(0)].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "time_range": (str(df["transaction_time"].min()), str(df["transaction_time"].max())),
    }


def prepare_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse time, validate target, and derive production-available features."""
    required = {TARGET, "transaction_time", "transaction_amount", "customer_id", "transaction_id"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = df.copy()
    prepared["transaction_time"] = pd.to_datetime(prepared["transaction_time"], errors="raise")
    if not set(prepared[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must be a binary 0/1 field.")
    if (prepared["transaction_amount"] < 0).any():
        raise ValueError("transaction_amount cannot be negative.")

    prepared["transaction_hour"] = prepared["transaction_time"].dt.hour
    prepared["transaction_day_of_week"] = prepared["transaction_time"].dt.dayofweek
    prepared["is_weekend"] = prepared["transaction_day_of_week"].ge(5).astype(int)
    prepared["log_transaction_amount"] = np.log1p(prepared["transaction_amount"])
    prepared["log_distance_from_home"] = np.log1p(prepared["distance_from_home"])
    prepared["log_previous_transaction_gap"] = np.log1p(prepared["previous_transaction_gap"])

    if prepared.isna().any().any():
        raise ValueError("Unexpected null values after preparation.")
    return prepared, {"derived_features": ["transaction_hour", "transaction_day_of_week", "is_weekend", "log_transaction_amount", "log_distance_from_home", "log_previous_transaction_gap"]}


def run_preparation_pipeline(raw_path: str | Path, processed_path: str | Path):
    """Load, audit, prepare, persist, and return transaction data."""
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_path)
    audit = audit_dataset(raw)
    prepared, log = prepare_transactions(raw)
    prepared.to_csv(processed_path, index=False)
    return raw, prepared, audit, log
