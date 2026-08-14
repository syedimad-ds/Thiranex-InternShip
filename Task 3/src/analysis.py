"""Hypothesis-driven business analyses for the Superstore dataset."""

from __future__ import annotations

import pandas as pd


def business_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Compute reproducible tables answering the five EDA hypotheses."""
    category = df.groupby("Category").agg(total_sales=("Sales", "sum"), total_profit=("Profit", "sum"))
    category["profit_margin"] = category["total_profit"] / category["total_sales"]

    monthly = df.groupby("Order Month", as_index=False).agg(total_sales=("Sales", "sum"), total_profit=("Profit", "sum"))
    quarterly = df.groupby("Order Quarter", sort=False).agg(total_sales=("Sales", "sum"), total_profit=("Profit", "sum"))

    order_sales = df.groupby(["Segment", "Order ID"], as_index=False)["Sales"].sum()
    segment = order_sales.groupby("Segment").agg(average_order_value=("Sales", "mean"), order_count=("Order ID", "nunique")).sort_values("average_order_value", ascending=False)

    discount = df.groupby("Discount").agg(average_profit=("Profit", "mean"), total_sales=("Sales", "sum"), line_items=("Profit", "size")).sort_index()
    region = df.groupby("Region").agg(total_sales=("Sales", "sum"), total_profit=("Profit", "sum"))
    region["profit_margin"] = region["total_profit"] / region["total_sales"]
    region = region.sort_values("total_profit", ascending=False)
    correlation = df[["Sales", "Quantity", "Discount", "Profit", "Shipping Days"]].corr(numeric_only=True)

    return {"category": category.sort_values("total_profit", ascending=False), "monthly": monthly, "quarterly": quarterly, "segment": segment, "discount": discount, "region": region, "correlation": correlation}
