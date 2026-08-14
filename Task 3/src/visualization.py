"""High-resolution charts for hypothesis-driven Superstore EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def generate_figures(df, tables, output_dir: str | Path) -> list[Path]:
    """Generate and save all Task 3 figures at 300 DPI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    data = tables["category"].reset_index()
    sns.barplot(data=data, x="Category", y="profit_margin", hue="Category", legend=False, ax=ax)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set(title="Profit Margin by Category", ylabel="Profit margin")
    paths.append(_save(fig, output_dir / "01_category_profit_margin.png"))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=tables["monthly"], x="Order Month", y="total_sales", marker="o", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="Monthly Sales Trend", xlabel="Order month", ylabel="Sales ($)")
    paths.append(_save(fig, output_dir / "02_monthly_sales_trend.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Segment", y="Sales", hue="Segment", legend=False, showfliers=False, ax=ax)
    ax.set(title="Line-Item Sales Distribution by Segment", ylabel="Sales ($)")
    paths.append(_save(fig, output_dir / "03_segment_sales_distribution.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    data = tables["discount"].reset_index()
    sns.lineplot(data=data, x="Discount", y="average_profit", marker="o", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Average Profit by Discount Level", ylabel="Average profit ($)")
    paths.append(_save(fig, output_dir / "04_discount_profit.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    data = tables["region"].reset_index()
    sns.barplot(data=data, x="Region", y="total_profit", hue="Region", legend=False, ax=ax)
    ax.set(title="Total Profit by Region", ylabel="Profit ($)")
    paths.append(_save(fig, output_dir / "05_regional_profit.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(tables["correlation"], annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    paths.append(_save(fig, output_dir / "06_correlation_heatmap.png"))
    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
