"""Custom plotting routines for Task 1 data quality and EDA visuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DPI = 300
DEFAULT_FIGSIZE = (10, 6)


def _ensure_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_missing_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Missing Data Heatmap",
) -> Path:
    """Plot a binary heatmap of missing values across columns."""
    missing_matrix = df.isna()
    if not missing_matrix.any().any():
        for col in df.select_dtypes(include=["object", "string"]).columns:
            missing_matrix[col] = df[col].astype(str).str.strip().eq("")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        missing_matrix,
        cbar=True,
        yticklabels=False,
        cmap="viridis",
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Features")
    ax.set_ylabel("Observations")
    return _save_figure(fig, output_path)


def plot_distribution_comparison(
    before: pd.Series,
    after: pd.Series,
    column: str,
    output_path: str | Path,
) -> Path:
    """Compare numeric distributions before and after cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    before_numeric = pd.to_numeric(before.astype(str).str.strip(), errors="coerce")

    sns.histplot(before_numeric.dropna(), kde=True, ax=axes[0], color="#E76F51")
    axes[0].set_title(f"{column} — Before Cleaning")
    axes[0].set_xlabel(column)

    sns.histplot(after, kde=True, ax=axes[1], color="#2A9D8F")
    axes[1].set_title(f"{column} — After Cleaning")
    axes[1].set_xlabel(column)

    fig.suptitle(f"Distribution Shift: {column}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_univariate_distributions(
    df: pd.DataFrame,
    numeric_columns: list[str],
    output_path: str | Path,
) -> Path:
    """Plot histogram/KDE panels for numeric features."""
    n_cols = len(numeric_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_columns):
        sns.histplot(df[col], kde=True, ax=ax, color="#457B9D")
        ax.set_title(col)
        ax.set_xlabel("")

    fig.suptitle("Univariate Numeric Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_bivariate_boxplots(
    df: pd.DataFrame,
    numeric_col: str,
    categorical_col: str,
    output_path: str | Path,
) -> Path:
    """Plot a count/box plot for numeric vs categorical relationship."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = df[categorical_col].value_counts().index
    sns.countplot(
        data=df, x=categorical_col, order=order, ax=axes[0],
        hue=categorical_col, legend=False, palette="Set2",
    )
    axes[0].set_title(f"Count by {categorical_col}")
    axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(
        data=df,
        x=categorical_col,
        y=numeric_col,
        order=order,
        ax=axes[1],
        hue=categorical_col,
        legend=False,
        palette="Set3",
    )
    axes[1].set_title(f"{numeric_col} by {categorical_col}")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle(
        f"Bivariate Analysis: {numeric_col} vs {categorical_col}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    annotate: bool = True,
) -> Path:
    """Plot an annotated correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=annotate,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def generate_all_task1_figures(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Generate the full Task 1 figure set."""
    figures_dir = _ensure_dir(figures_dir)
    saved: list[Path] = []

    saved.append(
        plot_missing_heatmap(
            raw_df,
            figures_dir / "01_missing_heatmap_before.png",
            title="Missing Data — Before Cleaning",
        )
    )
    saved.append(
        plot_missing_heatmap(
            cleaned_df,
            figures_dir / "02_missing_heatmap_after.png",
            title="Missing Data — After Cleaning",
        )
    )
    saved.append(
        plot_distribution_comparison(
            raw_df["TotalCharges"],
            cleaned_df["TotalCharges"],
            "TotalCharges",
            figures_dir / "03_totalcharges_distribution_shift.png",
        )
    )

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    saved.append(
        plot_univariate_distributions(
            cleaned_df,
            numeric_cols,
            figures_dir / "04_univariate_numeric_distributions.png",
        )
    )
    saved.append(
        plot_bivariate_boxplots(
            cleaned_df,
            numeric_col="MonthlyCharges",
            categorical_col="Contract",
            output_path=figures_dir / "05_monthlycharges_by_contract.png",
        )
    )
    saved.append(
        plot_bivariate_boxplots(
            cleaned_df,
            numeric_col="tenure",
            categorical_col="Churn",
            output_path=figures_dir / "06_tenure_by_churn.png",
        )
    )
    saved.append(
        plot_correlation_heatmap(
            cleaned_df,
            figures_dir / "07_correlation_heatmap.png",
        )
    )

    return saved
