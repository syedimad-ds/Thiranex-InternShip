"""High-resolution fraud-model diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_diagnostics(model, X_test, y_test, importance: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Export class, threshold, ranking, and explanatory diagnostics."""
    output_dir = Path(output_dir)
    sns.set_theme(style="whitegrid", palette="deep")
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_test.value_counts().rename(index={0: "Legitimate", 1: "Fraud"})
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    ax.set(title="Temporal Holdout Class Distribution", ylabel="Transactions")
    paths.append(_save(fig, output_dir / "01_holdout_class_distribution.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Legitimate", "Fraud"], cmap="Blues", ax=ax)
    ax.set_title("Fraud Screening Confusion Matrix (0.50 threshold)")
    paths.append(_save(fig, output_dir / "02_confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, name="Tuned HGB", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC Curve: Future Transaction Holdout")
    paths.append(_save(fig, output_dir / "03_roc_curve.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name="Tuned HGB", ax=ax)
    ax.axhline(y_test.mean(), color="gray", linestyle="--", label="Fraud prevalence")
    ax.legend()
    ax.set_title("Precision-Recall Curve")
    paths.append(_save(fig, output_dir / "04_precision_recall_curve.png"))

    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#176B87")
    ax.set(title="Permutation Importance (PR-AUC decrease)", xlabel="Mean importance")
    paths.append(_save(fig, output_dir / "05_permutation_importance.png"))
    return paths
