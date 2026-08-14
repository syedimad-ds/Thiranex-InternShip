"""Publication-ready diagnostic charts for Task 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def create_diagnostics(model, X_test, y_test, output_dir: str | Path) -> list[Path]:
    """Export confusion matrix, ROC, PR, and feature-importance figures."""
    output_dir = Path(output_dir)
    paths = []
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax)
    ax.set_title("Tuned Random Forest: Confusion Matrix")
    paths.append(_save(fig, output_dir / "01_confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Tuned Random Forest")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC Curve")
    paths.append(_save(fig, output_dir / "02_roc_curve.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Tuned Random Forest")
    ax.set_title("Precision-Recall Curve")
    paths.append(_save(fig, output_dir / "03_precision_recall_curve.png"))
    return paths


def plot_feature_importance(importance: pd.DataFrame, output_dir: str | Path, top_n: int = 12) -> Path:
    """Export a horizontal bar chart of the leading predictors."""
    top = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#176B87")
    ax.set_title("Top Feature Importances: Tuned Random Forest")
    ax.set_xlabel("Impurity-based importance")
    return _save(fig, Path(output_dir) / "04_feature_importance.png")
