"""Command-line entry point for the complete Task 2 modeling workflow."""

from __future__ import annotations

from pathlib import Path

from data_cleaning import run_cleaning_pipeline
from modeling import evaluate_model, feature_importance, save_model, train_and_compare, tune_random_forest
from visualization import create_diagnostics, plot_feature_importance


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/raw/heart.csv"
    cleaned_path = root / "data/processed/task2_cleaned.csv"
    figure_dir = root / "reports/figures"
    model_path = root / "models/tuned_random_forest.joblib"

    _, cleaned_df, audit, cleaning_log = run_cleaning_pipeline(raw_path, cleaned_path)
    comparison, _, X_train, X_test, y_train, y_test = train_and_compare(cleaned_df)
    search = tune_random_forest(X_train, y_train)
    metrics, _, _ = evaluate_model(search.best_estimator_, X_test, y_test)
    importance = feature_importance(search.best_estimator_)
    paths = create_diagnostics(search.best_estimator_, X_test, y_test, figure_dir)
    paths.append(plot_feature_importance(importance, figure_dir))
    save_model(search.best_estimator_, model_path)

    print(f"Raw audit: shape={audit['shape']}, duplicate rows={audit['duplicate_rows']}")
    print(f"Cleaning: {cleaning_log}")
    print("\nBaseline comparison (held-out test set):")
    print(comparison.round(3).to_string(index=False))
    print(f"\nBest CV ROC-AUC: {search.best_score_:.3f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Tuned holdout metrics: { {key: round(value, 3) for key, value in metrics.items()} }")
    print(f"Model saved to: {model_path}")
    print("Figures:\n" + "\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
