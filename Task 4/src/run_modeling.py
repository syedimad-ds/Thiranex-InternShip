"""Command-line runner for the Task 4 FinTech fraud project."""

from pathlib import Path

from data_cleaning import run_preparation_pipeline
from modeling import evaluate_model, fit_comparison, permutation_feature_importance, save_model, temporal_split, tune_gradient_boosting
from visualization import generate_diagnostics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _, data, audit, log = run_preparation_pipeline(root / "data/raw/credit_card_fraud_detection_500k.csv", root / "data/processed/task4_transactions_prepared.csv")
    X_train, X_test, y_train, y_test, holdout_start = temporal_split(data)
    comparison, _, positive_weight = fit_comparison(X_train, X_test, y_train, y_test)
    search, best_model = tune_gradient_boosting(X_train, y_train, positive_weight)
    metrics, _, _ = evaluate_model(best_model, X_test, y_test)
    importance = permutation_feature_importance(best_model, X_test, y_test)
    figure_paths = generate_diagnostics(best_model, X_test, y_test, importance, root / "reports/figures")
    model_path = save_model(best_model, root / "models/tuned_fraud_detector.joblib")

    print(f"Audit: shape={audit['shape']}, nulls={audit['missing_values']}, duplicates={audit['duplicate_rows']}, fraud_rate={audit['fraud_rate']:.2%}")
    print(f"Preparation: {log}")
    print(f"Temporal holdout begins: {holdout_start}; train fraud rate={y_train.mean():.2%}; test fraud rate={y_test.mean():.2%}")
    print("\nBaseline comparison:\n", comparison.round(3).to_string(index=False))
    print(f"\nBest tuning CV PR-AUC: {search.best_score_:.3f}; parameters: {search.best_params_}")
    print("Tuned holdout metrics:", {key: round(value, 3) for key, value in metrics.items()})
    print("\nTop importance:\n", importance.head(10).round(4).to_string(index=False))
    print(f"Model: {model_path}\nFigures:\n" + "\n".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
