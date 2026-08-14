"""Leak-resistant imbalanced fraud modeling and evaluation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data_cleaning import ID_COLUMNS, TARGET

CATEGORICAL = ["merchant_category", "transaction_type", "payment_method", "city", "country", "device_type", "operating_system", "browser", "card_type", "merchant_risk_level", "transaction_status"]
EXCLUDED = ID_COLUMNS + [TARGET, "transaction_time", "transaction_amount", "distance_from_home", "previous_transaction_gap"]


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Use newest transactions as holdout data to approximate deployment."""
    ordered = df.sort_values("transaction_time").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train, test = ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()
    features = [column for column in df.columns if column not in EXCLUDED]
    return train[features], test[features], train[TARGET].astype(int), test[TARGET].astype(int), test["transaction_time"].min()


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create transformations learned only from training data."""
    categorical = [column for column in CATEGORICAL if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def make_hgb_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Use compact ordinal codes for the histogram boosting estimator."""
    categorical = [column for column in CATEGORICAL if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical),
    ], sparse_threshold=0)


def make_models(X: pd.DataFrame, positive_weight: float) -> dict[str, Pipeline]:
    """Return linear baseline and nonlinear gradient-boosted pipelines."""
    return {
        "Logistic Regression": Pipeline([("preprocessor", make_preprocessor(X)), ("classifier", LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs"))]),
        "Histogram Gradient Boosting": Pipeline([("preprocessor", make_hgb_preprocessor(X)), ("classifier", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0, random_state=42))]),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5):
    """Evaluate with metrics appropriate for rare-event fraud screening."""
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_test, probability),
        "PR-AUC": average_precision_score(y_test, probability),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
    }, prediction, probability


def fit_comparison(X_train, X_test, y_train, y_test):
    """Fit baseline and ensemble models, balancing only the training stage."""
    positive_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    rows, models = [], {}
    for name, model in make_models(X_train, positive_weight).items():
        if name == "Histogram Gradient Boosting":
            weights = np.where(y_train.eq(1), positive_weight, 1.0)
            model.fit(X_train, y_train, classifier__sample_weight=weights)
        else:
            if len(X_train) > 100_000:
                baseline_data, _ = train_test_split(
                    pd.concat([X_train, y_train.rename(TARGET)], axis=1),
                    train_size=100_000,
                    stratify=y_train,
                    random_state=42,
                )
                baseline_y = baseline_data.pop(TARGET)
                model.fit(baseline_data, baseline_y)
            else:
                model.fit(X_train, y_train)
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})
        models[name] = model
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True), models, positive_weight


def tune_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, positive_weight: float):
    """Tune HGB on a stratified training sample, preserving the time holdout."""
    sample_size = min(40_000, len(X_train))
    combined = pd.concat([X_train, y_train.rename(TARGET)], axis=1)
    if sample_size < len(combined):
        sampled, _ = train_test_split(combined, train_size=sample_size, stratify=combined[TARGET], random_state=42)
    else:
        sampled = combined
    y_sample = sampled.pop(TARGET).astype(int)
    pipeline = make_models(X_train, positive_weight)["Histogram Gradient Boosting"]
    search = RandomizedSearchCV(
        pipeline,
        {"classifier__learning_rate": [0.06, 0.08, 0.1], "classifier__max_leaf_nodes": [15, 24, 31], "classifier__l2_regularization": [0.0, 1.0, 2.0], "classifier__max_iter": [80, 120]},
        n_iter=2,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        n_jobs=1,
        random_state=42,
        refit=True,
    )
    weights = np.where(y_sample.eq(1), positive_weight, 1.0)
    search.fit(sampled, y_sample, classifier__sample_weight=weights)
    best = search.best_estimator_
    full_weights = np.where(y_train.eq(1), positive_weight, 1.0)
    best.fit(X_train, y_train, classifier__sample_weight=full_weights)
    return search, best


def permutation_feature_importance(model, X_test, y_test, sample_size: int = 1_000) -> pd.DataFrame:
    """Compute model-agnostic importance on a bounded temporal holdout sample."""
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    result = permutation_importance(model, sample, y_test.loc[sample.index], scoring="average_precision", n_repeats=1, random_state=42, n_jobs=1)
    return pd.DataFrame({"feature": sample.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Persist the full feature transformation and estimator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
