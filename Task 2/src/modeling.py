"""Leak-free training, tuning, evaluation, and persistence for Task 2."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "target"
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build transformations fitted solely within each training fold."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Create a reproducible stratified 80/20 holdout split."""
    X, y = df.drop(columns=TARGET), df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return baseline and tree-ensemble pipelines."""
    preprocessor = make_preprocessor(X)
    return {
        "Logistic Regression": Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2_000, random_state=42))]),
        "Random Forest": Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=42))]),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    """Calculate primary classification metrics on held-out observations."""
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probability),
        "PR-AUC": average_precision_score(y_test, probability),
    }
    return metrics, predicted, probability


def train_and_compare(df: pd.DataFrame):
    """Fit comparison models without contaminating the final test set."""
    X_train, X_test, y_train, y_test = split_data(df)
    models = build_models(X_train)
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})
        fitted[name] = model
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True), fitted, X_train, X_test, y_train, y_test


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune an ensemble using stratified CV on training data only."""
    pipeline = build_models(X_train)["Random Forest"]
    search = GridSearchCV(
        pipeline,
        {"classifier__n_estimators": [300, 500], "classifier__max_depth": [None, 6, 10], "classifier__min_samples_leaf": [1, 2, 4], "classifier__max_features": ["sqrt", 0.7]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X_train, y_train)


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return sorted transformed feature importances for a tree-based model."""
    transformer = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise TypeError("Feature importance requires a tree-based classifier.")
    return pd.DataFrame({"feature": transformer.get_feature_names_out(), "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Persist a fitted full pipeline, including preprocessing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
