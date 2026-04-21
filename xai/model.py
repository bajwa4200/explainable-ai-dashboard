"""Train a simple classifier and compute permutation importance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    accuracy: float
    feature_names: list[str]
    importances: list[tuple[str, float]]


def train_classifier(random_state: int = 42) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, list[str]]:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test, list(data.feature_names)


def build_report(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, feature_names: list[str]) -> TrainResult:
    accuracy = float(model.score(X_test, y_test))
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0)
    pairs = sorted(
        zip(feature_names, result.importances_mean, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return TrainResult(
        accuracy=accuracy,
        feature_names=feature_names,
        importances=[(name, float(val)) for name, val in pairs],
    )
