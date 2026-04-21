"""Permutation importance wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def compute_importances(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 5,
    random_state: int = 0,
) -> list[tuple[str, float]]:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state
    )
    pairs = sorted(
        zip(feature_names, result.importances_mean, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(name, float(val)) for name, val in pairs]
