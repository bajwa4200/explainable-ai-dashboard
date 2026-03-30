"""Dataset helpers (Iris via sklearn)."""

from __future__ import annotations

from sklearn.datasets import load_iris


def load_iris_data():
    return load_iris()
