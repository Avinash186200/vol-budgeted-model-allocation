"""Small, fixed ensemble that predicts P(next-month return > 0) for one model."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _members(seed: int) -> list[Pipeline]:
    """The two fixed ensemble members (no tuning, no member search)."""
    return [
        Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=500)),
        ]),
        Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=50, max_depth=2, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=seed)),
        ]),
    ]


def predict_prob(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, seed: int = 42) -> float:
    """Average the members' P(y=1) for a single prediction row.

    Returns 0.5 (no view) if the training labels have one class only, or if
    every member fails to fit.
    """
    if len(np.unique(y_train)) < 2:
        return 0.5
    probs = []
    for m in _members(seed):
        try:
            m.fit(X_train, y_train)
            probs.append(float(m.predict_proba(X_pred)[0, 1]))
        except ValueError:
            continue
    return float(np.mean(probs)) if probs else 0.5
