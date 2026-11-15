"""
models.py — per-strategy ensemble classifier.

Ensemble members (7-member diverse ensemble):
  - LightGBM  x3 (shallow, deep, regularised configs)
  - Logistic Regression L2 + L1 (scaled)
  - HistGradientBoostingClassifier (sklearn native, handles NaN)
  - ExtraTreesClassifier (random splits — most diverse tree member)
  - RandomForestClassifier (bagged trees)

Each member trained independently; probabilities averaged.
Target: P(next-month return > 0).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _make_members() -> list:
    """Instantiate fresh ensemble members (called each walk-forward step)."""
    return [
        # LightGBM — shallow, fast convergence
        lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            num_leaves=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ),
        # LightGBM — regularised, slower learning
        lgb.LGBMClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            num_leaves=10, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=7, verbose=-1,
        ),
        # LightGBM — wider trees, different seed
        lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            num_leaves=20, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=0.05, reg_lambda=0.5,
            random_state=99, verbose=-1,
        ),
        # Logistic Regression L2
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l2", C=5, solver="saga", max_iter=1000, random_state=42
            )),
        ]),
        # Logistic Regression L1 (sparse feature selection)
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l1", C=2, solver="saga", max_iter=1000, random_state=42
            )),
        ]),
        # HistGradientBoosting — handles NaN natively
        HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=20, random_state=42,
        ),
        # ExtraTrees — most diverse tree model (fully random splits)
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    ExtraTreesClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=10,
                max_features="sqrt", random_state=42, n_jobs=1,
            )),
        ]),
        # RandomForest — bagged trees, different from ExtraTrees
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=10,
                max_features="sqrt", random_state=42, n_jobs=1,
            )),
        ]),
    ]


def _make_regressors() -> list:
    """Regression ensemble to predict actual return magnitude."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
    return [
        Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Ridge(alpha=1.0)),
        ]),
        Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Lasso(alpha=0.01, max_iter=2000)),
        ]),
        lgb.LGBMRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            num_leaves=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
        ),
        Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    ExtraTreesRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=10,
                max_features="sqrt", random_state=42, n_jobs=1,
            )),
        ]),
    ]


def predict_proba_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    y_train_ret: np.ndarray | None = None,
    reg_weight: float = 0.35,
) -> float:
    """
    Combined classification + regression ensemble signal.

    Classification: P(return > 0) averaged across 8 members → prob ∈ [0,1]
    Regression    : predicted return magnitude averaged across 4 members

    Combined signal (same approach that worked in VIXY P8020LV):
      mu_class = prob - 0.5
      mu_reg   = clipped normalised regression prediction ∈ [-0.5, 0.5]
      mu       = (1 - reg_weight) * mu_class + reg_weight * mu_reg
    Final output = mu + 0.5  (back to [0,1] probability-like scale)

    Falls back to classification-only (0.5) if fewer than 2 members succeed.
    reg_weight=0  → pure classifier (same as before)
    """
    # ── classification ────────────────────────────────────────────────────────
    member_probs = []
    for member in _make_members():
        try:
            member.fit(X_train, y_train)
            p = member.predict_proba(X_pred)[0, 1]
            member_probs.append(p)
        except Exception:
            pass

    if len(member_probs) < 2:
        return 0.5

    prob     = float(np.mean(member_probs))
    mu_class = prob - 0.5

    # ── regression (if return targets available) ──────────────────────────────
    if y_train_ret is not None and reg_weight > 0:
        reg_preds = []
        for reg in _make_regressors():
            try:
                reg.fit(X_train, y_train_ret)
                pred = float(reg.predict(X_pred)[0])
                reg_preds.append(pred)
            except Exception:
                pass

        if len(reg_preds) >= 2:
            raw_pred = float(np.mean(reg_preds))
            # normalise by in-sample return std to get comparable scale
            ret_std  = float(np.std(y_train_ret)) + 1e-8
            mu_reg   = np.clip(raw_pred / ret_std * 0.5, -0.5, 0.5)
            mu_combined = (1 - reg_weight) * mu_class + reg_weight * mu_reg
            return float(np.clip(mu_combined + 0.5, 0.0, 1.0))

    return prob
