"""
backtest.py — monthly walk-forward backtest engine.

For each OOS month T:
  1. Estimate factor covariance from trailing COV_WINDOW months
  2. Train ensemble per active model on expanding window (strict: data up to T-1)
  3. Predict P(positive return in T+1) using features at T
  4. Run vol-budgeted QP optimizer with shrinkage
  5. Record realized return for T+1 vs equal-weight and vol-parity benchmarks

No look-ahead by construction:
  - Features at month T use shift(1) per-strategy data (T-1 data)
  - Training target at T-1 = return in T (known by end of T)
  - Prediction at T → realize in T+1
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from config import OOS_MONTHS, MIN_TRAIN_MONTHS, COV_WINDOW
from features import (build_feature_matrix, get_spy_daily,
                      build_per_strategy_features, build_cs_features)
from models import predict_proba_ensemble
from optimizer import factor_covariance, optimize_cpo, _vol_parity


def run_backtest(
    monthly_ret: pd.DataFrame,
    macro_feats: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full walk-forward backtest.
    Returns a DataFrame with one row per OOS month containing:
      date, ret_cpo, ret_cpo_shrunk, ret_ew, ret_vp,
      pred_pv_cpo, pred_pv_shrunk, n_active, avg_prob
    """
    all_months = monthly_ret.index
    oos_months = all_months[-OOS_MONTHS:]
    model_cols = list(monthly_ret.columns)
    n_models   = len(model_cols)

    # SPY monthly returns for factor covariance model
    try:
        spy_daily   = get_spy_daily()
        spy_monthly = spy_daily.pct_change().resample("ME").last()
    except Exception:
        spy_monthly = pd.Series(dtype=float)

    # Pre-compute per-strategy and cross-sectional features once for the
    # full return history — avoids recomputing for every strategy × OOS month.
    if verbose:
        print("  Pre-computing cross-sectional features ...")
    all_strat_feats = build_per_strategy_features(monthly_ret)
    cs_feats        = build_cs_features(all_strat_feats)

    records = []

    # iterate OOS months except the last (no target available)
    for t_idx, t_date in enumerate(oos_months[:-1]):
        t_pos         = list(all_months).index(t_date)
        ret_next_date = all_months[t_pos + 1]

        # ── covariance window ─────────────────────────────────────────────────
        cov_slice  = monthly_ret.iloc[max(0, t_pos - COV_WINDOW): t_pos]
        spy_slice  = spy_monthly.reindex(cov_slice.index)
        cov_ann    = factor_covariance(cov_slice, spy_slice)
        sigma_ann  = np.sqrt(np.maximum(np.diag(cov_ann), 0))

        # ── active model flag ─────────────────────────────────────────────────
        # model is active if it has traded in >= 12 of last COV_WINDOW months
        active = np.array([
            monthly_ret[c].iloc[max(0, t_pos - COV_WINDOW): t_pos]
                         .notna().sum() >= 12
            for c in model_cols
        ])

        # ── per-model ensemble predictions ────────────────────────────────────
        probs = np.full(n_models, 0.5)

        for m_idx, col in enumerate(model_cols):
            if not active[m_idx]:
                continue

            feat_df = build_feature_matrix(monthly_ret, macro_feats, col,
                                           all_strat_feats=all_strat_feats,
                                           cs_feats=cs_feats)

            # strict train: rows 0..t_pos-1 (target at t_pos-1 = return at t_pos, known)
            train_df = feat_df.iloc[:t_pos].dropna(subset=["target"])
            if len(train_df) < MIN_TRAIN_MONTHS:
                continue

            X_train  = train_df.drop(columns=["target"]).values
            y_train  = train_df["target"].values
            # regression target: actual next-month returns (same rows as train)
            y_train_ret = monthly_ret[col].shift(-1).reindex(
                train_df.index).fillna(0).values

            # prediction at row t_pos: features use t_pos-1 data (shift(1)) → no look-ahead
            if t_pos >= len(feat_df):
                continue
            X_pred = feat_df.drop(columns=["target"]).iloc[[t_pos]].values

            probs[m_idx] = predict_proba_ensemble(
                X_train, y_train, X_pred, y_train_ret=y_train_ret
            )

        # ── expected return signal (centred) ──────────────────────────────────
        mu = probs - 0.5

        # ── dynamic shrinkage alpha ───────────────────────────────────────────
        # confidence = average |prob - 0.5| across active models (0=no signal)
        # alpha scales from 0.50 (low confidence) to 0.90 (high confidence)
        # so the portfolio stays near EW when predictions are uncertain.
        if active.sum() > 0:
            conf  = float(np.abs(probs[active] - 0.5).mean())   # 0 to 0.5
            alpha = float(np.clip(0.50 + 2.0 * conf, 0.50, 0.90))
        else:
            alpha = 0.50

        # ── optimize ──────────────────────────────────────────────────────────
        w_cpo_raw, w_cpo_shrunk, w_ew = optimize_cpo(
            mu, cov_ann, sigma_ann, active, model_cols, shrink_alpha=alpha
        )
        w_vp = _vol_parity(active, cov_ann, sigma_ann)

        # ── realized returns ──────────────────────────────────────────────────
        realized = monthly_ret.loc[ret_next_date].fillna(0).values
        ret_cpo      = float(w_cpo_raw   @ realized)
        ret_shrunk   = float(w_cpo_shrunk @ realized)
        ret_ew       = float(w_ew         @ realized)
        ret_vp       = float(w_vp         @ realized)

        pv_cpo    = np.sqrt(max(float(w_cpo_raw    @ cov_ann @ w_cpo_raw),    0))
        pv_shrunk = np.sqrt(max(float(w_cpo_shrunk @ cov_ann @ w_cpo_shrunk), 0))

        records.append({
            "date":          ret_next_date,
            "ret_cpo":       ret_cpo,
            "ret_cpo_shrunk":ret_shrunk,
            "ret_ew":        ret_ew,
            "ret_vp":        ret_vp,
            "pred_pv_cpo":   pv_cpo,
            "pred_pv_shrunk":pv_shrunk,
            "n_active":      int(active.sum()),
            "avg_prob":      float(probs[active].mean()) if active.sum() > 0 else 0.5,
            "weights_cpo":   w_cpo_shrunk.tolist(),
        })

        if verbose and (t_idx + 1) % 6 == 0:
            print(f"  {t_date.date()}  CPO={ret_cpo:+.3f}  "
                  f"Shrunk={ret_shrunk:+.3f}  EW={ret_ew:+.3f}  "
                  f"active={int(active.sum())}  avg_prob={probs[active].mean():.3f}")

    return pd.DataFrame(records).set_index("date")
