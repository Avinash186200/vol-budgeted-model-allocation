"""Monthly walk-forward backtest.

At each rebalance month-end t:
  1. Covariance from the ``cov_window`` months ending at t (returns known at t).
  2. For each active model, fit on rows dated <= t-1 (their target is month t,
     already realised) and predict the row dated t (target month t+1).
  3. Optimise, blend with equal weight, charge costs on turnover.
  4. Book the realised return of month t+1.
All selection is by **date**, never by row position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import align_macro, build_panels, feature_matrix
from .model import predict_prob
from .optimizer import Budget, default_budget, equal_weight, factor_covariance, optimize, vol_parity


@dataclass
class Config:
    """Backtest settings. All are fixed in advance; none is tuned on the test window."""

    oos_months: int = 24
    min_train: int = 36
    cov_window: int = 36
    min_live: int = 12
    macro_lag: int = 1
    shrink_alpha: float = 0.8
    cost_bps: float = 5.0
    seed: int = 42


def run_backtest(
    monthly: pd.DataFrame,
    macro: pd.DataFrame | None,
    market: pd.Series | None,
    cfg: Config = Config(),
    budget: Budget | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward.

    Returns:
        (returns, weights, predictions). ``returns`` has one row per realised
        month with net returns for opt_raw, opt_blend, equal_weight,
        vol_parity plus predicted vol; ``predictions`` holds each model's
        P(up) and the realised next-month return, for IC/AUC checks.
    """
    cols = list(monthly.columns)
    budget = budget or default_budget(cols)
    months = monthly.index
    panels = build_panels(monthly)
    mac = align_macro(macro, months, cfg.macro_lag) if macro is not None else pd.DataFrame(index=months)
    fms = {c: feature_matrix(monthly, panels, mac, c) for c in cols}

    rebal = months[-(cfg.oos_months + 1):-1]
    prev = {k: np.zeros(len(cols)) for k in ("opt_raw", "opt_blend", "equal_weight", "vol_parity")}
    rows, wrows, prows = [], [], []

    for t in rebal:
        t_pos = months.get_loc(t)
        nxt = months[t_pos + 1]
        win = monthly.loc[:t].iloc[-cfg.cov_window:]
        active = (win.notna().sum() >= cfg.min_live).to_numpy() & monthly.loc[t].notna().to_numpy()
        cov = factor_covariance(win, market)

        probs = np.full(len(cols), 0.5)
        for i, c in enumerate(cols):
            if not active[i]:
                continue
            fm = fms[c]
            tr = fm[(fm.index < t) & fm["target"].notna()]
            if len(tr) < cfg.min_train or t not in fm.index:
                continue
            Xc = [x for x in fm.columns if x not in ("target", "fwd_ret")]
            probs[i] = predict_prob(tr[Xc].to_numpy(), tr["target"].to_numpy(),
                                    fm.loc[[t], Xc].to_numpy(), seed=cfg.seed)
            prows.append({"date": t, "model": c, "prob": probs[i],
                          "fwd_ret": monthly.at[nxt, c]})

        w_raw = optimize(probs - 0.5, cov, active, cols, budget)
        w_ew = equal_weight(active, cov, budget.port_cap)
        ws = {
            "opt_raw": w_raw,
            "opt_blend": cfg.shrink_alpha * w_raw + (1 - cfg.shrink_alpha) * w_ew,
            "equal_weight": w_ew,
            "vol_parity": vol_parity(active, cov, budget.port_cap),
        }
        realised = monthly.loc[nxt].fillna(0.0).to_numpy()
        rec = {"date": nxt}
        for k, w in ws.items():
            turnover = float(np.abs(w - prev[k]).sum())
            rec[k] = float(w @ realised) - turnover * cfg.cost_bps / 1e4
            rec[f"pv_{k}"] = float(np.sqrt(max(w @ cov @ w, 0)))
            prev[k] = w * (1 + realised) / max(1 + float(w @ realised), 1e-9)
        rows.append(rec)
        wrows.append(pd.Series(ws["opt_blend"], index=cols, name=nxt))

    ret = pd.DataFrame(rows).set_index("date")
    return ret, pd.DataFrame(wrows), pd.DataFrame(prows)
