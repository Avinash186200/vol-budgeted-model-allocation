"""Walk-forward training, development-period selection and the single held-out run.

Timeline, in month-ends ``m``:

- the feature row ``m`` uses data up to the last day of ``m``;
- the target row ``s`` uses months ``s + 1 .. s + h``, so it is only known at the
  end of month ``s + h``;
- at decision month ``t`` a model may train on row ``s`` only if
  ``s + h + embargo <= t`` (purge the rows whose target is not yet realised, then
  drop ``embargo`` more months);
- the book chosen at ``t`` is held through month ``t + 1``.

The decision months are split once, in time order, into a development period and
a final held-out period (the last ``holdout_months``). Hyperparameters (the
regularisation of each model and the soft-blend temperature) are chosen by the
net Sharpe of the selector over the development period only. The held-out period
is then run once, with those choices frozen. Models are refit on an expanding
window every ``retrain_every`` months in both periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..min_variance.evaluation import sharpe
from .features import best_method, forward_scores, regime_features
from .methods import METHODS, Market, MethodPanel, SelectorConfig, run_book

GRID: dict[str, tuple[float, ...]] = {
    "logit": (0.03, 0.3, 3.0),     # inverse L2 strength C of the multinomial logistic regression
    "forest": (3, 10, 25),         # min_samples_leaf of the random forest
}
TEMPERATURES: tuple[float, ...] = (0.5, 1.0, 2.0)


def make_model(family: str, param: float):
    """A fresh, unfitted classifier."""
    if family == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=param, max_iter=5000))
    if family == "forest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=int(param), max_features="sqrt",
                                      random_state=0, n_jobs=1)
    raise ValueError(f"unknown model family: {family}")


@dataclass
class Dataset:
    """Features, targets and the decision months, all keyed by integer month position."""

    F: pd.DataFrame          # features, index = month position m
    y: pd.Series             # target (method index), index = month position s
    scores: pd.DataFrame     # forward scores behind the target
    dev: list[int]           # development decision months
    hold: list[int]          # held-out decision months
    features: list[str] = field(default_factory=list)


def train_rows(y_index: pd.Index, t: int, cfg: SelectorConfig) -> np.ndarray:
    """Target rows usable at decision month ``t``: ``s + horizon + embargo <= t``."""
    s = np.asarray(y_index)
    return s[s + cfg.horizon + cfg.embargo <= t]


def build_dataset(mk: Market, panel: MethodPanel, cfg: SelectorConfig) -> Dataset:
    """Features, targets and the development / held-out split of the decision months."""
    pos = {d: i for i, d in enumerate(mk.months)}
    F = regime_features(mk, panel, cfg)
    F.index = [pos[d] for d in F.index]
    sc = forward_scores(mk, panel, cfg)
    sc.index = [pos[d] for d in sc.index]
    sc = sc.loc[sc.index.intersection(F.index)]
    y = best_method(sc)
    decision = [t for t in F.index
                if t in panel.weights and t + 1 < len(mk.ends)
                and len(train_rows(y.index, t, cfg)) >= cfg.min_train]
    if len(decision) <= cfg.holdout_months + cfg.min_train // 3:
        raise ValueError(f"only {len(decision)} decision months; need more than holdout_months + a dev period")
    return Dataset(F=F, y=y, scores=sc, dev=decision[:-cfg.holdout_months], hold=decision[-cfg.holdout_months:],
                   features=list(F.columns))


def walk_forward_proba(ds: Dataset, months: list[int], family: str, param: float, cfg: SelectorConfig
                       ) -> pd.DataFrame:
    """Predicted probability of each method being best, for each decision month in ``months``.

    The model is refit, on every usable row so far, at the first month and then every
    ``retrain_every`` months. A method never seen as best in training gets probability 0.
    """
    K = len(METHODS)
    out = {}
    model, classes, last_fit = None, None, None
    for t in months:
        if model is None or t - last_fit >= cfg.retrain_every:
            rows = train_rows(ds.y.index, t, cfg)
            Xtr, ytr = ds.F.loc[rows].to_numpy(), ds.y.loc[rows].to_numpy()
            classes = np.unique(ytr)
            if len(classes) == 1:
                model = "constant"
            else:
                model = make_model(family, param).fit(Xtr, ytr)
            last_fit = t
        p = np.zeros(K)
        if model == "constant":
            p[classes[0]] = 1.0
        else:
            p[classes] = model.predict_proba(ds.F.loc[[t]].to_numpy())[0]
        out[t] = p
    return pd.DataFrame(out, index=list(METHODS)).T


def selector_weights(panel: MethodPanel, probs: pd.DataFrame, mode: str, temperature: float = 1.0
                     ) -> dict[int, np.ndarray]:
    """Turn per-month probabilities into a book.

    ``hard``: the book of the most probable method. ``soft``: the probability blend
    ``sum_k q_k w_k`` with ``q ∝ p^(1/temperature)`` (lower temperature, closer to hard).
    """
    out = {}
    for t, p in probs.iterrows():
        W = np.vstack([panel.weights[t][k] for k in METHODS])
        p = p.to_numpy(dtype=float)
        if mode == "hard":
            out[t] = W[int(np.argmax(p))]
        elif mode == "soft":
            q = np.power(np.clip(p, 0, None), 1.0 / temperature)
            q = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / len(q))
            out[t] = q @ W
        else:
            raise ValueError(f"unknown mode: {mode}")
    return out


def fixed_weights(panel: MethodPanel, months: list[int], method: str) -> dict[int, np.ndarray]:
    return {t: panel.weights[t][method] for t in months}


def blend_weights(panel: MethodPanel, months: list[int]) -> dict[int, np.ndarray]:
    """Equal blend of all methods: the "no model" ensemble."""
    return {t: np.mean([panel.weights[t][k] for k in METHODS], axis=0) for t in months}


def oracle_weights(panel: MethodPanel, ds: Dataset, months: list[int]) -> dict[int, np.ndarray]:
    """The method that did turn out best over the next ``h`` months. Uses the future: an upper bound."""
    return {t: panel.weights[t][METHODS[int(ds.y.loc[t])]] for t in months if t in ds.y.index}


@dataclass
class Development:
    """What the development period decided."""

    table: pd.DataFrame                    # one row per configuration tried, with its dev net Sharpe
    choices: dict[str, dict]               # selector name -> {"family", "param", "mode", "temperature"}
    best_fixed: str                        # fixed method with the best dev net Sharpe


def develop(mk: Market, panel: MethodPanel, ds: Dataset, cfg: SelectorConfig) -> Development:
    """Try every configuration over the development months and keep the best of each kind.

    Uses only ``ds.dev`` decision months, whose books are held through month
    ``max(ds.dev) + 1`` at the latest, and training rows realised by then. Nothing
    from the held-out period is read.
    """
    rows = []
    for family, params in GRID.items():
        for param in params:
            probs = walk_forward_proba(ds, ds.dev, family, param, cfg)
            for mode, temps in (("hard", (1.0,)), ("soft", TEMPERATURES)):
                for T in temps:
                    bk = run_book(mk, selector_weights(panel, probs, mode, T), cfg.cost_bps)
                    rows.append({"family": family, "param": param, "mode": mode, "temperature": T,
                                 "dev_net_sharpe": sharpe(bk.net)})
    table = pd.DataFrame(rows)
    choices = {}
    for family in GRID:
        for mode in ("hard", "soft"):
            sub = table[(table["family"] == family) & (table["mode"] == mode)]
            best = sub.loc[sub["dev_net_sharpe"].idxmax()]
            choices[f"{family}_{mode}"] = {"family": family, "param": float(best["param"]), "mode": mode,
                                           "temperature": float(best["temperature"]),
                                           "dev_net_sharpe": float(best["dev_net_sharpe"])}
    fixed = {k: sharpe(run_book(mk, fixed_weights(panel, ds.dev, k), cfg.cost_bps).net) for k in METHODS}
    return Development(table=table, choices=choices, best_fixed=max(fixed, key=fixed.get))


def run_holdout(mk: Market, panel: MethodPanel, ds: Dataset, dev: Development, cfg: SelectorConfig
                ) -> dict[str, dict]:
    """Run each chosen selector once over the held-out months. Returns books and probabilities."""
    out = {}
    for name, c in dev.choices.items():
        probs = walk_forward_proba(ds, ds.hold, c["family"], c["param"], cfg)
        w = selector_weights(panel, probs, c["mode"], c["temperature"])
        out[name] = {"probs": probs, "weights": w, "book": run_book(mk, w, cfg.cost_bps)}
    return out
