"""
market_timing_experiment.py -- test the idea: take MORE risk when the
MARKET is calm, LESS when the market is volatile (volatility timing keyed off
EXTERNAL market vol, i.e. SPY -- distinct from the earlier overlay that used the
portfolio's own vol).

Scales the VP daily return stream by a causal function of trailing SPY realized
vol (known through t-1). Sharpe is invariant to a constant scale, so only the
TIME-VARIATION of the risk dial matters. Specs:
  invvol_21 / invvol_63  : scale_t = 1 / spy_vol_{t-1}      (vol targeting)
  invvar_21 / invvar_63  : scale_t = 1 / spy_vol_{t-1}^2    (variance targeting)
  regime_21 / regime_63  : 1.3x when spy_vol below trailing-252d median (calm),
                            else 0.7x (volatile)  -- explicit risk-on/risk-off
Discipline: search {0,501} -> holdout {1002,1503,2004}. Beat VP on BOTH to count.
Uses the completed run's ret_vp streams + data/spy_daily.parquet. Fast.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

OFFSETS = [0, 501, 1002, 1503, 2004]
SEARCH, HOLDOUT = [0, 501], [1002, 1503, 2004]
RES = "results"


def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if len(r) and r.std() > 0 else np.nan


def load_spy_vol():
    spy = pd.read_parquet("data/spy_daily.parquet")
    s = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
    ret = s.pct_change()
    out = {}
    for lb in (21, 63):
        out[f"vol{lb}"] = ret.rolling(lb).std().shift(1)            # causal
    # trailing 252d median of 21d vol, for regime threshold (causal)
    out["med21"] = out["vol21"].rolling(252).median().shift(1)
    out["med63"] = out["vol63"].rolling(252).median().shift(1)
    return pd.DataFrame(out)


def timed(vp_dates, vp_vals, spyv, kind, lb):
    v = spyv[f"vol{lb}"].reindex(vp_dates).ffill().values
    if kind == "invvol":
        scale = 1.0 / np.where(v > 0, v, np.nan)
    elif kind == "invvar":
        scale = 1.0 / np.where(v > 0, v, np.nan) ** 2
    elif kind == "regime":
        med = spyv[f"med{lb}"].reindex(vp_dates).ffill().values
        scale = np.where(v <= med, 1.3, 0.7)
    scale = np.nan_to_num(scale, nan=np.nanmean(scale))
    return scale * vp_vals


def main():
    spyv = load_spy_vol()
    streams = {}
    for off in OFFSETS:
        suffix = f"_oos{off}" if off else ""
        df = pd.read_csv(f"{RES}/daily_wf_returns{suffix}.csv", index_col=0, parse_dates=True)
        streams[off] = (df.index, df["ret_vp"].astype(float).values)

    specs = [("VP", None, None)] + [(f"{k}_{lb}", k, lb)
             for k in ("invvol", "invvar", "regime") for lb in (21, 63)]

    def mean_sh(name, kind, lb, offs):
        vals = []
        for o in offs:
            d, vp = streams[o]
            s = vp if kind is None else timed(d, vp, spyv, kind, lb)
            vals.append(sharpe(s))
        return float(np.mean(vals)), [round(x, 3) for x in vals]

    vp_s, _ = mean_sh("VP", None, None, SEARCH)
    vp_h, vp_h_pw = mean_sh("VP", None, None, HOLDOUT)
    vp_a, _ = mean_sh("VP", None, None, OFFSETS)
    print(f"VP reference: search {vp_s:.3f} | holdout {vp_h:.3f} {vp_h_pw} | all5 {vp_a:.3f}\n")
    print(f"{'spec':<12s} {'search':>7s} {'holdout':>8s} {'dHold':>7s} {'all5':>7s}  per-window-holdout")
    rows = []
    for name, kind, lb in specs:
        sc, _ = mean_sh(name, kind, lb, SEARCH)
        hd, hpw = mean_sh(name, kind, lb, HOLDOUT)
        al, _ = mean_sh(name, kind, lb, OFFSETS)
        rows.append((name, sc, hd, al))
        tag = "  <-- beats VP both" if (kind and sc > vp_s and hd > vp_h) else ("  <-- VP" if kind is None else "")
        print(f"{name:<12s} {sc:7.3f} {hd:8.3f} {hd-vp_h:+7.3f} {al:7.3f}  {hpw}{tag}")

    fin = [r for r in rows if r[0] != "VP" and r[1] > vp_s and r[2] > vp_h]
    print("\nFINALISTS (beat VP on search AND holdout):", [r[0] for r in fin] if fin else "NONE")
    print("Note: SE on holdout mean Sharpe ~0.3-0.4 -> treat |dHold|<~0.3 as noise.")


if __name__ == "__main__":
    main()