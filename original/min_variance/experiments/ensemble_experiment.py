"""
ensemble_experiment.py -- ENSEMBLE OF PORTFOLIOS (untried, principled).

Not ensembling features/models (that axis is a documented dead end and the ML
view subtracts value here). Instead: blend the already-computed portfolio
return streams (VP, Variant D, Variant E, and the QP diagnostic variants) so
estimation errors partially cancel -- the "1/N across models" robustness idea.

Uses the EXISTING completed-run outputs (results/daily_wf_returns{suffix}.csv),
which contain 6 daily streams per window: ret_qp, ret_qp_sigtot, ret_qp_scaled,
ret_qp_scaled_nov2 (=Variant D), ret_variant_e (=E), ret_vp. No re-run needed.

Blends are EQUAL-CAPITAL (split NAV across sleeves) -> no look-ahead. Sharpe of
the blended daily stream is compared to VP across all 5 windows. Success = beat
VP mean Sharpe (<real figure removed>). Also reports the cross-stream correlation (high corr =>
little to gain from blending) and a causal vol-targeting overlay on VP.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

OFFSETS = [0, 501, 1002, 1503, 2004]
RES = "results"


def sharpe(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() * 252) / (r.std() * np.sqrt(252))


def vol_target(stream, lb=63, power=1):
    s = pd.Series(stream)
    sig = s.rolling(lb).std().shift(1)
    raw = (1.0 / np.power(sig.replace(0, np.nan), power)).replace([np.inf, -np.inf], np.nan)
    scale = (raw / raw.mean()).fillna(1.0).values
    return scale * s.values


def main():
    per_window = {}          # blend_name -> list of per-window Sharpe
    corr_accum = []
    streams_of_interest = ["ret_vp", "ret_qp_scaled_nov2", "ret_variant_e",
                           "ret_qp", "ret_qp_sigtot", "ret_qp_scaled"]

    for off in OFFSETS:
        suffix = f"_oos{off}" if off else ""
        df = pd.read_csv(f"{RES}/daily_wf_returns{suffix}.csv", index_col=0, parse_dates=True)
        avail = [c for c in streams_of_interest if c in df.columns]
        S = df[avail].astype(float)

        vp = S["ret_vp"].values
        e = S["ret_variant_e"].values
        d = S["ret_qp_scaled_nov2"].values

        blends = {
            "VP (base)":            vp,
            "VP+E (50/50)":         0.5 * vp + 0.5 * e,
            "VP+D+E (1/3)":         (vp + d + e) / 3.0,
            "all-6 equal":          S.mean(axis=1).values,
            "VP+E+voltgt(VP)":      (vp + e + vol_target(vp)) / 3.0,
            "voltgt(VP) 1/vol 63d": vol_target(vp, 63, 1),
            "VP+voltgt(VP)":        0.5 * vp + 0.5 * vol_target(vp, 63, 1),
        }
        for name, series in blends.items():
            per_window.setdefault(name, []).append(sharpe(series))

        # correlation of the core sleeves (how much diversification is possible)
        corr_accum.append(S[["ret_vp", "ret_qp_scaled_nov2", "ret_variant_e"]].corr())

    print("=== Ensemble of portfolios — mean Sharpe across 5 windows ===")
    vp_mean = np.mean(per_window["VP (base)"])
    order = sorted(per_window.items(), key=lambda kv: -np.mean(kv[1]))
    for name, vals in order:
        m = np.mean(vals)
        beat = int(np.sum(np.array(vals) > np.array(per_window["VP (base)"])))
        star = "  <-- VP" if name == "VP (base)" else ""
        print(f"  {name:<24s} mean {m:.3f}   delta {m - vp_mean:+.3f}   beats VP {beat}/5"
              f"   per-win {[round(v,2) for v in vals]}{star}")

    avg_corr = sum(corr_accum) / len(corr_accum)
    print("\n=== avg correlation VP / Variant D / Variant E (high => blending won't help) ===")
    print(avg_corr.round(3).to_string())
    print("\nNote: SE on the 5-window mean Sharpe is ~0.3 — treat deltas < ~0.3 as noise.")


if __name__ == "__main__":
    main()