"""
config.py — all settings for the Client CPO solution.
"""
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_RETURNS   = ROOT.parent / "cpo_data_obf.xlsx"
MACRO_PARQUET = DATA_DIR / "macro_features.parquet"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Pre-built validated feature set (same features used in VIXY / SPY models)
FEATS_CSV     = Path("<local-path>")

# ── API keys ────────────────────────────────────────────────────────────────
FINNHUB_KEY = "<REDACTED>"

# ── walk-forward settings ───────────────────────────────────────────────────
OOS_MONTHS       = 24    # last 24 months = OOS  (~Jun 2020 – May 2022)
MIN_TRAIN_MONTHS = 36    # min expanding-window size before first prediction
COV_WINDOW       = 36    # months of history used for covariance estimation

# ── model ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
LGB_PARAMS = dict(
    n_estimators    = 200,
    learning_rate   = 0.05,
    num_leaves      = 15,
    min_child_samples = 10,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    random_state    = RANDOM_STATE,
    verbose         = -1,
)

# ── Client constraint structure ────────────────────────────────────────────
PORT_VOL_CAP = 0.06   # 6% annualized

# per-model annualized vol caps  (weight × ann_vol ≤ cap)
MODEL_VOL_CAPS = {i: 0.01 for i in range(1, 63)}   # default 1%
MODEL_VOL_CAPS[17] = 0.007
MODEL_VOL_CAPS[18] = 0.005
MODEL_VOL_CAPS[45] = 0.007

# group definitions and their annualized vol caps
GROUPS = {
    1: (list(range(1,  13)), 0.030),
    2: (list(range(13, 26)), 0.036),
    3: (list(range(26, 42)), 0.030),
    4: (list(range(42, 55)), 0.030),
    5: (list(range(55, 61)), 0.013),
    6: (list(range(61, 63)), 0.015),
}
