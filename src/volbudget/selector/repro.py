"""Reproducibility helpers: single-threaded BLAS and a hash of the code that produced a result.

The selector results move when the BLAS thread count changes (a different summation order shifts the
covariance and the fitted models by tiny amounts, and a hard argmax can then flip). The study scripts
therefore pin every common BLAS/OpenMP pool to one thread before numpy is imported, and stamp each
``results.json`` with ``code_hash`` so cached runs from older code are never mixed into a summary.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
ROOT = Path(__file__).resolve().parents[3]


def pin_threads(env: dict | None = None) -> dict:
    """Set each BLAS/OpenMP thread variable to "1" unless already set. Returns the mapping it updated."""
    env = os.environ if env is None else env
    for k in THREAD_VARS:
        env.setdefault(k, "1")
    return env


def code_hash() -> str:
    """SHA-256 (first 16 hex digits) of the package source and the selector scripts, line endings normalised."""
    files = sorted((ROOT / "src" / "volbudget").rglob("*.py")) + [
        ROOT / "scripts" / "run_selector.py", ROOT / "scripts" / "run_selector_sweep.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]
