"""Shared test set-up."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy: results depend on the BLAS thread count

try:  # cvxpy first: on some Windows installs, importing it after pandas crashes the interpreter
    import cvxpy  # noqa: F401
except ImportError:
    pass
