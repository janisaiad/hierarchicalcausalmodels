"""
Optional NumPy / Numba helpers for hot paths in formula evaluation (large MC batches).

Numba is **optional**: if ``numba`` is not installed, everything falls back to NumPy
(and SciPy ``expit`` when available).  Set ``HCM_DISABLE_NUMBA=1`` to force the
non-Numba path even when Numba is installed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

NUMBA_AVAILABLE = False
_expit_numba_impl = None

if os.environ.get("HCM_DISABLE_NUMBA", "").strip().lower() not in ("1", "true", "yes"):
    try:
        from numba import njit

        NUMBA_AVAILABLE = True

        @njit(cache=True)
        def _expit_numba_impl_1d(z: np.ndarray) -> np.ndarray:
            n = z.shape[0]
            out = np.empty(n, dtype=np.float64)
            for i in range(n):
                v = z[i]
                if v >= 0.0:
                    ev = np.exp(-v)
                    out[i] = 1.0 / (1.0 + ev)
                else:
                    ev = np.exp(v)
                    out[i] = ev / (1.0 + ev)
            return out

        _expit_numba_impl = _expit_numba_impl_1d
    except ImportError:
        NUMBA_AVAILABLE = False


def expit_array(z: np.ndarray) -> np.ndarray:
    """Stable logistic sigmoid applied element-wise (1d or nd; ravel internally for Numba)."""
    z = np.asarray(z, dtype=np.float64)
    flat = z.ravel()
    if (
        NUMBA_AVAILABLE
        and _expit_numba_impl is not None
        and flat.size >= 4096
    ):
        out_flat = _expit_numba_impl(flat)
        return np.asarray(out_flat, dtype=np.float64).reshape(z.shape)
    try:
        from scipy.special import expit

        return np.asarray(expit(z), dtype=np.float64)
    except ImportError:
        zc = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-zc))


def logistic_positive_proba_batch(lr: Any, X_batch: np.ndarray) -> np.ndarray:
    """
    P(Y=1|X) for sklearn's binary LogisticRegression, without predict_proba overhead.

    Uses ``decision_function`` (BLAS) then a stable sigmoid.  Falls back to
    ``predict_proba`` if the estimator API is nonstandard.
    """
    X = np.asarray(X_batch, dtype=np.float64)
    if hasattr(lr, "decision_function"):
        logits = np.asarray(lr.decision_function(X), dtype=np.float64).reshape(-1)
        return expit_array(logits)
    if hasattr(lr, "predict_proba"):
        return np.asarray(lr.predict_proba(X)[:, 1], dtype=np.float64)
    return np.clip(np.asarray(lr.predict(X), dtype=np.float64), 0.0, 1.0)


def linear_predict_batch(lr: Any, X_batch: np.ndarray) -> np.ndarray:
    """
    LinearRegression-style ``predict`` using ``coef_`` and ``intercept_`` when present
    (avoids a bit of sklearn overhead on huge batches).
    """
    X = np.asarray(X_batch, dtype=np.float64)
    coef = getattr(lr, "coef_", None)
    icept = getattr(lr, "intercept_", None)
    if coef is not None and icept is not None:
        c = np.asarray(coef, dtype=np.float64).ravel()
        b = float(np.asarray(icept, dtype=np.float64).ravel()[0])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ c + b
    return np.asarray(lr.predict(np.atleast_2d(X)), dtype=np.float64).reshape(-1)
