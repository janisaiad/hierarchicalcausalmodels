"""Per-unit fit and aggregate utilities for confounder HCM ATE estimation."""

from __future__ import annotations

import typing
from typing import Any, Callable, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def fit_per_unit_estimators(
    unit_data: list[tuple[np.ndarray, np.ndarray]],
    fit_fn: Callable[[tuple[np.ndarray, np.ndarray]], T],
    n_jobs: int = 1,
) -> list[T]:
    """Run fit_fn on each unit's (a, y) and return list of results."""
    if n_jobs is None or n_jobs <= 1:
        return [fit_fn(d) for d in unit_data]
    try:
        from joblib import Parallel, delayed
        return list(Parallel(n_jobs=n_jobs)(delayed(fit_fn)(d) for d in unit_data))
    except ImportError:
        return [fit_fn(d) for d in unit_data]


def aggregate_per_unit_outputs(
    fits: list[Any],
    method: str = "mean",
    extract_fn: Optional[Callable[[Any], float]] = None,
) -> float:
    """Aggregate per-unit outputs; extract_fn maps each fit to a scalar, then we take mean."""
    if extract_fn is None:
        extract_fn = lambda x: float(x)
    values = [extract_fn(f) for f in fits]
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown aggregation method: {method}")


def fit_regressors_per_unit(
    A: np.ndarray,
    Y: np.ndarray,
    regressor_class: type,
    n_jobs: int = 1,
    regressor_kwargs: Optional[dict[str, Any]] = None,
) -> list[Any]:
    """Fit one regressor per unit (row). A, Y shape (n_units, n_obs). Returns list of fitted regressors."""
    n = A.shape[0]
    kwargs = dict(regressor_kwargs or {})

    def fit_one(i: int) -> Any:
        reg = regressor_class(**kwargs)
        a_i = A[i].reshape(-1, 1)
        y_i = Y[i]
        reg.fit(a_i, y_i)
        return reg

    if n_jobs is None or n_jobs <= 1:
        return [fit_one(i) for i in range(n)]
    try:
        from joblib import Parallel, delayed
        return list(Parallel(n_jobs=n_jobs)(delayed(fit_one)(i) for i in range(n)))
    except ImportError:
        return [fit_one(i) for i in range(n)]


def estimate_ate_confounder(
    A: np.ndarray,
    Y: np.ndarray,
    regressor_class: type,
    n_jobs: int = 1,
    regressor_kwargs: Optional[dict[str, Any]] = None,
) -> float:
    """Per-unit E[Y|A=1] - E[Y|A=0] then average over units."""
    regs = fit_regressors_per_unit(A, Y, regressor_class, n_jobs=n_jobs, regressor_kwargs=regressor_kwargs)
    n = A.shape[0]
    ate_list: list[float] = []
    for i in range(n):
        reg = regs[i]
        pred_0 = float(reg.predict([[0.0]])[0])
        pred_1 = float(reg.predict([[1.0]])[0])
        ate_list.append(pred_1 - pred_0)
    return float(np.mean(ate_list))


def device_kwargs_for_workers(
    n_workers: int,
    backend: str = "torch",
    use_cuda: bool = False,
    n_gpus: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return list of kwargs (e.g. device) for n_workers; for torch we assign CPU or GPU indices."""
    if backend != "torch":
        return [{} for _ in range(n_workers)]
    if not use_cuda or n_gpus is None or n_gpus <= 0:
        return [{"device": "cpu"} for _ in range(n_workers)]
    gpu_indices = [i % n_gpus for i in range(n_workers)]
    return [{"device": f"cuda:{g}" for g in gpu_indices}]
