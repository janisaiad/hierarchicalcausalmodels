"""Per-unit fit and aggregate utilities for confounder HCM ATE estimation."""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

import numpy as np

from .parallel import ParallelBackend, parallel_map
from .torch_estimators import (
    TorchBatchedBernoulliState,
    TorchBatchedBetaState,
    TorchBatchedGaussianState,
    TorchBatchedGammaState,
    TorchBatchedPoissonState,
    estimate_ate_confounder_torch_batched as _estimate_ate_confounder_torch_batched,
    torch_fit_batched_beta,
    torch_fit_batched_bernoulli,
    torch_fit_batched_gaussian,
    torch_fit_batched_gamma,
    torch_fit_batched_poisson,
)

T = TypeVar("T")


def fit_per_unit_estimators(
    unit_data: list[tuple[np.ndarray, np.ndarray]],
    fit_fn: Callable[[tuple[np.ndarray, np.ndarray]], T],
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
) -> list[T]:
    """Run fit_fn on each unit's (a, y) and return list of results."""
    return parallel_map(
        unit_data,
        fit_fn,
        n_jobs=n_jobs,
        backend=parallel_backend,
    )


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
    parallel_backend: ParallelBackend = "threads",
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

    return parallel_map(
        range(n),
        fit_one,
        n_jobs=n_jobs,
        backend=parallel_backend,
    )


def estimate_ate_confounder(
    A: np.ndarray,
    Y: np.ndarray,
    regressor_class: type,
    n_jobs: int = 1,
    regressor_kwargs: Optional[dict[str, Any]] = None,
    parallel_backend: ParallelBackend = "threads",
) -> float:
    """Per-unit E[Y|A=1] - E[Y|A=0] then average over units."""
    regs = fit_regressors_per_unit(
        A,
        Y,
        regressor_class,
        n_jobs=n_jobs,
        regressor_kwargs=regressor_kwargs,
        parallel_backend=parallel_backend,
    )
    n = A.shape[0]
    ate_list: list[float] = []
    for i in range(n):
        reg = regs[i]
        pred_0 = float(reg.predict([[0.0]])[0])
        pred_1 = float(reg.predict([[1.0]])[0])
        ate_list.append(pred_1 - pred_0)
    return float(np.mean(ate_list))


def fit_torch_batched_regressor_per_unit(
    A: np.ndarray,
    Y: np.ndarray,
    family: str = "gaussian",
    device: str = "cpu",
    devices: Optional[list[str]] = None,
    ridge: float = 1e-4,
    max_iter: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
 ) -> (
    TorchBatchedGaussianState
    | TorchBatchedBernoulliState
    | TorchBatchedPoissonState
    | TorchBatchedGammaState
    | TorchBatchedBetaState
 ):
    """Fit one Torch batched regressor per unit on CPU or CUDA."""
    family_l = family.lower().strip()
    if family_l in {"gaussian", "normal"}:
        return torch_fit_batched_gaussian(
            x_batch=A,
            y_batch=Y,
            device=device,
            devices=devices,
            ridge=ridge,
        )
    if family_l == "bernoulli":
        return torch_fit_batched_bernoulli(
            x_batch=A,
            y_batch=Y,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
    if family_l == "poisson":
        return torch_fit_batched_poisson(
            x_batch=A,
            y_batch=Y,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
    if family_l == "gamma":
        return torch_fit_batched_gamma(
            x_batch=A,
            y_batch=Y,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
    if family_l == "beta":
        return torch_fit_batched_beta(
            x_batch=A,
            y_batch=Y,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
    raise NotImplementedError(
        f"Torch batched per-unit estimation is currently implemented for 'bernoulli', 'poisson', 'gaussian', 'beta', and 'gamma', got {family!r}."
    )


def estimate_ate_confounder_torch_batched(
    A: np.ndarray,
    Y: np.ndarray,
    family: str = "gaussian",
    device: str = "cpu",
    devices: Optional[list[str]] = None,
    ridge: float = 1e-4,
    max_iter: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> float:
    """Estimate per-unit ATE with a Torch batched backend on CPU or CUDA."""
    return _estimate_ate_confounder_torch_batched(
        a=A,
        y=Y,
        family=family,
        device=device,
        devices=devices,
        ridge=ridge,
        max_iter=max_iter,
        lr=lr,
        weight_decay=weight_decay,
    )


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
