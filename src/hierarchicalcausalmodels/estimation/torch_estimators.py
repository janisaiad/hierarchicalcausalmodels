"""Torch-based estimators for single-GPU and multi-GPU batched estimation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


TorchDeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class TorchBatchedGaussianState:
    """Parameters of a batched Gaussian linear model."""

    coefficients: np.ndarray
    sigma: np.ndarray
    devices_used: tuple[str, ...]


@dataclass(frozen=True)
class TorchBatchedBernoulliState:
    """Parameters of a batched Bernoulli logistic model."""

    coefficients: np.ndarray
    devices_used: tuple[str, ...]


@dataclass(frozen=True)
class TorchBatchedPoissonState:
    """Parameters of a batched Poisson log-linear model."""

    coefficients: np.ndarray
    devices_used: tuple[str, ...]


@dataclass(frozen=True)
class TorchBatchedGammaState:
    """Parameters of a batched Gamma regression model."""

    coefficients: np.ndarray
    shape: np.ndarray
    devices_used: tuple[str, ...]


@dataclass(frozen=True)
class TorchBatchedBetaState:
    """Parameters of a batched Beta regression model."""

    coefficients: np.ndarray
    concentration: np.ndarray
    devices_used: tuple[str, ...]


def _normalize_devices(
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
) -> list[str]:
    if devices is not None and len(devices) > 0:
        normalized = [str(torch.device(name)) for name in devices]
    else:
        normalized = [str(torch.device(device))]

    available = torch.cuda.device_count()
    result: list[str] = []
    for name in normalized:
        if name.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested, but CUDA is not available.")
            index = torch.device(name).index or 0
            if index >= available:
                raise RuntimeError(
                    f"Requested CUDA device {name}, but only {available} device(s) are available."
                )
        result.append(name)
    return result


def _to_tensor3d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError("Expected a 2D or 3D array.")


def _to_tensor2d(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected a 1D or 2D array.")


def _split_batch(n_batch: int, n_parts: int) -> list[slice]:
    if n_parts <= 1 or n_batch <= 1:
        return [slice(0, n_batch)]
    indices = np.array_split(np.arange(n_batch), min(n_parts, n_batch))
    slices: list[slice] = []
    for idx in indices:
        if len(idx) == 0:
            continue
        slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return slices


def _solve_batched_gaussian_single_device(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device_name: str,
    ridge: float,
) -> TorchBatchedGaussianState:
    x_t = torch.as_tensor(_to_tensor3d(x_batch), dtype=torch.float32, device=device_name)
    y_t = torch.as_tensor(_to_tensor2d(y_batch), dtype=torch.float32, device=device_name)
    batch_size, n_obs, n_feat = x_t.shape

    ones = torch.ones((batch_size, n_obs, 1), dtype=x_t.dtype, device=x_t.device)
    design = torch.cat([ones, x_t], dim=2)
    xt = design.transpose(1, 2)
    gram = xt @ design
    eye = torch.eye(n_feat + 1, dtype=x_t.dtype, device=x_t.device).expand(batch_size, -1, -1)
    if ridge > 0.0:
        gram = gram + ridge * eye
    rhs = xt @ y_t.unsqueeze(-1)
    coeff = torch.linalg.solve(gram, rhs).squeeze(-1)
    fitted = (design @ coeff.unsqueeze(-1)).squeeze(-1)
    sigma = torch.sqrt(torch.mean((y_t - fitted) ** 2, dim=1) + 1e-9)
    return TorchBatchedGaussianState(
        coefficients=coeff.detach().cpu().numpy(),
        sigma=sigma.detach().cpu().numpy(),
        devices_used=(device_name,),
    )


def _solve_batched_bernoulli_single_device(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device_name: str,
    max_iter: int,
    lr: float,
    weight_decay: float,
) -> TorchBatchedBernoulliState:
    x_t = torch.as_tensor(_to_tensor3d(x_batch), dtype=torch.float32, device=device_name)
    y_t = torch.as_tensor(_to_tensor2d(y_batch), dtype=torch.float32, device=device_name)
    y_t = torch.clamp(y_t, 1e-6, 1.0 - 1e-6)
    batch_size, n_obs, n_feat = x_t.shape

    ones = torch.ones((batch_size, n_obs, 1), dtype=x_t.dtype, device=x_t.device)
    design = torch.cat([ones, x_t], dim=2)
    coeff = torch.zeros((batch_size, n_feat + 1), dtype=x_t.dtype, device=x_t.device, requires_grad=True)
    optimizer = torch.optim.Adam([coeff], lr=lr, weight_decay=weight_decay)

    for _ in range(max_iter):
        optimizer.zero_grad()
        logits = (design * coeff[:, None, :]).sum(dim=2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction="mean")
        loss.backward()
        optimizer.step()

    return TorchBatchedBernoulliState(
        coefficients=coeff.detach().cpu().numpy(),
        devices_used=(device_name,),
    )


def _solve_batched_poisson_single_device(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device_name: str,
    max_iter: int,
    lr: float,
    weight_decay: float,
) -> TorchBatchedPoissonState:
    x_t = torch.as_tensor(_to_tensor3d(x_batch), dtype=torch.float32, device=device_name)
    y_t = torch.as_tensor(_to_tensor2d(y_batch), dtype=torch.float32, device=device_name).clamp_min(0.0)
    batch_size, n_obs, n_feat = x_t.shape

    ones = torch.ones((batch_size, n_obs, 1), dtype=x_t.dtype, device=x_t.device)
    design = torch.cat([ones, x_t], dim=2)
    coeff = torch.zeros((batch_size, n_feat + 1), dtype=x_t.dtype, device=x_t.device, requires_grad=True)
    optimizer = torch.optim.Adam([coeff], lr=lr, weight_decay=weight_decay)

    for _ in range(max_iter):
        optimizer.zero_grad()
        log_rate = (design * coeff[:, None, :]).sum(dim=2)
        rate = torch.exp(torch.clamp(log_rate, -10.0, 10.0))
        loss = (rate - y_t * log_rate).mean()
        loss.backward()
        optimizer.step()

    return TorchBatchedPoissonState(
        coefficients=coeff.detach().cpu().numpy(),
        devices_used=(device_name,),
    )


def _solve_batched_gamma_single_device(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device_name: str,
    max_iter: int,
    lr: float,
    weight_decay: float,
) -> TorchBatchedGammaState:
    x_t = torch.as_tensor(_to_tensor3d(x_batch), dtype=torch.float32, device=device_name)
    y_t = torch.as_tensor(_to_tensor2d(y_batch), dtype=torch.float32, device=device_name).clamp_min(1e-6)
    batch_size, n_obs, n_feat = x_t.shape

    ones = torch.ones((batch_size, n_obs, 1), dtype=x_t.dtype, device=x_t.device)
    design = torch.cat([ones, x_t], dim=2)
    coeff = torch.zeros((batch_size, n_feat + 1), dtype=x_t.dtype, device=x_t.device, requires_grad=True)
    raw_shape = torch.zeros((batch_size,), dtype=x_t.dtype, device=x_t.device, requires_grad=True)
    optimizer = torch.optim.Adam([coeff, raw_shape], lr=lr, weight_decay=weight_decay)

    for _ in range(max_iter):
        optimizer.zero_grad()
        log_mean = (design * coeff[:, None, :]).sum(dim=2)
        mean = torch.exp(torch.clamp(log_mean, -10.0, 10.0))
        shape = torch.nn.functional.softplus(raw_shape) + 1e-3
        scale = mean / shape[:, None]
        log_prob = (
            (shape[:, None] - 1.0) * torch.log(y_t)
            - y_t / scale
            - torch.lgamma(shape)[:, None]
            - shape[:, None] * torch.log(scale)
        )
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()

    return TorchBatchedGammaState(
        coefficients=coeff.detach().cpu().numpy(),
        shape=(torch.nn.functional.softplus(raw_shape) + 1e-3).detach().cpu().numpy(),
        devices_used=(device_name,),
    )


def _solve_batched_beta_single_device(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device_name: str,
    max_iter: int,
    lr: float,
    weight_decay: float,
) -> TorchBatchedBetaState:
    x_t = torch.as_tensor(_to_tensor3d(x_batch), dtype=torch.float32, device=device_name)
    y_t = torch.as_tensor(_to_tensor2d(y_batch), dtype=torch.float32, device=device_name)
    y_t = torch.clamp(y_t, 1e-6, 1.0 - 1e-6)
    batch_size, n_obs, n_feat = x_t.shape

    ones = torch.ones((batch_size, n_obs, 1), dtype=x_t.dtype, device=x_t.device)
    design = torch.cat([ones, x_t], dim=2)
    coeff = torch.zeros((batch_size, n_feat + 1), dtype=x_t.dtype, device=x_t.device, requires_grad=True)
    raw_conc = torch.zeros((batch_size,), dtype=x_t.dtype, device=x_t.device, requires_grad=True)
    optimizer = torch.optim.Adam([coeff, raw_conc], lr=lr, weight_decay=weight_decay)

    for _ in range(max_iter):
        optimizer.zero_grad()
        logit_mu = (design * coeff[:, None, :]).sum(dim=2)
        mu = torch.sigmoid(logit_mu).clamp(1e-6, 1.0 - 1e-6)
        concentration = torch.nn.functional.softplus(raw_conc) + 1e-3
        alpha = mu * concentration[:, None]
        beta = (1.0 - mu) * concentration[:, None]
        log_prob = (
            torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + (alpha - 1.0) * torch.log(y_t)
            + (beta - 1.0) * torch.log(1.0 - y_t)
        )
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()

    return TorchBatchedBetaState(
        coefficients=coeff.detach().cpu().numpy(),
        concentration=(torch.nn.functional.softplus(raw_conc) + 1e-3).detach().cpu().numpy(),
        devices_used=(device_name,),
    )


def torch_fit_batched_gaussian(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    ridge: float = 1e-4,
) -> TorchBatchedGaussianState:
    """Fit many Gaussian regressions in one Torch batch, optionally sharded across GPUs."""
    x_np = _to_tensor3d(x_batch)
    y_np = _to_tensor2d(y_batch)
    if x_np.shape[:2] != y_np.shape:
        raise ValueError("x_batch and y_batch must agree on batch and observation axes.")

    device_names = _normalize_devices(device=device, devices=devices)
    if len(device_names) == 1 or x_np.shape[0] <= 1:
        return _solve_batched_gaussian_single_device(x_np, y_np, device_names[0], ridge)

    batch_slices = _split_batch(x_np.shape[0], len(device_names))
    shard_args = [
        (x_np[slc], y_np[slc], device_names[idx], ridge)
        for idx, slc in enumerate(batch_slices)
    ]
    with ThreadPoolExecutor(max_workers=len(shard_args)) as executor:
        states = list(executor.map(lambda args: _solve_batched_gaussian_single_device(*args), shard_args))
    return TorchBatchedGaussianState(
        coefficients=np.concatenate([state.coefficients for state in states], axis=0),
        sigma=np.concatenate([state.sigma for state in states], axis=0),
        devices_used=tuple(device_names[: len(states)]),
    )


def torch_fit_batched_bernoulli(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    max_iter: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> TorchBatchedBernoulliState:
    """Fit many Bernoulli regressions in one Torch batch, optionally sharded across GPUs."""
    x_np = _to_tensor3d(x_batch)
    y_np = _to_tensor2d(y_batch)
    if x_np.shape[:2] != y_np.shape:
        raise ValueError("x_batch and y_batch must agree on batch and observation axes.")

    device_names = _normalize_devices(device=device, devices=devices)
    if len(device_names) == 1 or x_np.shape[0] <= 1:
        return _solve_batched_bernoulli_single_device(
            x_np,
            y_np,
            device_names[0],
            max_iter,
            lr,
            weight_decay,
        )

    batch_slices = _split_batch(x_np.shape[0], len(device_names))
    shard_args = [
        (x_np[slc], y_np[slc], device_names[idx], max_iter, lr, weight_decay)
        for idx, slc in enumerate(batch_slices)
    ]
    with ThreadPoolExecutor(max_workers=len(shard_args)) as executor:
        states = list(executor.map(lambda args: _solve_batched_bernoulli_single_device(*args), shard_args))
    return TorchBatchedBernoulliState(
        coefficients=np.concatenate([state.coefficients for state in states], axis=0),
        devices_used=tuple(device_names[: len(states)]),
    )


def torch_fit_batched_poisson(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    max_iter: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> TorchBatchedPoissonState:
    """Fit many Poisson regressions in one Torch batch, optionally sharded across GPUs."""
    x_np = _to_tensor3d(x_batch)
    y_np = _to_tensor2d(y_batch)
    if x_np.shape[:2] != y_np.shape:
        raise ValueError("x_batch and y_batch must agree on batch and observation axes.")

    device_names = _normalize_devices(device=device, devices=devices)
    if len(device_names) == 1 or x_np.shape[0] <= 1:
        return _solve_batched_poisson_single_device(
            x_np,
            y_np,
            device_names[0],
            max_iter,
            lr,
            weight_decay,
        )

    batch_slices = _split_batch(x_np.shape[0], len(device_names))
    shard_args = [
        (x_np[slc], y_np[slc], device_names[idx], max_iter, lr, weight_decay)
        for idx, slc in enumerate(batch_slices)
    ]
    with ThreadPoolExecutor(max_workers=len(shard_args)) as executor:
        states = list(executor.map(lambda args: _solve_batched_poisson_single_device(*args), shard_args))
    return TorchBatchedPoissonState(
        coefficients=np.concatenate([state.coefficients for state in states], axis=0),
        devices_used=tuple(device_names[: len(states)]),
    )


def torch_fit_batched_gamma(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    max_iter: int = 300,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> TorchBatchedGammaState:
    """Fit many Gamma regressions in one Torch batch, optionally sharded across GPUs."""
    x_np = _to_tensor3d(x_batch)
    y_np = _to_tensor2d(y_batch)
    if x_np.shape[:2] != y_np.shape:
        raise ValueError("x_batch and y_batch must agree on batch and observation axes.")

    device_names = _normalize_devices(device=device, devices=devices)
    if len(device_names) == 1 or x_np.shape[0] <= 1:
        return _solve_batched_gamma_single_device(
            x_np,
            y_np,
            device_names[0],
            max_iter,
            lr,
            weight_decay,
        )

    batch_slices = _split_batch(x_np.shape[0], len(device_names))
    shard_args = [
        (x_np[slc], y_np[slc], device_names[idx], max_iter, lr, weight_decay)
        for idx, slc in enumerate(batch_slices)
    ]
    with ThreadPoolExecutor(max_workers=len(shard_args)) as executor:
        states = list(executor.map(lambda args: _solve_batched_gamma_single_device(*args), shard_args))
    return TorchBatchedGammaState(
        coefficients=np.concatenate([state.coefficients for state in states], axis=0),
        shape=np.concatenate([state.shape for state in states], axis=0),
        devices_used=tuple(device_names[: len(states)]),
    )


def torch_fit_batched_beta(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    max_iter: int = 300,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> TorchBatchedBetaState:
    """Fit many Beta regressions in one Torch batch, optionally sharded across GPUs."""
    x_np = _to_tensor3d(x_batch)
    y_np = _to_tensor2d(y_batch)
    if x_np.shape[:2] != y_np.shape:
        raise ValueError("x_batch and y_batch must agree on batch and observation axes.")

    device_names = _normalize_devices(device=device, devices=devices)
    if len(device_names) == 1 or x_np.shape[0] <= 1:
        return _solve_batched_beta_single_device(
            x_np,
            y_np,
            device_names[0],
            max_iter,
            lr,
            weight_decay,
        )

    batch_slices = _split_batch(x_np.shape[0], len(device_names))
    shard_args = [
        (x_np[slc], y_np[slc], device_names[idx], max_iter, lr, weight_decay)
        for idx, slc in enumerate(batch_slices)
    ]
    with ThreadPoolExecutor(max_workers=len(shard_args)) as executor:
        states = list(executor.map(lambda args: _solve_batched_beta_single_device(*args), shard_args))
    return TorchBatchedBetaState(
        coefficients=np.concatenate([state.coefficients for state in states], axis=0),
        concentration=np.concatenate([state.concentration for state in states], axis=0),
        devices_used=tuple(device_names[: len(states)]),
    )


def _predict_from_coefficients(coefficients: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    x_np = np.asarray(x_query, dtype=np.float32)
    if x_np.ndim == 2:
        x_np = x_np[:, :, None]
    if x_np.ndim != 3:
        raise ValueError("x_query must be a 2D or 3D array.")
    if x_np.shape[0] != coefficients.shape[0]:
        if coefficients.shape[0] == 1:
            x_np = np.repeat(x_np[None, ...], 1, axis=0)[0:1]
        else:
            raise ValueError("Batch dimension mismatch between coefficients and x_query.")
    ones = np.ones((x_np.shape[0], x_np.shape[1], 1), dtype=np.float32)
    design = np.concatenate([ones, x_np], axis=2)
    return np.sum(design * coefficients[:, None, :], axis=2)


def torch_predict_batched_gaussian(
    state: TorchBatchedGaussianState,
    x_query: np.ndarray,
) -> np.ndarray:
    """Predict conditional means for a fitted batched Gaussian model."""
    return _predict_from_coefficients(state.coefficients.astype(np.float32), x_query)


def torch_predict_batched_bernoulli(
    state: TorchBatchedBernoulliState,
    x_query: np.ndarray,
) -> np.ndarray:
    """Predict Bernoulli probabilities for a fitted batched logistic model."""
    logits = _predict_from_coefficients(state.coefficients.astype(np.float32), x_query)
    return 1.0 / (1.0 + np.exp(-logits))


def torch_predict_batched_poisson(
    state: TorchBatchedPoissonState,
    x_query: np.ndarray,
) -> np.ndarray:
    """Predict Poisson conditional means for a fitted batched log-linear model."""
    log_rate = _predict_from_coefficients(state.coefficients.astype(np.float32), x_query)
    return np.exp(np.clip(log_rate, -10.0, 10.0))


def torch_predict_batched_gamma_mean(
    state: TorchBatchedGammaState,
    x_query: np.ndarray,
) -> np.ndarray:
    """Predict Gamma conditional means for a fitted batched Gamma model."""
    log_mean = _predict_from_coefficients(state.coefficients.astype(np.float32), x_query)
    return np.exp(np.clip(log_mean, -10.0, 10.0))


def torch_predict_batched_beta_mean(
    state: TorchBatchedBetaState,
    x_query: np.ndarray,
) -> np.ndarray:
    """Predict Beta conditional means for a fitted batched Beta model."""
    logit_mu = _predict_from_coefficients(state.coefficients.astype(np.float32), x_query)
    return 1.0 / (1.0 + np.exp(-logit_mu))


def torch_compute_subunit_params(
    y: np.ndarray,
    family: str,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
) -> np.ndarray:
    """Compute per-unit subunit parameters on Torch for supported families."""
    family_l = family.lower().strip()
    y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim != 2:
        raise ValueError("y must be a 2D array of shape (n_units, n_subunits).")

    device_names = _normalize_devices(device=device, devices=devices)
    if len(device_names) > 1 and y_np.shape[0] > 1:
        shards = _split_batch(y_np.shape[0], len(device_names))
        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            parts = list(
                executor.map(
                    lambda args: torch_compute_subunit_params(*args),
                    [(y_np[slc], family_l, device_names[idx], None) for idx, slc in enumerate(shards)],
                )
            )
        return np.concatenate(parts, axis=0)

    y_t = torch.as_tensor(y_np, dtype=torch.float32, device=device_names[0])
    if family_l == "bernoulli":
        values = torch.clamp(torch.mean(y_t, dim=1), 1e-6, 1.0 - 1e-6)
        return values.detach().cpu().numpy()
    if family_l == "poisson":
        values = torch.mean(torch.clamp_min(y_t, 0.0), dim=1).clamp_min(1e-10)
        return values.detach().cpu().numpy()
    if family_l in {"gaussian", "normal"}:
        mean = torch.mean(y_t, dim=1)
        var = torch.var(y_t, dim=1, unbiased=False).clamp_min(1e-10)
        return torch.stack([mean, var], dim=1).detach().cpu().numpy()
    if family_l == "gamma":
        y_pos = torch.clamp_min(y_t, 1e-6)
        mean = torch.mean(y_pos, dim=1).clamp_min(1e-10)
        var = torch.var(y_pos, dim=1, unbiased=False).clamp_min(1e-10)
        shape = (mean ** 2 / var).clamp_min(1e-3)
        scale = (var / mean).clamp_min(1e-10)
        return torch.stack([shape, scale], dim=1).detach().cpu().numpy()
    if family_l == "beta":
        y_clip = torch.clamp(y_t, 1e-6, 1.0 - 1e-6)
        mean = torch.mean(y_clip, dim=1).clamp(1e-6, 1.0 - 1e-6)
        var = torch.var(y_clip, dim=1, unbiased=False).clamp_min(1e-10)
        concentration = (mean * (1.0 - mean) / var - 1.0).clamp_min(1e-2)
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        return torch.stack([alpha, beta], dim=1).detach().cpu().numpy()
    raise NotImplementedError(
        f"Torch subunit parameter estimation is currently implemented for 'bernoulli', 'poisson', 'gaussian', 'beta', and 'gamma', got {family!r}."
    )


def torch_conditional_expectations_per_unit(
    y: np.ndarray,
    x: np.ndarray,
    eval_values: np.ndarray,
    family: str,
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    ridge: float = 1e-4,
    max_iter: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    """Fit per-unit conditional models and evaluate them at shared query values."""
    family_l = family.lower().strip()
    x_batch = _to_tensor3d(x)
    y_batch = _to_tensor2d(y)
    eval_np = np.asarray(eval_values, dtype=np.float32).reshape(1, -1, 1)
    eval_batch = np.repeat(eval_np, x_batch.shape[0], axis=0)

    if family_l in {"gaussian", "normal"}:
        state = torch_fit_batched_gaussian(
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            devices=devices,
            ridge=ridge,
        )
        return torch_predict_batched_gaussian(state, eval_batch)
    if family_l == "bernoulli":
        state = torch_fit_batched_bernoulli(
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
        return torch_predict_batched_bernoulli(state, eval_batch)
    if family_l == "poisson":
        state = torch_fit_batched_poisson(
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
        return torch_predict_batched_poisson(state, eval_batch)
    if family_l == "gamma":
        state = torch_fit_batched_gamma(
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
        return torch_predict_batched_gamma_mean(state, eval_batch)
    if family_l == "beta":
        state = torch_fit_batched_beta(
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            devices=devices,
            max_iter=max_iter,
            lr=lr,
            weight_decay=weight_decay,
        )
        return torch_predict_batched_beta_mean(state, eval_batch)
    raise NotImplementedError(
        f"Torch conditional estimation is currently implemented for 'bernoulli', 'poisson', 'gaussian', 'beta', and 'gamma', got {family!r}."
    )


def estimate_ate_confounder_torch_batched(
    a: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    device: TorchDeviceLike = "cpu",
    devices: Optional[list[str]] = None,
    ridge: float = 1e-4,
    max_iter: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
) -> float:
    """Estimate per-unit ATE with a Torch batched regressor on CPU or CUDA."""
    eval_values = np.array([0.0, 1.0], dtype=np.float32)
    preds = torch_conditional_expectations_per_unit(
        y=y,
        x=a,
        eval_values=eval_values,
        family=family,
        device=device,
        devices=devices,
        ridge=ridge,
        max_iter=max_iter,
        lr=lr,
        weight_decay=weight_decay,
    )
    return float(np.mean(preds[:, 1] - preds[:, 0]))


class MLPRegressorPerUnit:
    """Minimal MLP regressor for (A, Y) per unit; .fit(X, y), .predict(X) compatible with sklearn-style usage."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        max_epochs: int = 50,
        hidden_sizes: tuple[int, ...] = (8,),
        lr: float = 1e-2,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self._model: Optional[torch.nn.Module] = None
        self._fitted = False

    def _build_model(self, input_size: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        prev = input_size
        for h in self.hidden_sizes:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressorPerUnit":
        """Fit MLP on (X, y); X shape (n, 1) or (n,)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        in_size = X.shape[1]
        self._model = self._build_model(in_size)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy(y).reshape(-1, 1).to(self.device)
        self._model.train()
        for _ in range(self.max_epochs):
            opt.zero_grad()
            out = self._model(Xt)
            loss = ((out - yt) ** 2).mean()
            loss.backward()
            opt.step()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict; X shape (n, 1) or (n,)."""
        if not self._fitted or self._model is None:
            raise RuntimeError("MLPRegressorPerUnit not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).to(self.device)
            out = self._model(Xt)
        return out.cpu().numpy().ravel()
