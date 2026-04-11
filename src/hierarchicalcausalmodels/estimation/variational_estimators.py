"""Variational conditional estimators built with NumPyro/JAX."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Use a safe default on machines where JAX GPU initialization is unstable
# unless the user explicitly opts into another platform before import.
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = os.environ.get("HCM_NUMPYRO_JAX_PLATFORM", "cpu")

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive, SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDiagonalNormal
    from numpyro.optim import Adam

    NUMPYRO_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    numpyro = None  # type: ignore
    dist = None  # type: ignore
    Predictive = None  # type: ignore
    SVI = None  # type: ignore
    Trace_ELBO = None  # type: ignore
    AutoDiagonalNormal = None  # type: ignore
    Adam = None  # type: ignore
    NUMPYRO_AVAILABLE = False


@dataclass(frozen=True)
class VariationalConditionalState:
    """Posterior state for a variational conditional estimator."""

    family: str
    params: dict[str, Any]
    posterior_samples: dict[str, np.ndarray]
    design_dim: int
    n_components: int
    num_posterior_samples: int
    device: str


def _require_numpyro() -> None:
    if not NUMPYRO_AVAILABLE:
        raise RuntimeError("NumPyro/JAX is required for the 'numpyro' estimation backend.")


def _resolve_jax_device(device: Optional[str]) -> Any:
    _require_numpyro()
    default_devices = jax.devices()
    if not default_devices:
        raise RuntimeError("JAX sees no devices for NumPyro estimation.")
    if device is None:
        return default_devices[0]
    device_l = str(device).lower().strip()
    if device_l.startswith("gpu") or device_l.startswith("cuda"):
        try:
            gpus = jax.devices("cuda")
        except RuntimeError:
            try:
                gpus = jax.devices("gpu")
            except RuntimeError:
                gpus = []
        if not gpus:
            raise RuntimeError("A GPU device was requested for NumPyro, but JAX sees no GPU.")
        if ":" in device_l:
            index = int(device_l.split(":")[-1])
            if index >= len(gpus):
                raise RuntimeError(f"Requested GPU index {index}, but only {len(gpus)} GPU device(s) exist.")
            return gpus[index]
        return gpus[0]
    try:
        cpus = jax.devices("cpu")
    except RuntimeError:
        cpus = []
    if cpus:
        return cpus[0]
    return default_devices[0]


def _design_matrix(x: Optional[np.ndarray], y: np.ndarray) -> np.ndarray:
    if x is None:
        return np.ones((len(y), 1), dtype=np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)
    if x_np.shape[0] != len(y):
        raise ValueError("X and Y must have the same number of observations.")
    return np.concatenate([np.ones((len(y), 1), dtype=np.float32), x_np], axis=1)


def _posterior_array(samples: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(samples[key], dtype=np.float32)


def _beta_model(x: Any, y: Optional[Any] = None) -> None:
    coeff = numpyro.sample("coeff", dist.Normal(0.0, 1.0).expand([x.shape[1]]))
    raw_concentration = numpyro.sample("raw_concentration", dist.Normal(0.0, 1.0))
    concentration = jax.nn.softplus(raw_concentration) + 1e-3
    mu = jax.nn.sigmoid(jnp.matmul(x, coeff))
    alpha = jnp.clip(mu * concentration, 1e-4, None)
    beta = jnp.clip((1.0 - mu) * concentration, 1e-4, None)
    numpyro.sample("y", dist.Beta(alpha, beta), obs=y)


def _gamma_model(x: Any, y: Optional[Any] = None) -> None:
    coeff = numpyro.sample("coeff", dist.Normal(0.0, 1.0).expand([x.shape[1]]))
    raw_shape = numpyro.sample("raw_shape", dist.Normal(0.0, 1.0))
    shape = jax.nn.softplus(raw_shape) + 1e-3
    mean = jnp.exp(jnp.matmul(x, coeff))
    rate = jnp.clip(shape / mean, 1e-5, None)
    numpyro.sample("y", dist.Gamma(concentration=shape, rate=rate), obs=y)


def _gaussian_mixture_model(x: Any, y: Optional[Any] = None, n_components: int = 2) -> None:
    mix_logits = numpyro.sample("mix_logits", dist.Normal(0.0, 1.0).expand([n_components]))
    coeff = numpyro.sample("coeff", dist.Normal(0.0, 1.0).expand([n_components, x.shape[1]]))
    raw_scale = numpyro.sample("raw_scale", dist.Normal(0.0, 1.0).expand([n_components]))
    loc = jnp.matmul(x, coeff.T)
    scale = jax.nn.softplus(raw_scale) + 1e-3
    weights = jax.nn.softmax(mix_logits)
    mixture = dist.MixtureSameFamily(
        mixing_distribution=dist.Categorical(probs=weights),
        component_distribution=dist.Normal(loc=loc, scale=scale),
    )
    numpyro.sample("y", mixture, obs=y)


def fit_variational_conditional_estimator(
    y: np.ndarray,
    x: Optional[np.ndarray],
    family: str,
    *,
    n_components: int = 2,
    num_steps: int = 2500,
    learning_rate: float = 1e-2,
    num_posterior_samples: int = 256,
    seed: int = 0,
    device: Optional[str] = None,
) -> VariationalConditionalState:
    """Fit a variational conditional estimator with NumPyro."""
    _require_numpyro()
    family_l = family.lower().strip()
    if family_l not in {"beta", "gamma", "gaussian_mixture", "gmm"}:
        raise NotImplementedError(
            f"Variational NumPyro estimation is currently implemented for 'beta', 'gamma', and 'gaussian_mixture', got {family!r}."
        )

    y_np = np.asarray(y, dtype=np.float32).reshape(-1)
    if family_l == "beta":
        y_np = np.clip(y_np, 1e-6, 1.0 - 1e-6)
    elif family_l == "gamma":
        y_np = np.clip(y_np, 1e-6, None)

    design = _design_matrix(x, y_np)
    target_device = _resolve_jax_device(device)
    x_jax = jax.device_put(jnp.asarray(design), target_device)
    y_jax = jax.device_put(jnp.asarray(y_np), target_device)

    if family_l == "beta":
        model = _beta_model
        model_kwargs: dict[str, Any] = {}
        canonical_family = "beta"
    elif family_l == "gamma":
        model = _gamma_model
        model_kwargs = {}
        canonical_family = "gamma"
    else:
        model = _gaussian_mixture_model
        model_kwargs = {"n_components": int(max(n_components, 1))}
        canonical_family = "gaussian_mixture"

    guide = AutoDiagonalNormal(model)
    optimizer = Adam(float(learning_rate))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(int(seed)), int(num_steps), x_jax, y_jax, **model_kwargs)
    predictive = Predictive(guide, params=result.params, num_samples=int(num_posterior_samples))
    posterior_samples = predictive(jax.random.PRNGKey(int(seed) + 1), x_jax, None, **model_kwargs)
    posterior_np = {key: np.asarray(value) for key, value in posterior_samples.items()}
    return VariationalConditionalState(
        family=canonical_family,
        params=result.params,
        posterior_samples=posterior_np,
        design_dim=design.shape[1],
        n_components=int(model_kwargs.get("n_components", 1)),
        num_posterior_samples=int(num_posterior_samples),
        device=str(target_device),
    )


def _design_query(x_query: Optional[np.ndarray], design_dim: int) -> np.ndarray:
    if x_query is None:
        return np.ones((1, design_dim), dtype=np.float32)
    x_np = np.asarray(x_query, dtype=np.float32)
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    design = np.concatenate([np.ones((x_np.shape[0], 1), dtype=np.float32), x_np], axis=1)
    if design.shape[1] != design_dim:
        raise ValueError(f"Expected {design_dim - 1} conditioning feature(s), got {design.shape[1] - 1}.")
    return design


def variational_expectation_samples(
    state: VariationalConditionalState,
    x_query: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return posterior draws of E[Y|X=x] for each query point."""
    design = _design_query(x_query, state.design_dim)
    coeff = _posterior_array(state.posterior_samples, "coeff")
    if state.family == "beta":
        logits = np.einsum("qp,sp->sq", design, coeff)
        return 1.0 / (1.0 + np.exp(-logits))
    if state.family == "gamma":
        log_mean = np.einsum("qp,sp->sq", design, coeff)
        return np.exp(np.clip(log_mean, -12.0, 12.0))
    if state.family == "gaussian_mixture":
        mix_logits = _posterior_array(state.posterior_samples, "mix_logits")
        weights = np.exp(mix_logits - mix_logits.max(axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True)
        loc = np.einsum("qp,skp->sqk", design, coeff)
        return np.sum(weights[:, None, :] * loc, axis=2)
    raise NotImplementedError(f"Unsupported variational family {state.family!r}.")


def variational_density_samples(
    state: VariationalConditionalState,
    y_query: float,
    x_query: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return posterior draws of p(y|x) for each query point."""
    design = _design_query(x_query, state.design_dim)
    coeff = _posterior_array(state.posterior_samples, "coeff")
    y_val = float(y_query)
    if state.family == "beta":
        raw_concentration = _posterior_array(state.posterior_samples, "raw_concentration")
        concentration = np.log1p(np.exp(raw_concentration)) + 1e-3
        logits = np.einsum("qp,sp->sq", design, coeff)
        mu = 1.0 / (1.0 + np.exp(-logits))
        alpha = np.clip(mu * concentration[:, None], 1e-4, None)
        beta = np.clip((1.0 - mu) * concentration[:, None], 1e-4, None)
        y_clip = np.clip(y_val, 1e-6, 1.0 - 1e-6)
        log_pdf = (
            jax.scipy.special.gammaln(alpha + beta)
            - jax.scipy.special.gammaln(alpha)
            - jax.scipy.special.gammaln(beta)
            + (alpha - 1.0) * np.log(y_clip)
            + (beta - 1.0) * np.log(1.0 - y_clip)
        )
        return np.asarray(np.exp(log_pdf), dtype=np.float32)
    if state.family == "gamma":
        raw_shape = _posterior_array(state.posterior_samples, "raw_shape")
        shape = np.log1p(np.exp(raw_shape)) + 1e-3
        log_mean = np.einsum("qp,sp->sq", design, coeff)
        mean = np.exp(np.clip(log_mean, -12.0, 12.0))
        rate = np.clip(shape[:, None] / mean, 1e-5, None)
        y_pos = max(y_val, 1e-6)
        log_pdf = (
            shape[:, None] * np.log(rate)
            - np.asarray(jax.scipy.special.gammaln(shape))[:, None]
            + (shape[:, None] - 1.0) * np.log(y_pos)
            - rate * y_pos
        )
        return np.asarray(np.exp(log_pdf), dtype=np.float32)
    if state.family == "gaussian_mixture":
        mix_logits = _posterior_array(state.posterior_samples, "mix_logits")
        raw_scale = _posterior_array(state.posterior_samples, "raw_scale")
        weights = np.exp(mix_logits - mix_logits.max(axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True)
        scale = np.log1p(np.exp(raw_scale)) + 1e-3
        loc = np.einsum("qp,skp->sqk", design, coeff)
        z = (y_val - loc) / scale[:, None, :]
        normal_pdf = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * scale[:, None, :])
        return np.sum(weights[:, None, :] * normal_pdf, axis=2)
    raise NotImplementedError(f"Unsupported variational family {state.family!r}.")


def variational_mean_prediction(
    state: VariationalConditionalState,
    x_query: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Posterior mean of E[Y|X=x] for each query point."""
    return np.mean(variational_expectation_samples(state, x_query=x_query), axis=0)


def variational_credible_interval(
    state: VariationalConditionalState,
    x_query: Optional[np.ndarray] = None,
    credible_mass: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior credible interval for E[Y|X=x] for each query point."""
    draws = variational_expectation_samples(state, x_query=x_query)
    alpha = float((1.0 - credible_mass) / 2.0)
    lower = np.quantile(draws, alpha, axis=0)
    upper = np.quantile(draws, 1.0 - alpha, axis=0)
    return np.asarray(lower), np.asarray(upper)
