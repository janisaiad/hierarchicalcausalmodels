"""Synthetic benchmark for NumPyro/JAX variational estimators on CPU or GPU."""

from __future__ import annotations

import os
import time

import numpy as np

from hierarchicalcausalmodels.estimation import (
    NUMPYRO_AVAILABLE,
    ConditionalDensityEstimator,
    fit_variational_conditional_estimator,
    variational_credible_interval,
    variational_mean_prediction,
)

def _benchmark_family(family: str, y: np.ndarray, x: np.ndarray, estimator_kwargs: dict[str, object]) -> None:
    if not NUMPYRO_AVAILABLE:
        print("NumPyro/JAX is not available in this environment.")
        return
    start = time.perf_counter()
    state = fit_variational_conditional_estimator(
        y=y,
        x=x,
        family=family,
        n_components=int(estimator_kwargs.get("n_components", 2)),
        num_steps=int(estimator_kwargs.get("num_steps", 700)),
        learning_rate=float(estimator_kwargs.get("learning_rate", 0.02)),
        num_posterior_samples=int(estimator_kwargs.get("num_posterior_samples", 128)),
        device=str(estimator_kwargs.get("device", "cpu")),
    )
    elapsed = time.perf_counter() - start
    mean_pred = variational_mean_prediction(state, np.array([[0.0], [1.0]]))
    low, high = variational_credible_interval(state, np.array([[0.0], [1.0]]), credible_mass=0.8)
    print(f"\n[{family}]")
    print(f"  device       : {state.device}")
    print(f"  elapsed (s)  : {elapsed:.3f}")
    print(f"  mean preds   : {np.round(mean_pred, 4)}")
    print(f"  ci low       : {np.round(low, 4)}")
    print(f"  ci high      : {np.round(high, 4)}")


def main() -> None:
    rng = np.random.default_rng(77)
    x = rng.binomial(1, 0.5, size=1000).astype(float).reshape(-1, 1)
    device = os.environ.get("HCM_NUMPYRO_DEVICE", "cpu").strip().lower()

    y_beta = rng.beta(a=2.0 + 2.0 * x.ravel(), b=6.0 - 1.5 * x.ravel()).astype(float)
    _benchmark_family(
        "beta",
        y_beta,
        x,
        {"device": device, "num_steps": 700, "learning_rate": 0.02, "num_posterior_samples": 128},
    )

    y_gamma = rng.gamma(shape=4.0, scale=np.exp(-0.1 + 0.35 * x.ravel()) / 4.0).astype(float)
    _benchmark_family(
        "gamma",
        y_gamma,
        x,
        {"device": device, "num_steps": 700, "learning_rate": 0.02, "num_posterior_samples": 128},
    )

    comp = rng.binomial(1, 0.25 + 0.45 * x.ravel(), size=len(x))
    y_mix = np.where(
        comp == 0,
        rng.normal(loc=-1.0 + 1.2 * x.ravel(), scale=0.35, size=len(x)),
        rng.normal(loc=1.9 + 1.0 * x.ravel(), scale=0.45, size=len(x)),
    )
    _benchmark_family(
        "gaussian_mixture",
        y_mix,
        x,
        {
            "device": device,
            "n_components": 2,
            "num_steps": 900,
            "learning_rate": 0.02,
            "num_posterior_samples": 128,
        },
    )

    est = ConditionalDensityEstimator(
        family="gaussian_mixture",
        backend="numpyro",
        estimator_kwargs={
            "device": device,
            "n_components": 2,
            "num_steps": 900,
            "learning_rate": 0.02,
            "num_posterior_samples": 128,
        },
    )
    est.fit(y_mix, x)
    print("\n[gaussian_mixture direct estimator]")
    print(f"  E[Y|A=0]     : {est.expectation(np.array([0.0])):.4f}")
    print(f"  E[Y|A=1]     : {est.expectation(np.array([1.0])):.4f}")


if __name__ == "__main__":
    main()
