"""Quick convergence checks for Poisson, Beta, and Gamma families."""

from __future__ import annotations

import numpy as np

from hierarchicalcausalmodels.estimation import (
    ConditionalDensityEstimator,
    estimate_ate_confounder_torch_batched,
    torch_compute_subunit_params,
    torch_conditional_expectations_per_unit,
)


def _run_family_demo(family: str, y_builder, expected_gap: float) -> None:
    rng = np.random.default_rng(123)
    n_units = 32
    n_obs = 80
    a = rng.binomial(1, 0.5, size=(n_units, n_obs)).astype(float)
    y = y_builder(rng, a).astype(float)

    preds = torch_conditional_expectations_per_unit(
        y=y,
        x=a,
        eval_values=np.array([0.0, 1.0]),
        family=family,
        device="cpu",
        max_iter=220,
        lr=0.05,
    )
    ate = float(np.mean(preds[:, 1] - preds[:, 0]))
    q_params = torch_compute_subunit_params(y, family=family, device="cpu")

    flat_est = ConditionalDensityEstimator(
        family=family,
        backend="torch",
        torch_kwargs={"device": "cpu", "max_iter": 220, "lr": 0.05},
    )
    flat_est.fit(y.ravel(), np.repeat(a[:, :, None], 1, axis=2).reshape(-1, 1))
    mu0 = flat_est.expectation(np.array([0.0]))
    mu1 = flat_est.expectation(np.array([1.0]))

    print(f"\n[{family}]")
    print(f"  q_params shape            : {q_params.shape}")
    print(f"  per-unit mean gap         : {ate:.4f}")
    print(f"  pooled E[Y|A=0], E[Y|A=1] : {mu0:.4f}, {mu1:.4f}")
    print(f"  expected rough gap        : {expected_gap:.4f}")


def main() -> None:
    _run_family_demo(
        "poisson",
        lambda rng, a: rng.poisson(np.exp(0.1 + 0.6 * a)),
        expected_gap=np.exp(0.7) - np.exp(0.1),
    )
    _run_family_demo(
        "gamma",
        lambda rng, a: rng.gamma(shape=3.5, scale=np.exp(-0.2 + 0.35 * a) / 3.5),
        expected_gap=np.exp(0.15) - np.exp(-0.2),
    )
    _run_family_demo(
        "beta",
        lambda rng, a: rng.beta(a=2.0 + 2.5 * a, b=7.0 - 2.0 * a),
        expected_gap=0.5625 - (2.0 / 9.0),
    )

    rng = np.random.default_rng(456)
    a = rng.binomial(1, 0.5, size=(24, 60)).astype(float)
    y = rng.gamma(shape=3.0, scale=np.exp(-0.25 + 0.30 * a) / 3.0)
    ate_gamma = estimate_ate_confounder_torch_batched(
        a,
        y,
        family="gamma",
        device="cpu",
        max_iter=220,
        lr=0.05,
    )
    print(f"\n[gamma] estimate_ate_confounder_torch_batched: {ate_gamma:.4f}")


if __name__ == "__main__":
    main()
