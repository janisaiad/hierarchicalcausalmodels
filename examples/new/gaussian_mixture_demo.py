"""Gaussian mixture estimator demo on synthetic conditional and subunit data."""

from __future__ import annotations

import numpy as np

from hierarchicalcausalmodels.estimation import ConditionalDensityEstimator, SubunitParamEstimator


def main() -> None:
    rng = np.random.default_rng(999)

    n = 1600
    a = rng.binomial(1, 0.5, size=n).astype(float)
    comp = rng.binomial(1, 0.30 + 0.35 * a, size=n)
    y = np.where(
        comp == 0,
        rng.normal(loc=-1.2 + 1.6 * a, scale=0.30, size=n),
        rng.normal(loc=1.8 + 1.1 * a, scale=0.45, size=n),
    )

    est = ConditionalDensityEstimator(
        family="gaussian_mixture",
        estimator_kwargs={"n_components": 2, "max_iter_gmm": 250, "random_state_gmm": 0},
    )
    est.fit(y, a.reshape(-1, 1))

    mu0 = est.expectation(np.array([0.0]))
    mu1 = est.expectation(np.array([1.0]))
    dens0 = est.prob(mu0, np.array([0.0]))
    dens1 = est.prob(mu1, np.array([1.0]))

    print("[gaussian_mixture conditional]")
    print(f"  E[Y|A=0]           : {mu0:.4f}")
    print(f"  E[Y|A=1]           : {mu1:.4f}")
    print(f"  mean gap           : {mu1 - mu0:.4f}")
    print(f"  density at mean(0) : {dens0:.6f}")
    print(f"  density at mean(1) : {dens1:.6f}")

    y_units = np.vstack([
        np.concatenate([
            rng.normal(-1.0 + 0.1 * i, 0.25, size=25),
            rng.normal(2.0 + 0.05 * i, 0.35, size=25),
        ])
        for i in range(12)
    ])
    q_params = SubunitParamEstimator(family="gaussian_mixture").fit(y_units)

    print("\n[gaussian_mixture subunit params]")
    print(f"  q_params shape     : {q_params.shape}")
    print(f"  first unit weights : {np.round(q_params[0, :2], 4)}")
    print(f"  first unit means   : {np.round(q_params[0, 2:4], 4)}")
    print(f"  first unit vars    : {np.round(q_params[0, 4:6], 4)}")


if __name__ == "__main__":
    main()
