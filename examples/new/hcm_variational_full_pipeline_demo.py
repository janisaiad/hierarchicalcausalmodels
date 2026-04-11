"""Small full-pipeline HCM demo with variational estimators and parallel evaluation."""

from __future__ import annotations

import time

import numpy as np

from hierarchicalcausalmodels.estimation import estimate_causal_effect


class _FakeResult:
    def __init__(self, formula_latex: str):
        self.identifiable = True
        self.formula_latex = formula_latex


def _run_case(name: str, y: np.ndarray, a: np.ndarray, family: str) -> None:
    result = _FakeResult(
        r"\sum_{Qy_a}{P\left(Qy_a\right) \cdot P\left(Qy\mid Qa,Qy_a\right)}"
    )
    data = {"A": a, "Y": y}
    families = {"A": "bernoulli", "Y": family, "Q^{y|a}": family}

    start_np = time.perf_counter()
    est_np = estimate_causal_effect(
        result,
        data=data,
        intervention={"Q^a": 1.0},
        distribution_families=families,
        estimator_backend="numpy",
        estimator_kwargs={family: {"n_components": 2}},
        n_mc_samples=128,
        random_seed=13,
        n_jobs=2,
        parallel_backend="processes",
    )
    elapsed_np = time.perf_counter() - start_np

    start_vi = time.perf_counter()
    est_vi = estimate_causal_effect(
        result,
        data=data,
        intervention={"Q^a": 1.0},
        distribution_families=families,
        estimator_backend="numpyro",
        estimator_kwargs={
            "__default__": {
                "device": "cpu",
                "num_steps": 350,
                "learning_rate": 0.02,
                "num_posterior_samples": 96,
            },
            family: {"n_components": 2},
        },
        n_mc_samples=128,
        random_seed=13,
        n_jobs=2,
        parallel_backend="processes",
    )
    elapsed_vi = time.perf_counter() - start_vi

    print(f"\n[{name}]")
    print(f"  family              : {family}")
    print(f"  numpy estimate      : {est_np:.4f}  ({elapsed_np:.3f}s)")
    print(f"  numpyro estimate    : {est_vi:.4f}  ({elapsed_vi:.3f}s)")
    print(f"  abs difference      : {abs(est_vi - est_np):.4f}")


def main() -> None:
    rng = np.random.default_rng(2026)
    n_units = 8
    n_subunits = 24

    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array([rng.binomial(1, u_i, size=n_subunits) for u_i in u], dtype=float)

    y_beta = np.array(
        [rng.beta(a=2.0 + 2.2 * a_i, b=6.0 - 1.6 * a_i) for a_i in a],
        dtype=float,
    )
    _run_case("beta graph", y_beta, a, "beta")

    y_gamma = np.array(
        [rng.gamma(shape=4.0, scale=np.exp(-0.2 + 0.35 * a_i) / 4.0) for a_i in a],
        dtype=float,
    )
    _run_case("gamma graph", y_gamma, a, "gamma")

    y_mix = []
    for i in range(n_units):
        comp = rng.binomial(1, 0.25 + 0.4 * a[i], size=n_subunits)
        y_i = np.where(
            comp == 0,
            rng.normal(loc=-1.0 + 1.1 * a[i], scale=0.30, size=n_subunits),
            rng.normal(loc=1.8 + 0.9 * a[i], scale=0.45, size=n_subunits),
        )
        y_mix.append(y_i)
    _run_case("gaussian mixture graph", np.asarray(y_mix, dtype=float), a, "gaussian_mixture")


if __name__ == "__main__":
    main()
