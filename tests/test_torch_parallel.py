import numpy as np
import pytest
import torch

from hierarchicalcausalmodels.estimation import (
    ConditionalDensityEstimator,
    SubunitParamEstimator,
    estimate_ate_confounder_torch_batched,
    estimate_causal_effect,
    torch_compute_subunit_params,
    torch_conditional_expectations_per_unit,
)


class _FakeResult:
    def __init__(self, formula_latex: str):
        self.identifiable = True
        self.formula_latex = formula_latex


def test_torch_batched_gaussian_cpu_matches_signal():
    rng = np.random.default_rng(301)
    n_units = 24
    n_obs = 48
    a = rng.binomial(1, 0.5, size=(n_units, n_obs)).astype(float)
    y = 1.5 + 2.25 * a + rng.normal(scale=0.1, size=(n_units, n_obs))

    ate = estimate_ate_confounder_torch_batched(
        a,
        y,
        family="gaussian",
        device="cpu",
        ridge=1e-5,
    )

    assert abs(ate - 2.25) < 0.15


def test_torch_backend_hcm_matches_numpy_backend_on_cpu():
    rng = np.random.default_rng(302)
    n_units = 16
    n_subunits = 40

    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array([rng.binomial(1, u_i, size=n_subunits) for u_i in u], dtype=float)
    logits = 1.0 * a + u[:, None] - 0.25
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = np.array([rng.binomial(1, probs_i) for probs_i in probs], dtype=float)

    result = _FakeResult(
        r"\sum_{Qy_a}{P\left(Qy_a\right) \cdot P\left(Qy\mid Qa,Qy_a\right)}"
    )
    data = {"A": a, "Y": y}
    kwargs = {
        "distribution_families": {"A": "bernoulli", "Y": "bernoulli"},
        "intervention": {"Q^a": 1.0},
        "n_mc_samples": 256,
        "random_seed": 77,
    }

    est_numpy = estimate_causal_effect(
        result,
        data=data,
        estimator_backend="numpy",
        **kwargs,
    )
    est_torch = estimate_causal_effect(
        result,
        data=data,
        estimator_backend="torch",
        torch_kwargs={"device": "cpu", "max_iter": 120, "lr": 0.05},
        **kwargs,
    )

    assert np.isfinite(est_numpy)
    assert np.isfinite(est_torch)
    assert abs(est_numpy - est_torch) < 0.1


def test_torch_subunit_and_conditional_helpers_cpu():
    rng = np.random.default_rng(303)
    y_bin = rng.binomial(1, 0.3, size=(10, 20)).astype(float)
    q = torch_compute_subunit_params(y_bin, family="bernoulli", device="cpu")
    assert q.shape == (10,)
    assert np.all(q > 0.0)
    assert np.all(q < 1.0)

    x = rng.binomial(1, 0.5, size=(10, 20)).astype(float)
    y = rng.binomial(1, 0.2 + 0.5 * x, size=(10, 20)).astype(float)
    preds = torch_conditional_expectations_per_unit(
        y=y,
        x=x,
        eval_values=np.array([0.0, 1.0]),
        family="bernoulli",
        device="cpu",
        max_iter=120,
    )
    assert preds.shape == (10, 2)
    assert np.all(preds[:, 1] >= preds[:, 0] - 0.15)


@pytest.mark.parametrize(
    ("family", "y_builder", "lower", "upper"),
    [
        (
            "poisson",
            lambda rng, x: rng.poisson(lam=np.exp(0.2 + 0.5 * x)),
            0.8,
            2.5,
        ),
        (
            "gamma",
            lambda rng, x: rng.gamma(shape=3.0, scale=np.exp(-0.3 + 0.2 * x) / 3.0),
            0.4,
            1.5,
        ),
        (
            "beta",
            lambda rng, x: rng.beta(a=2.0 + 2.0 * x, b=6.0 - 2.0 * x),
            0.15,
            0.75,
        ),
    ],
)
def test_torch_extended_families_cpu(family, y_builder, lower, upper):
    rng = np.random.default_rng(305)
    x = rng.binomial(1, 0.5, size=(12, 36)).astype(float)
    y = y_builder(rng, x).astype(float)

    preds = torch_conditional_expectations_per_unit(
        y=y,
        x=x,
        eval_values=np.array([0.0, 1.0]),
        family=family,
        device="cpu",
        max_iter=180,
        lr=0.05,
    )

    assert preds.shape == (12, 2)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= lower - 1.0)
    assert np.all(preds <= upper + 3.0)


def test_gaussian_mixture_conditional_estimator_and_subunit_params():
    rng = np.random.default_rng(306)
    n = 800
    x = rng.binomial(1, 0.5, size=n).astype(float)
    comp = rng.binomial(1, 0.35 + 0.3 * x, size=n)
    y = np.where(
        comp == 0,
        rng.normal(loc=-1.0 + 1.5 * x, scale=0.35, size=n),
        rng.normal(loc=2.0 + 1.0 * x, scale=0.45, size=n),
    )

    est = ConditionalDensityEstimator(
        family="gaussian_mixture",
        estimator_kwargs={"n_components": 2, "max_iter_gmm": 200},
    )
    est.fit(y, x.reshape(-1, 1))

    mean_0 = est.expectation(np.array([0.0]))
    mean_1 = est.expectation(np.array([1.0]))
    dens_0 = est.prob(mean_0, np.array([0.0]))
    dens_1 = est.prob(mean_1, np.array([1.0]))

    assert np.isfinite(mean_0)
    assert np.isfinite(mean_1)
    assert mean_1 > mean_0
    assert dens_0 > 0.0
    assert dens_1 > 0.0

    y_units = np.vstack([
        np.concatenate([
            rng.normal(-1.0, 0.3, size=20),
            rng.normal(2.2, 0.4, size=20),
        ])
        for _ in range(8)
    ])
    q_params = SubunitParamEstimator(family="gaussian_mixture").fit(y_units)
    assert q_params.shape == (8, 6)
    weights = q_params[:, :2]
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_cuda_smoke():
    rng = np.random.default_rng(304)
    x = rng.binomial(1, 0.5, size=(8, 24)).astype(float)
    y = rng.binomial(1, 0.25 + 0.45 * x, size=(8, 24)).astype(float)

    preds = torch_conditional_expectations_per_unit(
        y=y,
        x=x,
        eval_values=np.array([0.0, 1.0]),
        family="bernoulli",
        device="cuda:0",
        max_iter=100,
    )

    assert preds.shape == (8, 2)
    assert np.all(np.isfinite(preds))
