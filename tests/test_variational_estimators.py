import numpy as np
import pytest

from hierarchicalcausalmodels.estimation import (
    NUMPYRO_AVAILABLE,
    ConditionalDensityEstimator,
    SubunitParamEstimator,
    estimate_causal_effect,
    fit_variational_conditional_estimator,
    variational_credible_interval,
    variational_mean_prediction,
)


class _FakeResult:
    def __init__(self, formula_latex: str):
        self.identifiable = True
        self.formula_latex = formula_latex


pytestmark = pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro/JAX not available")


@pytest.mark.parametrize("family", ["beta", "gamma"])
def test_variational_family_mean_predictions_are_ordered(family: str):
    rng = np.random.default_rng(401)
    x = rng.binomial(1, 0.5, size=600).astype(float)
    if family == "beta":
        y = rng.beta(a=2.0 + 2.0 * x, b=6.5 - 2.0 * x).astype(float)
    else:
        y = rng.gamma(shape=4.0, scale=np.exp(-0.2 + 0.35 * x) / 4.0).astype(float)

    state = fit_variational_conditional_estimator(
        y=y,
        x=x.reshape(-1, 1),
        family=family,
        num_steps=600,
        learning_rate=0.02,
        num_posterior_samples=128,
        device="cpu",
    )
    mean_pred = variational_mean_prediction(state, np.array([[0.0], [1.0]]))
    lower, upper = variational_credible_interval(state, np.array([[0.0], [1.0]]), credible_mass=0.8)

    assert mean_pred.shape == (2,)
    assert np.all(np.isfinite(mean_pred))
    assert mean_pred[1] > mean_pred[0]
    assert np.all(lower <= upper)


def test_variational_gaussian_mixture_conditional_estimator():
    rng = np.random.default_rng(402)
    x = rng.binomial(1, 0.5, size=900).astype(float)
    comp = rng.binomial(1, 0.25 + 0.4 * x, size=900)
    y = np.where(
        comp == 0,
        rng.normal(loc=-1.0 + 1.3 * x, scale=0.30, size=900),
        rng.normal(loc=1.7 + 1.2 * x, scale=0.45, size=900),
    )

    est = ConditionalDensityEstimator(
        family="gaussian_mixture",
        backend="numpyro",
        estimator_kwargs={
            "n_components": 2,
            "num_steps": 700,
            "learning_rate": 0.02,
            "num_posterior_samples": 128,
            "device": "cpu",
        },
    )
    est.fit(y, x.reshape(-1, 1))

    mu0 = est.expectation(np.array([0.0]))
    mu1 = est.expectation(np.array([1.0]))
    p0 = est.prob(mu0, np.array([0.0]))
    p1 = est.prob(mu1, np.array([1.0]))

    assert np.isfinite(mu0)
    assert np.isfinite(mu1)
    assert mu1 > mu0
    assert p0 > 0.0
    assert p1 > 0.0


def test_hcm_path_accepts_estimator_kwargs_and_numpyro_backend():
    rng = np.random.default_rng(403)
    n_units = 10
    n_subunits = 30
    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array([rng.binomial(1, u_i, size=n_subunits) for u_i in u], dtype=float)
    y = np.array(
        [rng.beta(a=2.0 + 2.0 * a_i, b=6.0 - 2.0 * a_i) for a_i in a],
        dtype=float,
    )

    result = _FakeResult(
        r"\sum_{Qy_a}{P\left(Qy_a\right) \cdot P\left(Qy\mid Qa,Qy_a\right)}"
    )
    estimate = estimate_causal_effect(
        result,
        data={"A": a, "Y": y},
        intervention={"Q^a": 1.0},
        distribution_families={"A": "bernoulli", "Y": "beta", "Q^{y|a}": "beta"},
        estimator_backend="numpyro",
        estimator_kwargs={
            "__default__": {
                "num_steps": 400,
                "learning_rate": 0.02,
                "num_posterior_samples": 96,
                "device": "cpu",
            }
        },
        n_mc_samples=128,
        random_seed=7,
    )
    baseline = estimate_causal_effect(
        result,
        data={"A": a, "Y": y},
        intervention={"Q^a": 1.0},
        distribution_families={"A": "bernoulli", "Y": "beta", "Q^{y|a}": "beta"},
        estimator_backend="numpy",
        estimator_kwargs={
            "__default__": {
                "num_steps": 400,
                "learning_rate": 0.02,
                "num_posterior_samples": 96,
                "device": "cpu",
            }
        },
        n_mc_samples=128,
        random_seed=7,
    )

    assert np.isfinite(estimate)
    assert np.isfinite(baseline)
    assert abs(estimate - baseline) < 3.5


def test_subunit_estimator_accepts_public_n_components():
    rng = np.random.default_rng(404)
    y_units = np.vstack([
        np.concatenate([
            rng.normal(-1.2, 0.3, size=18),
            rng.normal(0.2, 0.25, size=14),
            rng.normal(2.2, 0.4, size=18),
        ])
        for _ in range(6)
    ])
    q = SubunitParamEstimator(
        family="gaussian_mixture",
        estimator_kwargs={"n_components": 3, "max_iter_gmm": 200},
    ).fit(y_units)
    assert q.shape == (6, 9)
    assert np.allclose(q[:, :3].sum(axis=1), 1.0, atol=1e-3)
