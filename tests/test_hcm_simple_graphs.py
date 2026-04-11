import numpy as np

from hierarchicalcausalmodels.estimation import estimate_causal_effect


class _FakeResult:
    def __init__(self, formula_latex: str):
        self.identifiable = True
        self.formula_latex = formula_latex


def _run_simple_hcm_graph(family: str, y: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    result = _FakeResult(
        r"\sum_{Qy_a}{P\left(Qy_a\right) \cdot P\left(Qy\mid Qa,Qy_a\right)}"
    )
    data = {"A": a, "Y": y}
    distribution_families = {"A": "bernoulli", "Y": family, "Q^{y|a}": family}

    est_numpy = estimate_causal_effect(
        result,
        data=data,
        intervention={"Q^a": 1.0},
        distribution_families=distribution_families,
        estimator_backend="numpy",
        estimator_kwargs={family: {"n_components": 2}},
        n_mc_samples=96,
        random_seed=11,
        n_jobs=2,
        parallel_backend="processes",
    )
    est_numpyro = estimate_causal_effect(
        result,
        data=data,
        intervention={"Q^a": 1.0},
        distribution_families=distribution_families,
        estimator_backend="numpyro",
        estimator_kwargs={
            "__default__": {
                "device": "cpu",
                "num_steps": 500,
                "learning_rate": 0.02,
                "num_posterior_samples": 128,
            },
            "gaussian_mixture": {
                "num_steps": 1400,
                "learning_rate": 0.01,
                "num_posterior_samples": 192,
                "n_components": 2,
            },
        },
        n_mc_samples=96,
        random_seed=11,
        n_jobs=2,
        parallel_backend="threads",
    )
    return float(est_numpy), float(est_numpyro)


def test_simple_hcm_gamma_graph_agrees_with_numpyro():
    rng = np.random.default_rng(501)
    n_units = 6
    n_subunits = 20
    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array([rng.binomial(1, u_i, size=n_subunits) for u_i in u], dtype=float)
    y = np.array(
        [rng.gamma(shape=4.0, scale=np.exp(-0.25 + 0.4 * a_i) / 4.0) for a_i in a],
        dtype=float,
    )

    est_numpy, est_numpyro = _run_simple_hcm_graph("gamma", y, a)
    assert np.isfinite(est_numpy)
    assert np.isfinite(est_numpyro)
    assert abs(est_numpy - est_numpyro) < 0.15


def test_simple_hcm_beta_graph_agrees_with_numpyro():
    rng = np.random.default_rng(502)
    n_units = 6
    n_subunits = 20
    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array([rng.binomial(1, u_i, size=n_subunits) for u_i in u], dtype=float)
    y = np.array(
        [rng.beta(a=2.0 + 2.0 * a_i, b=6.2 - 1.7 * a_i) for a_i in a],
        dtype=float,
    )

    est_numpy, est_numpyro = _run_simple_hcm_graph("beta", y, a)
    assert np.isfinite(est_numpy)
    assert np.isfinite(est_numpyro)
    assert abs(est_numpy - est_numpyro) < 0.4


def test_simple_hcm_gaussian_mixture_graph_direction_and_gap():
    rng = np.random.default_rng(503)
    n_units = 6
    n_subunits = 24
    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array([rng.binomial(1, u_i, size=n_subunits) for u_i in u], dtype=float)

    y_rows = []
    for i in range(n_units):
        comp = rng.binomial(1, 0.20 + 0.45 * a[i], size=n_subunits)
        y_i = np.where(
            comp == 0,
            rng.normal(loc=-1.1 + 1.2 * a[i], scale=0.25, size=n_subunits),
            rng.normal(loc=1.9 + 0.9 * a[i], scale=0.35, size=n_subunits),
        )
        y_rows.append(y_i)
    y = np.asarray(y_rows, dtype=float)

    est_numpy, est_numpyro = _run_simple_hcm_graph("gaussian_mixture", y, a)
    assert np.isfinite(est_numpy)
    assert np.isfinite(est_numpyro)
    assert est_numpy > 0.0
    assert est_numpyro > 0.0
    assert abs(est_numpy - est_numpyro) < 0.35
