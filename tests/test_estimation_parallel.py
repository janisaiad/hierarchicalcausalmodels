import numpy as np

from hierarchicalcausalmodels.estimation import (
    aggregate_per_unit_outputs,
    estimate_causal_effect,
    fit_per_unit_estimators,
    fit_regressors_per_unit,
)


class _FakeResult:
    def __init__(self, formula_latex: str):
        self.identifiable = True
        self.formula_latex = formula_latex


def _fit_unit_stats(data: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    a, y = data
    mask_0 = a == 0
    mask_1 = a == 1
    n0 = max(int(mask_0.sum()), 1)
    n1 = max(int(mask_1.sum()), 1)
    mu_0 = (float(y[mask_0].sum()) + 1.0) / (n0 + 2.0)
    mu_1 = (float(y[mask_1].sum()) + 1.0) / (n1 + 2.0)
    return mu_0, mu_1


def test_per_unit_helpers_are_exported_and_parallel_safe():
    rng = np.random.default_rng(7)
    a = rng.binomial(1, 0.5, size=(8, 24))
    y = rng.binomial(1, 0.2 + 0.5 * a, size=(8, 24))
    unit_data = [(a[i], y[i]) for i in range(a.shape[0])]

    seq_fits = fit_per_unit_estimators(unit_data, _fit_unit_stats, n_jobs=1)
    par_fits = fit_per_unit_estimators(
        unit_data,
        _fit_unit_stats,
        n_jobs=2,
        parallel_backend="threads",
    )

    assert seq_fits == par_fits

    seq_ate = aggregate_per_unit_outputs(
        seq_fits,
        "mean",
        extract_fn=lambda value: value[1] - value[0],
    )
    par_ate = aggregate_per_unit_outputs(
        par_fits,
        "mean",
        extract_fn=lambda value: value[1] - value[0],
    )

    assert np.isclose(seq_ate, par_ate)


def test_fit_regressors_per_unit_parallel_matches_sequential():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(11)
    a = rng.binomial(1, 0.5, size=(6, 20)).astype(float)
    y = (0.1 + 0.7 * a + rng.normal(scale=0.05, size=(6, 20))).astype(float)

    regs_seq = fit_regressors_per_unit(a, y, LinearRegression, n_jobs=1)
    regs_par = fit_regressors_per_unit(
        a,
        y,
        LinearRegression,
        n_jobs=2,
        parallel_backend="threads",
    )

    preds_seq = np.array([float(reg.predict([[1.0]])[0]) for reg in regs_seq])
    preds_par = np.array([float(reg.predict([[1.0]])[0]) for reg in regs_par])

    assert np.allclose(preds_seq, preds_par)


def test_estimate_causal_effect_parallel_matches_sequential():
    rng = np.random.default_rng(21)
    n_units = 12
    n_subunits = 40

    u = rng.beta(2.0, 2.0, size=n_units)
    a = np.array(
        [rng.binomial(1, u_i, size=n_subunits) for u_i in u],
        dtype=float,
    )
    logits = 1.1 * a + u[:, None] - 0.3
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = np.array(
        [rng.binomial(1, probs_i) for probs_i in probs],
        dtype=float,
    )

    result = _FakeResult(
        r"\sum_{Qy_a}{P\left(Qy_a\right) \cdot P\left(Qy\mid Qa,Qy_a\right)}"
    )
    data = {"A": a, "Y": y}
    kwargs = {
        "distribution_families": {"A": "bernoulli", "Y": "bernoulli"},
        "intervention": {"Q^a": 1.0},
        "n_mc_samples": 512,
        "random_seed": 123,
    }

    est_seq = estimate_causal_effect(result, data=data, n_jobs=1, **kwargs)
    est_par = estimate_causal_effect(
        result,
        data=data,
        n_jobs=2,
        parallel_backend="threads",
        **kwargs,
    )

    assert np.isfinite(est_seq)
    assert np.isfinite(est_par)
    assert abs(est_seq - est_par) < 0.05
