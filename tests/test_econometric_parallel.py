import numpy as np

from hierarchicalcausalmodels.estimation import (
    IV2SLSModelSpec,
    OLSModelSpec,
    fit_2sls,
    fit_many_2sls,
    fit_many_ols,
    fit_reduced_form,
)


def test_fit_many_ols_parallel_matches_sequential():
    rng = np.random.default_rng(101)
    n = 300
    treatment = rng.binomial(1, 0.4, size=n).astype(float)
    aide = rng.binomial(1, 0.3, size=n).astype(float)
    free_lunch = rng.binomial(1, 0.5, size=n).astype(float)
    school = rng.integers(0, 8, size=n)
    y = 50.0 + 5.0 * treatment + 1.5 * aide - 3.0 * free_lunch + rng.normal(0.0, 2.0, size=n)

    spec_1 = OLSModelSpec(
        outcome=y,
        regressors=np.column_stack([treatment, aide]),
        regressor_names=("small", "aide"),
        controls=free_lunch,
        control_names=("free_lunch",),
        fixed_effects=school,
        fixed_effect_name="school",
        label="with_fe",
    )
    spec_2 = OLSModelSpec(
        outcome=y,
        regressors=np.column_stack([treatment, aide]),
        regressor_names=("small", "aide"),
        controls=free_lunch,
        control_names=("free_lunch",),
        label="plain",
    )

    seq_results = fit_many_ols([spec_1, spec_2], n_jobs=1)
    par_results = fit_many_ols(
        [spec_1, spec_2],
        n_jobs=2,
        parallel_backend="threads",
    )

    assert len(seq_results) == len(par_results) == 2
    for seq, par in zip(seq_results, par_results, strict=False):
        assert seq.coefficient_names == par.coefficient_names
        assert np.allclose(seq.coefficients, par.coefficients)
        assert np.allclose(seq.std_errors, par.std_errors)

    assert seq_results[0].coefficient_dict()["small"] > 4.0


def test_2sls_and_reduced_form_parallel_batch():
    rng = np.random.default_rng(202)
    n = 1500
    z_small = rng.binomial(1, 0.35, size=n).astype(float)
    w_free_lunch = rng.binomial(1, 0.45, size=n).astype(float)
    school = rng.integers(0, 10, size=n)

    latent = rng.normal(size=n)
    first_stage_noise = 0.6 * latent + rng.normal(scale=0.6, size=n)
    outcome_noise = 0.7 * latent + rng.normal(scale=0.6, size=n)

    actual_class_size = 22.0 - 6.5 * z_small + 1.5 * w_free_lunch + first_stage_noise
    outcome = 45.0 - 0.9 * actual_class_size - 4.0 * w_free_lunch + outcome_noise

    reduced_form = fit_reduced_form(
        outcome=outcome,
        instruments=z_small,
        instrument_names=("assigned_small",),
        controls=w_free_lunch,
        control_names=("free_lunch",),
        fixed_effects=school,
        fixed_effect_name="school",
    )
    iv_single = fit_2sls(
        outcome=outcome,
        endogenous=actual_class_size,
        instruments=z_small,
        endogenous_names=("actual_class_size",),
        instrument_names=("assigned_small",),
        controls=w_free_lunch,
        control_names=("free_lunch",),
        fixed_effects=school,
        fixed_effect_name="school",
    )

    specs = [
        IV2SLSModelSpec(
            outcome=outcome,
            endogenous=actual_class_size,
            instruments=z_small,
            endogenous_names=("actual_class_size",),
            instrument_names=("assigned_small",),
            controls=w_free_lunch,
            control_names=("free_lunch",),
            fixed_effects=school,
            fixed_effect_name="school",
            label="iv_star_like",
        )
    ]
    iv_batch = fit_many_2sls(specs, n_jobs=2, parallel_backend="threads")[0]

    assert reduced_form.coefficient_dict()["assigned_small"] > 3.0
    assert iv_single.coefficient_dict()["actual_class_size"] < -0.5
    assert iv_batch.coefficient_names == iv_single.coefficient_names
    assert np.allclose(iv_batch.coefficients, iv_single.coefficients)
    assert np.allclose(iv_batch.std_errors, iv_single.std_errors)
