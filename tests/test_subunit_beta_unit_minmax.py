"""``SubunitParamEstimator`` family ``beta_unit_minmax`` (per-unit min–max then Beta MoM)."""

from __future__ import annotations

import numpy as np

from hierarchicalcausalmodels.estimation.causal_estimators import (
    ConditionalDensityEstimator,
    SubunitParamEstimator,
    SUBUNIT_ONLY_FAMILIES,
)


def test_beta_unit_minmax_endpoints_map_to_unit_interval_params() -> None:
    est = SubunitParamEstimator(family="beta_unit_minmax")
    q = est.fit_unit(np.array([0.0, 10.0, 5.0]))
    assert q.shape == (2,)
    assert np.all(q > 0)


def test_beta_unit_minmax_matrix_shape() -> None:
    rng = np.random.default_rng(2)
    Y = rng.uniform(20.0, 80.0, size=(4, 12))
    est = SubunitParamEstimator(family="beta_minmax_unit")
    Q = est.fit(Y)
    assert Q.shape == (4, 2)


def test_conditional_density_rejects_subunit_only_family() -> None:
    try:
        ConditionalDensityEstimator(family="beta_unit_minmax")
    except ValueError as e:
        assert "SubunitParamEstimator" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_subunit_only_families_frozen() -> None:
    assert "beta_unit_minmax" in SUBUNIT_ONLY_FAMILIES
