"""Tests for ConditionalDensityEstimator family ``categorical``."""

from __future__ import annotations

import numpy as np
import pytest

from hierarchicalcausalmodels.estimation.causal_estimators import ConditionalDensityEstimator


def test_categorical_marginal_prob_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.choice([1.0, 2.0, 3.0, 4.0], size=400, p=[0.1, 0.2, 0.3, 0.4])
    est = ConditionalDensityEstimator(family="categorical")
    est.fit(y, None)
    s = 0.0
    for c in (1.0, 2.0, 3.0, 4.0):
        s += est.prob(c, None)
    assert s == pytest.approx(1.0, abs=1e-6)


def test_categorical_conditional_vs_marginal() -> None:
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=(n, 1))
    logits = np.column_stack([np.zeros(n), x.ravel(), -x.ravel()])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice([10.0, 20.0, 30.0], p=p[i]) for i in range(n)])
    est = ConditionalDensityEstimator(family="categorical", regularization=10.0)
    est.fit(y, x)
    p10 = est.prob(10.0, np.array([0.0]))
    assert 0.0 < p10 < 1.0
    ex = est.expectation(np.array([0.0]))
    assert 10.0 <= ex <= 30.0


def test_categorical_alias_multinomial() -> None:
    est = ConditionalDensityEstimator(family="multinomial")
    assert est.family == "categorical"
