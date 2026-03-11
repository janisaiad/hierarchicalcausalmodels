"""
Software tests for ATE recovery: confounder model and estimator run correctly.

Demos and multi-seed runs live in examples/new/ate_recovery_demo (notebook).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_NEW = ROOT / "examples" / "new"
if str(EXAMPLES_NEW) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_NEW))

from confounder_model import ConfounderModel, default_confounder_params


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_confounder_true_ate_deterministic(rng):
    """True ATE is deterministic and in [0, 1] for default params."""
    model = ConfounderModel(**default_confounder_params(omega=0.2))
    ate = model.true_ate()
    assert 0 <= ate <= 1
    assert 0.28 <= ate <= 0.38


def test_confounder_estimate_ate_returns_float(rng):
    """estimate_ate runs and returns a float in [-1, 1]."""
    model = ConfounderModel(**default_confounder_params(omega=0.2))
    A, Y = model.sample(100, 100, rng)
    est = model.estimate_ate(A, Y)
    assert isinstance(est, float)
    assert -1 <= est <= 1


def test_confounder_ate_recovery_single_run(rng):
    """Single run: estimated ATE is close to true ATE (software sanity)."""
    model = ConfounderModel(**default_confounder_params(omega=0.2))
    A, Y = model.sample(600, 600, rng)
    assert abs(model.estimate_ate(A, Y) - model.true_ate()) < 0.08
    