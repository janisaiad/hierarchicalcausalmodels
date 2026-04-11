"""
we verify that estimate_causal_effect's internal monte carlo (n_mc_samples) stabilises the
plug-in ate for fixed observed data — the same ladder of n_mc should give converging ate_hat.

we do **not** assert |ate_hat - external truth| -> 0 when only n_mc_samples grows: the
identificand is evaluated at the **empirical** distribution, so finite-sample bias vs a
population truth (curated dgp or large structural mc) remains unless n_units / n_sub grows.

run: uv run pytest scripts/tests/test_gallery_ate_mc_convergence.py -m slow --tb=short

we keep runtimes modest (target a few minutes total): a smaller MC ladder, a handful of
gallery cases, and ``estimator_backend="torch"`` so the batched MC path can use CUDA when
available (vectorised logits / conditional means on the full MC batch at once).
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EX = _ROOT / "examples" / "new"
sys.path.insert(0, str(_EX))

import hierarchicalcausalmodels.do_calculus as dc_pkg  # noqa: E402
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, build_cgm_for_case  # noqa: E402
from hierarchicalcausalmodels.estimation import estimate_causal_effect  # noqa: E402
from hierarchicalcausalmodels.models import HSCMParametric  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _noop(_d=None):
    return None


def simulate_binary_hscm(hscm, n_units, n_sub, rng):
    G = nx.DiGraph()
    G.add_nodes_from(hscm.nodes)
    G.add_edges_from(hscm.edges)
    order = list(nx.topological_sort(G))
    stor = {}
    for n in order:
        parents = [p for p, c in hscm.edges if c == n]
        is_sub = n in hscm.subunit_nodes
        if is_sub:
            if not parents:
                p = rng.uniform(0.25, 0.75, size=(n_units, n_sub))
                arr = rng.binomial(1, p).astype(float)
            else:
                acc = np.zeros((n_units, n_sub), dtype=float)
                for p in parents:
                    if p in hscm.subunit_nodes:
                        acc += 0.75 * stor[p]
                    else:
                        acc += 0.75 * stor[p][:, None]
                prob = np.clip(sigmoid(acc - 0.2), 0.02, 0.98)
                arr = rng.binomial(1, prob).astype(float)
        else:
            if not parents:
                p = rng.uniform(0.25, 0.75, size=n_units)
                arr = rng.binomial(1, p).astype(float)
            else:
                acc = np.zeros(n_units, dtype=float)
                for p in parents:
                    if p in hscm.subunit_nodes:
                        acc += 0.75 * stor[p].mean(axis=1)
                    else:
                        acc += 0.75 * stor[p]
                prob = np.clip(sigmoid(acc - 0.2), 0.02, 0.98)
                arr = rng.binomial(1, prob).astype(float)
        stor[n] = arr
    return {n.lstrip("_"): stor[n] for n in stor}


def _bern_families(data_dict):
    return {k: "bernoulli" for k in data_dict}


def _mc_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _ate_hat(res, data, x_node, n_mc, seed0, seed1):
    fam = _bern_families(data)
    dev = _mc_torch_device()
    e1 = estimate_causal_effect(
        res,
        data=data,
        intervention={x_node: 1.0},
        distribution_families=fam,
        n_mc_samples=n_mc,
        random_seed=seed0,
        estimator_backend="torch",
        torch_kwargs={"device": dev},
    )
    e0 = estimate_causal_effect(
        res,
        data=data,
        intervention={x_node: 0.0},
        distribution_families=fam,
        n_mc_samples=n_mc,
        random_seed=seed1,
        estimator_backend="torch",
        torch_kwargs={"device": dev},
    )
    return float(e1 - e0)


_MC_LADDER_LO = 1000
_MC_LADDER_HI = 3200
_MC_LADDER_TOL = 1.2e-2

_GALLERY_CASES_FOR_MC_LADDER = (
    "confounder_aug",
    "instrument_mar",
    "ID_ex3_collapse",
    "ID_ex2_aug",
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    [c for c in COLLAPSED_DO_CALCULUS_CASES if c[0] in _GALLERY_CASES_FOR_MC_LADDER],
    ids=lambda c: c[0],
)
def test_internal_mc_stabilises_ate_for_identifiable_gallery_cases(case):
    if not dc_pkg.PYAGNUM_AVAILABLE:
        pytest.skip("pyagrum not installed")
    expected_id = case[10]
    if not expected_id:
        pytest.skip("non-identifiable case")
    cgm, _u, y_node, x_node, _exp = build_cgm_for_case(case)
    unobs_paper = set(case[9]) & set(cgm.dag.nodes)
    res = dc_pkg.identify_effect(cgm, Y=y_node, X=x_node, unobserved=unobs_paper)
    if not res.identifiable:
        pytest.skip("identify_effect not identifiable for paper latents")
    nu, ns = 55, 18
    hscm = HSCMParametric(
        nodes=set(case[1]),
        edges=set(case[2]),
        unit_nodes=set(case[3]),
        subunit_nodes=set(case[4]),
        sizes=[ns] * nu,
        node_functions={n: _noop for n in case[1]},
        data={},
    )
    rng = np.random.default_rng(424242 + hash(case[0]) % 10_000)
    data_obs = simulate_binary_hscm(hscm, nu, ns, rng)
    n_lo, n_hi = _MC_LADDER_LO, _MC_LADDER_HI
    ate_lo = _ate_hat(res, data_obs, x_node, n_lo, 0, 1)
    ate_hi = _ate_hat(res, data_obs, x_node, n_hi, 0, 1)
    diff = abs(ate_lo - ate_hi)
    assert np.isfinite(diff), "we expect finite ate difference across mc ladders"
    assert diff < _MC_LADDER_TOL, (
        "we expect internal mc noise < {} between {} and {} draws ({})".format(
            _MC_LADDER_TOL, n_lo, n_hi, case[0]
        )
    )


@pytest.mark.slow
def test_same_seed_reproduces_ate_confounder_aug():
    if not dc_pkg.PYAGNUM_AVAILABLE:
        pytest.skip("pyagrum not installed")
    case = next(c for c in COLLAPSED_DO_CALCULUS_CASES if c[0] == "confounder_aug")
    cgm, _u, y_node, x_node, _e = build_cgm_for_case(case)
    unobs_paper = set(case[9]) & set(cgm.dag.nodes)
    res = dc_pkg.identify_effect(cgm, Y=y_node, X=x_node, unobserved=unobs_paper)
    assert res.identifiable
    nu, ns = 40, 12
    hscm = HSCMParametric(
        nodes=set(case[1]),
        edges=set(case[2]),
        unit_nodes=set(case[3]),
        subunit_nodes=set(case[4]),
        sizes=[ns] * nu,
        node_functions={n: _noop for n in case[1]},
        data={},
    )
    rng = np.random.default_rng(7)
    data_obs = simulate_binary_hscm(hscm, nu, ns, rng)
    n_mc = 2500
    a1 = _ate_hat(res, data_obs, x_node, n_mc, 0, 1)
    a2 = _ate_hat(res, data_obs, x_node, n_mc, 0, 1)
    assert np.isfinite(a1) and np.isfinite(a2)
    assert abs(a1 - a2) < 1e-9
