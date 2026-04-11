"""
we smoke-test the same collapsed_cases loop as hcm_framework_test §4 (DAG + paper ID + optional est).
we avoid importing hcm_framework_test.py (jupytext / do_calculus side effects on import).
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
from hierarchicalcausalmodels.models import HSCMParametric  # noqa: E402


def _noop(_d=None):
    return None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


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


@pytest.mark.parametrize("case", COLLAPSED_DO_CALCULUS_CASES, ids=lambda c: c[0])
def test_collapsed_case_is_dag(case):
    cgm, *_rest = build_cgm_for_case(case)
    assert nx.is_directed_acyclic_graph(cgm.dag)


@pytest.mark.parametrize("case", COLLAPSED_DO_CALCULUS_CASES, ids=lambda c: c[0])
def test_paper_latent_identify_matches_table(case):
    if not dc_pkg.PYAGNUM_AVAILABLE:
        pytest.skip("pyagrum not installed")
    cgm, _u, y_n, x_n, expected_id = build_cgm_for_case(case)
    unobs_paper = set(case[9]) & set(cgm.dag.nodes)
    res = dc_pkg.identify_effect(cgm, Y=y_n, X=x_n, unobserved=unobs_paper)
    assert res.identifiable == expected_id


def test_generic_sim_runs_for_non_curated_case():
    case = next(c for c in COLLAPSED_DO_CALCULUS_CASES if c[0] == "ID_ex3_collapse")
    nu, ns = 25, 8
    hscm = HSCMParametric(
        nodes=set(case[1]),
        edges=set(case[2]),
        unit_nodes=set(case[3]),
        subunit_nodes=set(case[4]),
        sizes=[ns] * nu,
        node_functions={n: _noop for n in case[1]},
        data={},
    )
    rng = np.random.default_rng(0)
    d = simulate_binary_hscm(hscm, nu, ns, rng)
    assert "A" in d and d["A"].shape == (nu, ns)


@pytest.mark.slow
def test_estimate_runs_when_paper_identifiable_confounder_aug():
    if not dc_pkg.PYAGNUM_AVAILABLE:
        pytest.skip("pyagrum not installed")
    from hierarchicalcausalmodels.estimation import estimate_causal_effect  # noqa: E402

    case = next(c for c in COLLAPSED_DO_CALCULUS_CASES if c[0] == "confounder_aug")
    cgm, _u, y_n, x_n, _e = build_cgm_for_case(case)
    unobs_paper = set(case[9]) & set(cgm.dag.nodes)
    res = dc_pkg.identify_effect(cgm, Y=y_n, X=x_n, unobserved=unobs_paper)
    assert res.identifiable
    nu, ns = 40, 12
    rng = np.random.default_rng(1)
    hscm = HSCMParametric(
        nodes=set(case[1]),
        edges=set(case[2]),
        unit_nodes=set(case[3]),
        subunit_nodes=set(case[4]),
        sizes=[ns] * nu,
        node_functions={n: _noop for n in case[1]},
        data={},
    )
    data = simulate_binary_hscm(hscm, nu, ns, rng)
    fam = {k: "bernoulli" for k in data}
    try:
        import torch as _torch

        _dev = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        _dev = "cpu"
    _kw = dict(
        distribution_families=fam,
        n_mc_samples=800,
        estimator_backend="torch",
        torch_kwargs={"device": _dev},
    )
    e1 = estimate_causal_effect(res, data=data, intervention={x_n: 1.0}, **_kw)
    e0 = estimate_causal_effect(res, data=data, intervention={x_n: 0.0}, **_kw)
    assert np.isfinite(e1) and np.isfinite(e0)
