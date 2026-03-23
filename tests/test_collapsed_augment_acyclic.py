"""
we assert every collapsed -> augment -> marginalize pipeline yields a DAG (no directed cycles).
we use do_calculus from examples/new (sys.path) so tests match notebook demos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

_EXAMPLES_NEW = Path(__file__).resolve().parents[1] / "examples" / "new"
if str(_EXAMPLES_NEW) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_NEW))

import do_calculus as dc  # noqa: E402
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, _make_hscm, build_cgm_for_case  # noqa: E402


def _assert_dag(name: str, cgm) -> None:
    g = cgm.dag
    assert nx.is_directed_acyclic_graph(g), "expected DAG for case {!r}, got cycle in edges: {!r}".format(
        name,
        list(g.edges()),
    )


@pytest.mark.parametrize("case", COLLAPSED_DO_CALCULUS_CASES, ids=lambda c: c[0])
def test_pipeline_collapsed_augment_marginalize_is_dag(case):
    """full build_cgm_for_case graph must stay acyclic."""
    cgm, _u, _y, _x, _exp = build_cgm_for_case(dc, case)
    _assert_dag(case[0], cgm)


@pytest.mark.parametrize("case", COLLAPSED_DO_CALCULUS_CASES, ids=lambda c: c[0])
def test_after_collapse_only_is_dag(case):
    """collapse(hscm) is a DAG for every case."""
    (
        name,
        nodes,
        edges,
        unit_nodes,
        subunit_nodes,
        _aug,
        _mar,
        _y,
        _x,
        _u,
        _exp,
    ) = case
    hscm = _make_hscm(nodes, edges, unit_nodes, subunit_nodes)
    cgm = dc.collapse(hscm)
    _assert_dag(name + " (collapse)", cgm)


@pytest.mark.parametrize("case", COLLAPSED_DO_CALCULUS_CASES, ids=lambda c: c[0])
def test_after_augment_when_present_is_dag(case):
    """if augment is defined, graph right after augment_collapsed_model is a DAG."""
    (
        name,
        nodes,
        edges,
        unit_nodes,
        subunit_nodes,
        augment,
        _mar,
        _y,
        _x,
        _u,
        _exp,
    ) = case
    if augment is None:
        pytest.skip("no augment step")
    hscm = _make_hscm(nodes, edges, unit_nodes, subunit_nodes)
    cgm = dc.collapse(hscm)
    q_hat, parents = augment
    cgm = dc.augment_collapsed_model(cgm, q_hat, parents)
    _assert_dag(name + " (after augment)", cgm)


def test_augment_adds_only_incoming_edges_paper_style():
    """sanity: new node has only incoming edges from q_hat_parents (no redirect)."""
    from collapsed_cases import _make_hscm

    case = next(c for c in COLLAPSED_DO_CALCULUS_CASES if c[0] == "confounder_interferer_aug")
    (
        _name,
        nodes,
        edges,
        unit_nodes,
        subunit_nodes,
        augment,
        _mar,
        _y,
        _x,
        _u,
        _exp,
    ) = case
    hscm = _make_hscm(nodes, edges, unit_nodes, subunit_nodes)
    c0 = dc.collapse(hscm)
    q_hat, parents = augment
    c1 = dc.augment_collapsed_model(c0, q_hat, parents)
    succ = set(c1.dag.successors(q_hat))
    assert succ == set(), "we expect q_hat to have no outgoing edges in paper-style augment, got {!r}".format(succ)
    preds = set(c1.dag.predecessors(q_hat))
    assert preds == parents
