#!/usr/bin/env python3
"""
run from repo root: uv run python examples/new/verify_augment_dags.py
we print DAG checks for collapse / augment / full pipeline (same cases as collapsed_cases).
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hierarchicalcausalmodels.do_calculus as dc  # noqa: E402
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, _make_hscm, build_cgm_for_case  # noqa: E402


def main() -> int:
    ok = True
    for case in COLLAPSED_DO_CALCULUS_CASES:
        name = case[0]
        cgm_full, _u, _y, _x, _e = build_cgm_for_case(case)
        dag_ok = nx.is_directed_acyclic_graph(cgm_full.dag)
        print("{} pipeline DAG: {}".format(name, "ok" if dag_ok else "CYCLE"))
        if not dag_ok:
            print("  edges:", list(cgm_full.dag.edges()))
            ok = False
        augment = case[5]
        if augment is not None:
            hscm = _make_hscm(case[1], case[2], case[3], case[4])
            c_after = dc.augment_collapsed_model(dc.collapse(hscm), augment[0], augment[1])
            a_ok = nx.is_directed_acyclic_graph(c_after.dag)
            print("  augment-only DAG: {}".format("ok" if a_ok else "CYCLE"))
            if not a_ok:
                print("  edges:", list(c_after.dag.edges()))
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
