#!/usr/bin/env python3
"""
we smoke-test each collapsed_cases row with small plates and low n_mc_samples.
run from repo root: uv run python examples/new/gallery_quick_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX))

import networkx as nx
import numpy as np

import do_calculus as dc_pkg
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, build_cgm_for_case
from estimation import estimate_causal_effect
from hierarchicalcausalmodels.models import HSCMParametric


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


# we keep tiny defaults so this finishes in seconds (not minutes).
QUICK_NU = 32
QUICK_NS = 8
QUICK_MC = 48


def main() -> int:
    if not dc_pkg.PYAGNUM_AVAILABLE:
        print("pyagrum not installed; skip.")
        return 1
    rng = np.random.default_rng(42)
    failed: list[tuple[str, str]] = []
    print(
        "quick smoke: {} units × {} subunits, n_mc_samples={}\n".format(
            QUICK_NU, QUICK_NS, QUICK_MC
        )
    )
    for case in COLLAPSED_DO_CALCULUS_CASES:
        name = case[0]
        y_n = case[7]
        x_n = case[8]
        cgm, _u, _y, _x, _exp = build_cgm_for_case(dc_pkg, case)
        unobs_paper = set(case[9]) & set(cgm.dag.nodes)
        res = dc_pkg.identify_effect(cgm, Y=y_n, X=x_n, unobserved=unobs_paper)
        if not res.identifiable:
            print("[skip] {:28s} not identifiable".format(name))
            continue
        hscm = HSCMParametric(
            nodes=set(case[1]),
            edges=set(case[2]),
            unit_nodes=set(case[3]),
            subunit_nodes=set(case[4]),
            sizes=[QUICK_NS] * QUICK_NU,
            node_functions={n: _noop for n in case[1]},
            data={},
        )
        data = simulate_binary_hscm(hscm, QUICK_NU, QUICK_NS, rng)
        fam = {k: "bernoulli" for k in data}
        try:
            e1 = estimate_causal_effect(
                res,
                data=data,
                intervention={x_n: 1.0},
                distribution_families=fam,
                n_mc_samples=QUICK_MC,
                random_seed=0,
            )
            e0 = estimate_causal_effect(
                res,
                data=data,
                intervention={x_n: 0.0},
                distribution_families=fam,
                n_mc_samples=QUICK_MC,
                random_seed=1,
            )
            ate = float(e1 - e0)
            if not (np.isfinite(e1) and np.isfinite(e0) and np.isfinite(ate)):
                raise ValueError("non-finite e1={!r} e0={!r}".format(e1, e0))
        except Exception as ex:
            failed.append((name, str(ex)))
            print("[FAIL] {:28s} {}".format(name, str(ex)[:120]))
            continue
        print("[ok]   {:28s} ATE_hat={:.6f}".format(name, ate))
    if failed:
        print("\nfailed {} case(s):".format(len(failed)))
        for n, msg in failed:
            print("  - {}: {}".format(n, msg[:200]))
        return 1
    print("\nall identifiable cases passed (finite ATE, no NaN).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
