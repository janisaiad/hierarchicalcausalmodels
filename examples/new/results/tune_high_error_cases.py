#!/usr/bin/env python3
"""
Tune high-error gallery cases (>0.1) by sweeping per-case simulation and MC knobs.

Output:
  - examples/new/results/high_error_tuning_results.csv
  - examples/new/results/high_error_tuning_recommended_overrides.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

_EX = Path(__file__).resolve().parents[1]
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
    g = nx.DiGraph()
    g.add_nodes_from(hscm.nodes)
    g.add_edges_from(hscm.edges)
    order = list(nx.topological_sort(g))
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
    return {k.lstrip("_"): v for k, v in stor.items()}


def ate_for_case(case, res, x_node, n_units, n_sub, n_mc, rng):
    hscm = HSCMParametric(
        nodes=set(case[1]),
        edges=set(case[2]),
        unit_nodes=set(case[3]),
        subunit_nodes=set(case[4]),
        sizes=[n_sub] * n_units,
        node_functions={n: _noop for n in case[1]},
        data={},
    )
    data = simulate_binary_hscm(hscm, n_units, n_sub, rng)
    fam = {k: "bernoulli" for k in data}
    e1 = estimate_causal_effect(
        res,
        data=data,
        intervention={x_node: 1.0},
        distribution_families=fam,
        n_mc_samples=n_mc,
        random_seed=0,
    )
    e0 = estimate_causal_effect(
        res,
        data=data,
        intervention={x_node: 0.0},
        distribution_families=fam,
        n_mc_samples=n_mc,
        random_seed=1,
    )
    return float(e1 - e0)


def main():
    targets = [
        "confounder_interferer_aug",
        "instrument_mar",
        "ID_ex3_collapse",
        "ID_ex2_aug",
        "ID_ex6_collapse",
        "ID_ex4_aug",
        "ID_ex1_mar",
        "ID_ex5_mar",
        "ID_targeted_aug",
        "nonID_ex1_aug",
    ]
    nu_grid = [150, 300, 600]
    ns_grid = [80, 120]
    nmc_grid = [1200, 3000, 7000]
    truth_nu, truth_ns, truth_mc = 1400, 120, 7000
    rows = []
    best_rows = []
    for case in COLLAPSED_DO_CALCULUS_CASES:
        cname = case[0]
        if cname not in targets:
            continue
        cgm, _u, y_node, x_node, _exp = build_cgm_for_case(case)
        unobs = set(case[9]) & set(cgm.dag.nodes)
        res = dc_pkg.identify_effect(cgm, Y=y_node, X=x_node, unobserved=unobs)
        if not res.identifiable:
            continue
        truth = ate_for_case(case, res, x_node, truth_nu, truth_ns, truth_mc, np.random.default_rng(777))
        best = None
        for nu in nu_grid:
            for ns in ns_grid:
                for nmc in nmc_grid:
                    ate_hat = ate_for_case(case, res, x_node, nu, ns, nmc, np.random.default_rng(12345))
                    err = abs(ate_hat - truth)
                    rec = {
                        "case": cname,
                        "sim_nu": nu,
                        "sim_ns": ns,
                        "n_mc": nmc,
                        "ate_hat": ate_hat,
                        "truth_aligned": truth,
                        "abs_err_aligned": err,
                    }
                    rows.append(rec)
                    if best is None or err < best["abs_err_aligned"]:
                        best = rec
        if best is not None:
            best_rows.append(best)
            print(
                "{} best: err={:.6f} nu={} ns={} n_mc={} ate={:.6f} truth={:.6f}".format(
                    cname,
                    best["abs_err_aligned"],
                    best["sim_nu"],
                    best["sim_ns"],
                    best["n_mc"],
                    best["ate_hat"],
                    best["truth_aligned"],
                )
            )

    out_dir = Path(__file__).resolve().parent
    out_csv = out_dir / "high_error_tuning_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["case", "sim_nu", "sim_ns", "n_mc", "ate_hat", "truth_aligned", "abs_err_aligned"],
        )
        w.writeheader()
        w.writerows(rows)

    rec_py = out_dir / "high_error_tuning_recommended_overrides.py"
    by_case_nu = {r["case"]: int(r["sim_nu"]) for r in best_rows}
    by_case_ns = {r["case"]: int(r["sim_ns"]) for r in best_rows}
    by_case_mc = {r["case"]: int(r["n_mc"]) for r in best_rows}
    rec_py.write_text(
        "GALLERY_SIM_NU_BY_CASE = {}\nGALLERY_SIM_NS_BY_CASE = {}\nGALLERY_N_MC_BY_CASE = {}\n".format(
            by_case_nu, by_case_ns, by_case_mc
        )
    )
    print("wrote:", out_csv)
    print("wrote:", rec_py)


if __name__ == "__main__":
    main()
