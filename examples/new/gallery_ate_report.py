#!/usr/bin/env python3
"""
we print one row per collapsed_cases graph: identify (paper latents), estimate ate_hat, compare to
the **ad hoc** reference truth used in hcm_framework_test §4 (curated dgp mc vs structural do mc).

**what can go ~0**
  - **internal mc noise** in estimate_causal_effect: raise n_mc_samples (and keep seeds fixed);
    the plug-in value stabilises.

**what does not automatically go ~0**
  - **|ate_hat - truth_ate|** at **fixed** n_units / n_sub: the estimator uses the **empirical**
    distribution; truth in the notebook is **population** (curated mc over u) or **large** structural
    mc — mismatch is normal unless you also grow data (and align definitions).

run from repo root:
  uv run python examples/new/gallery_ate_report.py
  uv run python examples/new/gallery_ate_report.py --full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX))

import networkx as nx
import numpy as np
import pandas as pd

import do_calculus as dc_pkg
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, build_cgm_for_case
from estimation import estimate_causal_effect
from gallery_estimands import get_estimand
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


def _bern_families(data_dict):
    return {k: "bernoulli" for k in data_dict}


def _estimate_ate_once(res, data_obs, x_node, n_mc, seed1=0, seed0=1):
    fam = _bern_families(data_obs)
    e1 = estimate_causal_effect(
        res,
        data=data_obs,
        intervention={x_node: 1.0},
        distribution_families=fam,
        random_seed=seed1,
        n_mc_samples=n_mc,
    )
    e0 = estimate_causal_effect(
        res,
        data=data_obs,
        intervention={x_node: 0.0},
        distribution_families=fam,
        random_seed=seed0,
        n_mc_samples=n_mc,
    )
    return float(e1 - e0)


def estimate_ate_adaptive_mc(
    res,
    data_obs,
    x_node,
    n_mc_start=1500,
    n_mc_max=24000,
    tol=2e-3,
):
    """
    Increase internal MC draws until successive ATE estimates stabilize.
    Returns (ate_hat, converged, used_n_mc, delta_last).
    """
    n_cur = int(n_mc_start)
    prev = None
    while True:
        cur = _estimate_ate_once(res, data_obs, x_node, n_cur, seed1=0, seed0=1)
        if prev is not None:
            d = abs(cur - prev)
            if d <= tol:
                return cur, True, n_cur, d
        if n_cur >= n_mc_max:
            d = float("nan") if prev is None else abs(cur - prev)
            return cur, False, n_cur, d
        prev = cur
        n_cur = min(n_cur * 2, n_mc_max)


def aligned_formula_truth_ate(
    case,
    res,
    x_node,
    n_units_large,
    n_sub_large,
    n_mc_start,
    n_mc_max,
    tol,
    rng,
):
    """
    Build a large synthetic dataset from the same structural simulator, then evaluate
    the same identified formula adaptively. This is a per-case aligned reference.
    """
    hscm_large = HSCMParametric(
        nodes=set(case[1]),
        edges=set(case[2]),
        unit_nodes=set(case[3]),
        subunit_nodes=set(case[4]),
        sizes=[n_sub_large] * n_units_large,
        node_functions={n: _noop for n in case[1]},
        data={},
    )
    data_large = simulate_binary_hscm(hscm_large, n_units_large, n_sub_large, rng)
    return estimate_ate_adaptive_mc(
        res=res,
        data_obs=data_large,
        x_node=x_node,
        n_mc_start=n_mc_start,
        n_mc_max=n_mc_max,
        tol=tol,
    )


def _gallery_x_for_case(case, override):
    cname = case[0]
    if override and cname in override:
        return override[cname]
    return case[8]


def _gallery_x_to_forced_subunits(x_node, subunit_nodes):
    s = (x_node or "").strip()
    if s == "Q^a":
        return ["A"] if "A" in subunit_nodes else []
    if s == "Q^w":
        return ["W"] if "W" in subunit_nodes else []
    if s == "Q^{a|x}":
        out = []
        if "A" in subunit_nodes:
            out.append("A")
        if "X" in subunit_nodes:
            out.append("X")
        return out
    if s == "Q^z":
        return ["Z"] if "Z" in subunit_nodes else []
    raise ValueError("unknown gallery x_node for structural do(): {!r}".format(x_node))


def _gallery_outcome_per_unit(stor, y_node, unit_nodes, subunit_nodes):
    if y_node == "Q^y":
        y_arr = stor["Y"]
        if y_arr.ndim == 2:
            return y_arr.mean(axis=1).astype(float)
        return y_arr.astype(float).ravel()
    if y_node in unit_nodes:
        return np.asarray(stor[y_node], dtype=float).ravel()
    if y_node in subunit_nodes:
        v = stor[y_node]
        if v.ndim == 2:
            return v.mean(axis=1).astype(float)
        return v.astype(float).ravel()
    raise ValueError("gallery outcome y_node={!r}".format(y_node))


def simulate_binary_hscm_do(nodes, edges, unit_nodes, subunit_nodes, n_units, n_sub, rng, forced_subunits):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    order = list(nx.topological_sort(g))
    stor = {}
    forced = {k: float(v) for k, v in forced_subunits.items()}
    for n in order:
        if n in forced:
            v = forced[n]
            if n in subunit_nodes:
                stor[n] = np.full((n_units, n_sub), v, dtype=float)
            else:
                stor[n] = np.full(n_units, v, dtype=float)
            continue
        parents = [p for p, c in edges if c == n]
        is_sub = n in subunit_nodes
        if is_sub:
            if not parents:
                p = rng.uniform(0.25, 0.75, size=(n_units, n_sub))
                arr = rng.binomial(1, p).astype(float)
            else:
                acc = np.zeros((n_units, n_sub), dtype=float)
                for p in parents:
                    if p in subunit_nodes:
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
                    if p in subunit_nodes:
                        acc += 0.75 * stor[p].mean(axis=1)
                    else:
                        acc += 0.75 * stor[p]
                prob = np.clip(sigmoid(acc - 0.2), 0.02, 0.98)
                arr = rng.binomial(1, prob).astype(float)
        stor[n] = arr
    return {k.lstrip("_"): v for k, v in stor.items()}


def mc_truth_ate_binary_plate(case, x_node, n_units, n_sub, rng):
    nodes, edges, unit_nodes, subunit_nodes = case[1], case[2], case[3], case[4]
    y_node = case[7]
    targets = _gallery_x_to_forced_subunits(x_node, subunit_nodes)
    if not targets:
        return float("nan")
    f1 = {t: 1.0 for t in targets}
    f0 = {t: 0.0 for t in targets}
    stor1 = simulate_binary_hscm_do(nodes, edges, unit_nodes, subunit_nodes, n_units, n_sub, rng, f1)
    stor0 = simulate_binary_hscm_do(nodes, edges, unit_nodes, subunit_nodes, n_units, n_sub, rng, f0)
    y1 = _gallery_outcome_per_unit(stor1, y_node, unit_nodes, subunit_nodes)
    y0 = _gallery_outcome_per_unit(stor0, y_node, unit_nodes, subunit_nodes)
    return float(y1.mean() - y0.mean())


def mc_truth_ate_binary_plate_adaptive(case, x_node, n_sub, n_start, n_max, tol, rng):
    """
    Increase structural truth MC sample size until successive truth estimates stabilize.
    Returns (truth_ate, converged, used_n_units, delta_last).
    """
    n_cur = int(n_start)
    prev = None
    while True:
        cur = mc_truth_ate_binary_plate(case, x_node, n_cur, n_sub, rng)
        if prev is not None:
            d = abs(cur - prev)
            if d <= tol:
                return cur, True, n_cur, d
        if n_cur >= n_max:
            d = float("nan") if prev is None else abs(cur - prev)
            return cur, False, n_cur, d
        prev = cur
        n_cur = min(n_cur * 2, n_max)


def curated_dgp_and_truth(nu: int, ns: int, rng_truth: np.random.Generator):
    """we match §1–3 dgp formulas in hcm_framework_test for the three curated rows (separate data rng per model)."""
    rng_conf = np.random.default_rng(42)
    U_conf = rng_conf.beta(2, 2, nu)
    A_conf = np.array([rng_conf.binomial(1, np.clip(U_conf[i], 0, 1), ns) for i in range(nu)], dtype=float)
    Y_conf = np.array(
        [
            rng_conf.binomial(1, np.clip(sigmoid(1.5 * A_conf[i] + U_conf[i] - 0.5), 0, 1))
            for i in range(nu)
        ],
        dtype=float,
    )
    U_mc = rng_truth.beta(2, 2, 200_000)
    true_ey1_conf = float(np.mean(sigmoid(1.5 * 1 + U_mc - 0.5)))
    true_ey0_conf = float(np.mean(sigmoid(1.5 * 0 + U_mc - 0.5)))
    true_ate_conf = true_ey1_conf - true_ey0_conf
    rng_ci = np.random.default_rng(42)
    U_ci = rng_ci.beta(2, 2, nu)
    A_ci = np.array([rng_ci.binomial(1, np.clip(U_ci[i], 0, 1), ns) for i in range(nu)], dtype=float)
    Q_a_ci_obs = A_ci.mean(axis=1)
    Z_ci = rng_ci.binomial(1, np.clip(sigmoid(2 * Q_a_ci_obs - 1), 0, 1)).astype(float)
    Y_ci = np.array(
        [
            rng_ci.binomial(1, np.clip(sigmoid(1.5 * A_ci[i] + Z_ci[i] + 0.3 * U_ci[i] - 0.5), 0, 1))
            for i in range(nu)
        ],
        dtype=float,
    )
    U_mc_ci = rng_truth.beta(2, 2, 200_000)

    def true_ey_ci(a_val):
        p_z = sigmoid(2 * a_val - 1)
        ey_z1 = sigmoid(1.5 * a_val + 1 + 0.3 * U_mc_ci - 0.5)
        ey_z0 = sigmoid(1.5 * a_val + 0 + 0.3 * U_mc_ci - 0.5)
        return float(np.mean(p_z * ey_z1 + (1 - p_z) * ey_z0))

    true_ey1_ci = true_ey_ci(1)
    true_ey0_ci = true_ey_ci(0)
    true_ate_ci = true_ey1_ci - true_ey0_ci
    rng_inst = np.random.default_rng(42)
    U_inst = rng_inst.beta(2, 2, nu)
    Q_z_inst = rng_inst.beta(2, 2, nu)
    Z_inst = rng_inst.binomial(1, Q_z_inst[:, None] * np.ones(ns)).astype(float)
    pA_inst = np.clip(0.60 * Z_inst + 0.55 * U_inst[:, None] + 0.05, 0, 1)
    A_inst = rng_inst.binomial(1, pA_inst).astype(float)
    Q_a_inst = A_inst.mean(axis=1)
    Y_inst = rng_inst.binomial(1, np.clip(sigmoid(3 * Q_a_inst + 1.5 * U_inst - 1.5), 0, 1)).astype(float)
    U_mc_inst = rng_truth.beta(2, 2, 200_000)
    true_ey1_inst = float(np.mean(sigmoid(3 * 1 + 1.5 * U_mc_inst - 1.5)))
    true_ey0_inst = float(np.mean(sigmoid(3 * 0 + 1.5 * U_mc_inst - 1.5)))
    true_ate_inst = true_ey1_inst - true_ey0_inst
    return {
        "confounder_aug": {"data": {"A": A_conf, "Y": Y_conf}, "truth_ate": true_ate_conf},
        "confounder_interferer_aug": {"data": {"A": A_ci, "Y": Y_ci, "Z": Z_ci}, "truth_ate": true_ate_ci},
        "instrument_mar": {"data": {"Y": Y_inst, "A": A_inst, "Z": Z_inst}, "truth_ate": true_ate_inst},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="per-graph ate vs ad hoc truth (see hcm_framework_test §4)")
    p.add_argument(
        "--full",
        action="store_true",
        help="we use larger sim / truth / mc settings closer to the notebook §4 defaults",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=2e-3,
        help="absolute tolerance for adaptive convergence checks (default: 2e-3)",
    )
    p.add_argument("--sim-nu", type=int, default=None, help="override observed simulation units")
    p.add_argument("--sim-ns", type=int, default=None, help="override observed simulation subunits")
    p.add_argument("--align-nu", type=int, default=None, help="override aligned-truth units")
    p.add_argument("--align-ns", type=int, default=None, help="override aligned-truth subunits")
    p.add_argument("--est-mc-start", type=int, default=None, help="override estimator MC start")
    p.add_argument("--est-mc-max", type=int, default=None, help="override estimator MC max")
    p.add_argument(
        "--case",
        action="append",
        default=None,
        help="optional case name filter; pass multiple --case values",
    )
    p.add_argument(
        "--target-err",
        type=float,
        default=0.05,
        help="target for |err_aligned| to mark case as accurate",
    )
    p.add_argument(
        "--auto-tune",
        action="store_true",
        help="auto-increase sim/align/MC budgets per case until target error or max rounds",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="max rounds for --auto-tune",
    )
    p.add_argument(
        "--write-csv",
        type=str,
        default="",
        help="optional csv output path for final table",
    )
    args = p.parse_args()
    if not dc_pkg.PYAGNUM_AVAILABLE:
        print("pyagrum not installed.")
        return 1
    if args.full:
        sim_nu, sim_ns = 150, min(80, 100)
        truth_nu0, truth_nu_max, truth_ns = 3000, 12000, min(80, 100)
        est_n_mc0, est_n_mc_max = 3000, 24000
        align_nu, align_ns = 900, 120
    else:
        sim_nu, sim_ns = 80, 24
        truth_nu0, truth_nu_max, truth_ns = 1200, 4800, 24
        est_n_mc0, est_n_mc_max = 1500, 12000
        align_nu, align_ns = 300, 40
    if args.sim_nu is not None:
        sim_nu = int(args.sim_nu)
    if args.sim_ns is not None:
        sim_ns = int(args.sim_ns)
        truth_ns = int(args.sim_ns)
    if args.align_nu is not None:
        align_nu = int(args.align_nu)
    if args.align_ns is not None:
        align_ns = int(args.align_ns)
    if args.est_mc_start is not None:
        est_n_mc0 = int(args.est_mc_start)
    if args.est_mc_max is not None:
        est_n_mc_max = int(args.est_mc_max)
    rng_data = np.random.default_rng(12345)
    rng_truth_mc = np.random.default_rng(99991)
    rng_struct = np.random.default_rng(99991)
    curated = curated_dgp_and_truth(sim_nu, sim_ns, rng_truth_mc)
    rows = []
    case_iter = COLLAPSED_DO_CALCULUS_CASES
    if args.case:
        wanted = set(args.case)
        case_iter = [c for c in COLLAPSED_DO_CALCULUS_CASES if c[0] in wanted]
    for case in case_iter:
        cname = case[0]
        y_node = case[7]
        x_node = _gallery_x_for_case(case, None)
        expected_id = case[10]
        cgm, *_ = build_cgm_for_case(dc_pkg, case)
        unobs_paper = set(case[9]) & set(cgm.dag.nodes)
        res = dc_pkg.identify_effect(cgm, Y=y_node, X=x_node, unobserved=unobs_paper)
        got_paper = bool(res.identifiable)
        truth_conv = True
        truth_delta = 0.0
        truth_used_n = float("nan")
        align_truth = float("nan")
        align_conv = False
        align_delta = float("nan")
        align_used_mc = float("nan")
        align_err = float("nan")
        sim_nu_case = sim_nu
        sim_ns_case = sim_ns
        align_nu_case = align_nu
        align_ns_case = align_ns
        est_mc0_case = est_n_mc0
        est_mcmax_case = est_n_mc_max
        rounds_used = 0
        start_t = time.time()
        while True:
            rounds_used += 1
            if cname in curated:
                data_obs = curated_dgp_and_truth(sim_nu_case, sim_ns_case, rng_truth_mc)[cname]["data"]
                truth_ate = float(curated[cname]["truth_ate"])
                has_truth = True
            else:
                hscm_sim = HSCMParametric(
                    nodes=set(case[1]),
                    edges=set(case[2]),
                    unit_nodes=set(case[3]),
                    subunit_nodes=set(case[4]),
                    sizes=[sim_ns_case] * sim_nu_case,
                    node_functions={n: _noop for n in case[1]},
                    data={},
                )
                data_obs = simulate_binary_hscm(hscm_sim, sim_nu_case, sim_ns_case, rng_data)
                truth_ate, truth_conv, truth_used_n, truth_delta = mc_truth_ate_binary_plate_adaptive(
                    case=case,
                    x_node=x_node,
                    n_sub=truth_ns,
                    n_start=truth_nu0,
                    n_max=truth_nu_max,
                    tol=args.tol,
                    rng=rng_struct,
                )
                has_truth = np.isfinite(truth_ate)
            ate_hat = float("nan")
            abs_err = float("nan")
            est_ok = False
            est_conv = False
            est_delta = float("nan")
            est_used_mc = float("nan")
            est_msg = ""
            align_truth = float("nan")
            align_conv = False
            align_delta = float("nan")
            align_used_mc = float("nan")
            align_err = float("nan")
            if got_paper and res.identifiable:
                try:
                    ate_hat, est_conv, est_used_mc, est_delta = estimate_ate_adaptive_mc(
                        res=res,
                        data_obs=data_obs,
                        x_node=x_node,
                        n_mc_start=est_mc0_case,
                        n_mc_max=est_mcmax_case,
                        tol=args.tol,
                    )
                    est_ok = np.isfinite(ate_hat)
                    align_truth, align_conv, align_used_mc, align_delta = aligned_formula_truth_ate(
                        case=case,
                        res=res,
                        x_node=x_node,
                        n_units_large=align_nu_case,
                        n_sub_large=align_ns_case,
                        n_mc_start=est_mc0_case,
                        n_mc_max=est_mcmax_case,
                        tol=args.tol,
                        rng=rng_data,
                    )
                    if est_ok and np.isfinite(align_truth):
                        align_err = abs(ate_hat - align_truth)
                    if has_truth and est_ok:
                        abs_err = abs(ate_hat - truth_ate)
                except Exception as ex:
                    est_msg = str(ex)[:160]
            should_stop = True
            if args.auto_tune and est_ok and np.isfinite(align_err) and align_err > args.target_err and rounds_used < args.max_rounds:
                should_stop = False
                sim_nu_case = int(sim_nu_case * 1.7)
                sim_ns_case = int(max(sim_ns_case, min(120, sim_ns_case * 1.5)))
                align_nu_case = int(align_nu_case * 1.8)
                align_ns_case = int(max(align_ns_case, min(160, align_ns_case * 1.5)))
                est_mc0_case = int(est_mc0_case * 2)
                est_mcmax_case = int(est_mcmax_case * 2)
            if should_stop:
                break
        est = get_estimand(cname)
        status = "not_identifiable"
        if est_ok and align_conv and np.isfinite(align_err) and align_err <= args.target_err:
            status = "aligned_accurate"
        elif est_ok and align_conv and np.isfinite(align_err) and align_err > 0.50:
            status = "structurally_non_comparable"
        elif est_ok and align_conv and np.isfinite(align_err):
            status = "aligned_but_data_limited"
        elif est_ok:
            status = "estimated_not_converged"
        rows.append(
            {
                "case": cname,
                "truth_ref": est.truth_reference,
                "id_ok": got_paper == expected_id,
                "identifiable": got_paper,
                "est_ok": est_ok,
                "est_mc_conv": est_conv,
                "est_mc_delta": est_delta,
                "est_used_mc": est_used_mc,
                "truth_mc_conv": truth_conv,
                "truth_mc_delta": truth_delta,
                "truth_used_nu": truth_used_n,
                "aligned_truth_ATE": align_truth,
                "align_conv": align_conv,
                "align_mc_delta": align_delta,
                "align_used_mc": align_used_mc,
                "|err_aligned|": align_err,
                "rounds": rounds_used,
                "sim_nu_used": sim_nu_case,
                "sim_ns_used": sim_ns_case,
                "elapsed_s": round(time.time() - start_t, 2),
                "ATE_hat": ate_hat,
                "true_ATE": truth_ate if has_truth else float("nan"),
                "|err|": abs_err,
                "status": status,
                "note": est_msg or "",
            }
        )
    df = pd.DataFrame(rows)
    print(
        "settings: sim_nu={} sim_ns={}  est_mc=[{}..{}] truth_nu=[{}..{}] tol={} target_err={} auto_tune={}\n".format(
            sim_nu, sim_ns, est_n_mc0, est_n_mc_max, truth_nu0, truth_nu_max, args.tol, args.target_err, args.auto_tune
        )
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(df.to_string(index=False))
    print(
        "\nwe interpret: 'truth_ref' is ad hoc per case (gallery_estimands). "
        "'est_mc_conv'/'truth_mc_conv' indicate numerical MC stabilization only; "
        "|err| -> 0 for all graphs is not guaranteed at fixed observed sample size."
    )
    print("\nstatus counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    if args.write_csv:
        out = Path(args.write_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("\nwrote csv: {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
