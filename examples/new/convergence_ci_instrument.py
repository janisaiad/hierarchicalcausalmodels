#!/usr/bin/env python3
"""
we scale n_units (fixed n_sub) and report |ATE_hat - true_ATE| for §2 C&I and §3 instrument DGPs.
run: uv run python examples/new/convergence_ci_instrument.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX))

import numpy as np
from causalgraphicalmodels import CausalGraphicalModel

from hierarchicalcausalmodels.models import HSCMParametric

import hierarchicalcausalmodels.do_calculus as dc
from hierarchicalcausalmodels.estimation import estimate_causal_effect


def _noop(_d=None):
    return None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def truth_ci_ate(u_mc: np.ndarray) -> tuple[float, float, float]:
    def ey(a_val: float) -> float:
        p_z = sigmoid(2 * a_val - 1)
        ey_z1 = sigmoid(1.5 * a_val + 1 + 0.3 * u_mc - 0.5)
        ey_z0 = sigmoid(1.5 * a_val + 0 + 0.3 * u_mc - 0.5)
        return float(np.mean(p_z * ey_z1 + (1 - p_z) * ey_z0))

    e1 = ey(1.0)
    e0 = ey(0.0)
    return e1, e0, e1 - e0


def truth_inst_ate(u_mc: np.ndarray) -> tuple[float, float, float]:
    e1 = float(np.mean(sigmoid(3 * 1 + 1.5 * u_mc - 1.5)))
    e0 = float(np.mean(sigmoid(3 * 0 + 1.5 * u_mc - 1.5)))
    return e1, e0, e1 - e0


def sample_ci(n_units: int, n_sub: int, rng: np.random.Generator):
    u = rng.beta(2, 2, n_units)
    a = np.array(
        [rng.binomial(1, np.clip(u[i], 0, 1), n_sub) for i in range(n_units)],
        dtype=float,
    )
    q_a = a.mean(axis=1)
    z = rng.binomial(1, np.clip(sigmoid(2 * q_a - 1), 0, 1)).astype(float)
    y = np.array(
        [
            rng.binomial(
                1, np.clip(sigmoid(1.5 * a[i] + z[i] + 0.3 * u[i] - 0.5), 0, 1)
            )
            for i in range(n_units)
        ],
        dtype=float,
    )
    return {"A": a, "Y": y, "Z": z}


def sample_inst(n_units: int, n_sub: int, rng: np.random.Generator):
    u = rng.beta(2, 2, n_units)
    q_z = rng.beta(2, 2, n_units)
    z = rng.binomial(1, q_z[:, None] * np.ones(n_sub)).astype(float)
    p_a = np.clip(0.60 * z + 0.55 * u[:, None] + 0.05, 0, 1)
    a = rng.binomial(1, p_a).astype(float)
    q_a = a.mean(axis=1)
    y = rng.binomial(1, np.clip(sigmoid(3 * q_a + 1.5 * u - 1.5), 0, 1)).astype(float)
    return {"Y": y, "A": a, "Z": z}


def build_result_ci(n_units: int, n_sub: int):
    h = HSCMParametric(
        nodes={"U", "Z", "A", "Y"},
        edges={("U", "A"), ("U", "Y"), ("A", "Y"), ("A", "Z"), ("Z", "Y")},
        unit_nodes={"U", "Z"},
        subunit_nodes={"A", "Y"},
        sizes=[n_sub] * n_units,
        node_functions={"U": _noop, "Z": _noop, "A": _noop, "Y": _noop},
        data={},
    )
    c0 = dc.collapse(h)
    c0.unobserved_variables = {"U"}
    c1 = dc.augment_collapsed_model(c0, "Q^y", {"Q^{y|a}", "Q^a"})
    c1.unobserved_variables = {"U"}
    return dc.identify_effect(c1, Y={"Q^y"}, X={"Q^a"}, unobserved={"U"})


def build_result_inst(n_units: int, n_sub: int):
    h = HSCMParametric(
        nodes={"U", "Y", "Z", "A"},
        edges={("U", "A"), ("U", "Y"), ("Z", "A"), ("A", "Y")},
        unit_nodes={"U", "Y"},
        subunit_nodes={"Z", "A"},
        sizes=[n_sub] * n_units,
        node_functions={"U": _noop, "Y": _noop, "Z": _noop, "A": _noop},
        data={},
    )
    c0 = dc.collapse(h)
    aug = dc.augment_collapsed_model(c0, "Q^a", {"Q^z", "Q^{a|z}"})
    mar = dc.marginalize_augmented_model(aug, "Q^a", {"Q^z"})
    mar.unobserved_variables = {"U"}
    nodes = list(mar.dag.nodes)
    edges = list(mar.dag.edges)
    if ("Q^a", "Y") not in edges:
        mar = CausalGraphicalModel(nodes=nodes, edges=edges + [("Q^a", "Y")])
    return dc.identify_effect(mar, Y={"Y"}, X={"Q^a"}, unobserved={"U"})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="C&I and instrument ATE convergence vs n_units")
    ap.add_argument(
        "--n-units",
        type=int,
        nargs="+",
        default=[300, 1000, 5000],
        help="unit counts to try (e.g. 1000 5000)",
    )
    ap.add_argument("--n-sub", type=int, default=80, help="subunits per unit")
    args = ap.parse_args()

    if not dc.PYAGNUM_AVAILABLE:
        print("pyagrum required")
        sys.exit(1)

    n_sub = int(args.n_sub)
    n_grid = tuple(int(x) for x in args.n_units)
    rng_truth = np.random.default_rng(99991)
    u_mc = rng_truth.beta(2, 2, 400_000)
    _, _, true_ci = truth_ci_ate(u_mc)
    _, _, true_inst = truth_inst_ate(u_mc)

    print("fixed truth (large MC):  C&I ATE = {:.6f}   instrument ATE = {:.6f}".format(true_ci, true_inst))
    print("n_sub = {}   (inner n_mc_samples scales down when n_u is large so the script finishes)\n".format(n_sub))

    print("{:>6}  {:>12}  {:>12}  {:>10}  |  {:>12}  {:>12}  {:>10}".format(
        "n_u", "ATE_hat_ci", "err_ci", "|err|", "ATE_hat_iv", "err_iv", "|err|",
    ))
    print("-" * 88)

    for n_u in n_grid:
        n_mc = 1200 if n_u < 3000 else 600
        print("  ... n_u = {}  n_mc = {} (running)".format(n_u, n_mc), flush=True)
        rng = np.random.default_rng(42 + int(n_u))
        res_ci = build_result_ci(n_u, n_sub)
        res_iv = build_result_inst(n_u, n_sub)
        if not res_ci.identifiable or not res_iv.identifiable:
            print(n_u, "ID failed")
            continue
        d_ci = sample_ci(n_u, n_sub, rng)
        d_iv = sample_inst(n_u, n_sub, rng)
        fam_ci = {k: "bernoulli" for k in d_ci}
        fam_iv = {k: "bernoulli" for k in d_iv}
        e1c = estimate_causal_effect(
            res_ci,
            data=d_ci,
            intervention={"Q^a": 1.0},
            distribution_families=fam_ci,
            n_mc_samples=n_mc,
            random_seed=0,
        )
        e0c = estimate_causal_effect(
            res_ci,
            data=d_ci,
            intervention={"Q^a": 0.0},
            distribution_families=fam_ci,
            n_mc_samples=n_mc,
            random_seed=1,
        )
        hat_ci = float(e1c - e0c)
        err_ci = hat_ci - true_ci
        e1i = estimate_causal_effect(
            res_iv,
            data=d_iv,
            intervention={"Q^a": 1.0},
            distribution_families=fam_iv,
            n_mc_samples=n_mc,
            random_seed=0,
        )
        e0i = estimate_causal_effect(
            res_iv,
            data=d_iv,
            intervention={"Q^a": 0.0},
            distribution_families=fam_iv,
            n_mc_samples=n_mc,
            random_seed=1,
        )
        hat_iv = float(e1i - e0i)
        err_iv = hat_iv - true_inst
        print(
            "{:>6}  {:>12.6f}  {:>12.6f}  {:>10.6f}  |  {:>12.6f}  {:>12.6f}  {:>10.6f}".format(
                n_u,
                hat_ci,
                err_ci,
                abs(err_ci),
                hat_iv,
                err_iv,
                abs(err_iv),
            )
        )

    print("\nwe expect |err| to shrink with n_u if estimator is consistent; instrument row may plateau if estimand != structural truth.")


if __name__ == "__main__":
    main()
