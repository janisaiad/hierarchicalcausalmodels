from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from causalgraphicalmodels import CausalGraphicalModel

from hierarchicalcausalmodels.models import HSCMParametric
from hierarchicalcausalmodels.do_calculus import (
    collapse,
    augment_collapsed_model,
    marginalize_augmented_model,
)


def _empty_fun(*args, **kwargs):
    return None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _make_hscm(nodes, edges, unit_nodes, subunit_nodes):
    return HSCMParametric(
        nodes=nodes,
        edges=edges,
        unit_nodes=unit_nodes,
        subunit_nodes=subunit_nodes,
        sizes=[3],
        node_functions={n: _empty_fun for n in nodes},
        data={},
    )


COLLAPSED_DO_CALCULUS_CASES = [
    ("confounder_aug", {"U", "A", "Y"}, {("U", "A"), ("U", "Y"), ("A", "Y")}, {"U"}, {"A", "Y"}, ("Q^y", {"Q^{y|a}", "Q^a"}), None, "Q^y", "Q^a", {"U"}, True),
    ("confounder_interferer_aug", {"U", "Z", "A", "Y"}, {("U", "A"), ("U", "Y"), ("A", "Y"), ("A", "Z"), ("Z", "Y")}, {"U", "Z"}, {"A", "Y"}, ("Q^y", {"Q^{y|a}", "Q^a"}), None, "Q^y", "Q^a", {"U"}, True),
    ("instrument_mar", {"U", "Y", "Z", "A"}, {("U", "A"), ("U", "Y"), ("Z", "A"), ("A", "Y")}, {"U", "Y"}, {"Z", "A"}, ("Q^a", {"Q^z", "Q^{a|z}"}), ("Q^a", {"Q^z"}), "Y", "Q^a", {"U"}, True),
    ("ID_ex3_collapse", {"U", "A", "W", "Y"}, {("U", "A"), ("U", "W"), ("A", "W"), ("A", "Y"), ("W", "Y")}, {"U", "Y"}, {"A", "W"}, None, None, "Y", "Q^a", {"U"}, True),
    ("ID_ex2_aug", {"U", "A", "Z", "Y"}, {("U", "A"), ("U", "Z"), ("U", "Y"), ("A", "Z"), ("A", "Y"), ("Z", "Y")}, {"U"}, {"A", "Z", "Y"}, ("Q^y", {"Q^a", "Q^{z|a}", "Q^{y|a|z}"}), None, "Q^y", "Q^a", {"U"}, True),
    ("ID_ex6_collapse", {"A", "Z", "U", "W", "Y", "Up"}, {("A", "Z"), ("U", "Z"), ("U", "A"), ("A", "W"), ("W", "Z"), ("A", "Y"), ("W", "Y"), ("Z", "Y"), ("Up", "W"), ("Up", "Y")}, {"U", "W", "Y", "Up"}, {"A", "Z"}, None, None, "Y", "Q^a", {"U", "Up"}, True),
    ("ID_ex4_aug", {"Z", "A", "U", "W", "Y"}, {("Z", "A"), ("U", "A"), ("U", "W"), ("U", "Y"), ("A", "W"), ("A", "Y"), ("W", "Y")}, {"U", "W"}, {"Z", "A", "Y"}, ("Q^a", {"Q^z", "Q^{a|z}"}), None, "W", "Q^a", {"U"}, True),
    ("ID_ex1_mar", {"Z", "W", "A", "U", "Y"}, {("Z", "W"), ("U", "A"), ("U", "W"), ("U", "Y"), ("A", "W"), ("W", "Y")}, {"U", "Y"}, {"Z", "A", "W"}, ("Q^w", {"Q^a", "Q^{w|a|z}", "Q^z"}), ("Q^w", {"Q^z"}), "Y", "Q^w", {"U"}, True),
    ("ID_ex5_mar", {"Z", "A", "X", "Y", "U"}, {("Z", "A"), ("U", "A"), ("U", "X"), ("U", "Y"), ("X", "A"), ("X", "Y"), ("A", "Y")}, {"U", "Y"}, {"Z", "A", "X"}, ("Q^{a|x}", {"Q^z", "Q^{a|x|z}"}), ("Q^{a|x}", {"Q^z"}), "Y", "Q^{a|x}", {"U"}, True),
    ("ID_targeted_aug", {"A", "X", "Y", "U", "Z"}, {("U", "A"), ("U", "Y"), ("U", "X"), ("A", "Z"), ("A", "Y"), ("Z", "Y"), ("X", "A"), ("X", "Y"), ("X", "Z")}, {"U", "Z"}, {"A", "X", "Y"}, ("Q^y", {"Q^x", "Q^{a|x}", "Q^{y|a|x}"}), None, "Q^y", "Q^{a|x}", {"U"}, True),
    ("nonID_ex5_aug", {"U", "A", "Y", "Z"}, {("U", "A"), ("U", "Y"), ("A", "Y"), ("A", "Z"), ("Z", "Y"), ("U", "Z")}, {"U", "Z"}, {"A", "Y"}, ("Q^y", {"Q^a", "Q^{y|a}"}), None, "Q^y", "Q^a", {"U"}, True),
    ("nonID_ex4_aug", {"U", "Up", "A", "Z", "Y"}, {("U", "A"), ("U", "Y"), ("Up", "A"), ("Up", "Z"), ("Z", "A"), ("A", "Y")}, {"U", "Up", "Y"}, {"A", "Z"}, ("Q^a", {"Q^z", "Q^{a|z}"}), None, "Y", "Q^a", {"U", "Up"}, True),
    ("nonID_ex1_aug", {"U", "A", "W", "Y"}, {("U", "A"), ("U", "W"), ("A", "W"), ("A", "Y"), ("W", "Y")}, {"U", "W"}, {"A", "Y"}, ("Q^y", {"Q^a", "Q^{y|a}"}), None, "Q^y", "Q^a", {"U"}, True),
]


def _ensure_treatment_edge_to_outcome(cgm, y_node: str, x_node: str, apply: bool) -> CausalGraphicalModel:
    if not apply:
        return cgm
    nodes = list(cgm.dag.nodes)
    edges = list(cgm.dag.edges)
    if x_node not in nodes or y_node not in nodes:
        return cgm
    if (x_node, y_node) in edges:
        return cgm
    return CausalGraphicalModel(nodes=nodes, edges=edges + [(x_node, y_node)])


def build_cgm_for_case(case):
    (name, nodes, edges, unit_nodes, subunit_nodes, augment, marginalize, Y, X, unobserved, expected_id) = case
    hscm = _make_hscm(nodes, edges, unit_nodes, subunit_nodes)
    cgm = collapse(hscm)
    if augment is not None:
        q_hat, parents = augment
        cgm = augment_collapsed_model(cgm, q_hat, parents)
    if marginalize is not None:
        q_hat, special_parents = marginalize
        cgm = marginalize_augmented_model(cgm, q_hat, special_parents)
    _patch_xy = {
        "instrument_mar": ("Q^a", "Y"),
        "ID_ex2_aug": ("Q^a", "Q^y"),
        "ID_ex6_collapse": ("Q^a", "Y"),
        "ID_ex4_aug": ("Q^a", "W"),
        "nonID_ex4_aug": ("Q^a", "Y"),
    }
    if name in _patch_xy:
        xa, ya = _patch_xy[name]
        cgm = _ensure_treatment_edge_to_outcome(cgm, ya, xa, apply=True)
    return cgm, unobserved, Y, X, expected_id


def gallery_unobserved_set(case, cgm, mode, y_node, x_node):
    unit_nodes = case[3]
    if mode == "all_unit_nodes":
        base = set(unit_nodes) & set(cgm.dag.nodes)
        return base - {y_node, x_node}
    if mode == "case_default":
        return set(case[9]) & set(cgm.dag.nodes)
    raise ValueError("unknown GALLERY_UNOBSERVED_MODE: {!r}".format(mode))


def gallery_x_for_case(case, x_override_by_case=None):
    cname = case[0]
    if x_override_by_case and cname in x_override_by_case:
        return x_override_by_case[cname]
    return case[8]


def gallery_case_knobs(case_name, sim_nu, sim_ns, n_mc, sim_nu_by_case=None, sim_ns_by_case=None, n_mc_by_case=None):
    sim_nu_by_case = sim_nu_by_case or {}
    sim_ns_by_case = sim_ns_by_case or {}
    n_mc_by_case = n_mc_by_case or {}
    nu = int(sim_nu_by_case.get(case_name, sim_nu))
    ns = int(sim_ns_by_case.get(case_name, sim_ns))
    nmc = int(n_mc_by_case.get(case_name, n_mc))
    return nu, ns, nmc


def simulate_binary_hscm(hscm, n_units, n_sub, rng):
    G = nx.DiGraph()
    G.add_nodes_from(hscm.nodes)
    G.add_edges_from(hscm.edges)
    order = list(nx.topological_sort(G))
    stor: dict[Any, Any] = {}
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


def bern_families(data_dict):
    return {k: "bernoulli" for k in data_dict}


def curated_gallery_data_large(cname, n_units, n_sub, rng):
    if cname == "confounder_aug":
        u = rng.beta(2, 2, n_units)
        a = np.array([rng.binomial(1, np.clip(u[i], 0, 1), n_sub) for i in range(n_units)], dtype=float)
        y = np.array([rng.binomial(1, np.clip(sigmoid(1.5 * a[i] + u[i] - 0.5), 0, 1)) for i in range(n_units)], dtype=float)
        return {"A": a, "Y": y}
    if cname == "confounder_interferer_aug":
        u = rng.beta(2, 2, n_units)
        a = np.array([rng.binomial(1, np.clip(u[i], 0, 1), n_sub) for i in range(n_units)], dtype=float)
        q_a = a.mean(axis=1)
        z = rng.binomial(1, np.clip(sigmoid(2 * q_a - 1), 0, 1)).astype(float)
        y = np.array([rng.binomial(1, np.clip(sigmoid(1.5 * a[i] + z[i] + 0.3 * u[i] - 0.5), 0, 1)) for i in range(n_units)], dtype=float)
        return {"A": a, "Y": y, "Z": z}
    if cname == "instrument_mar":
        u = rng.beta(2, 2, n_units)
        q_z = rng.beta(2, 2, n_units)
        z = rng.binomial(1, q_z[:, None] * np.ones(n_sub)).astype(float)
        p_a = np.clip(0.60 * z + 0.55 * u[:, None] + 0.05, 0, 1)
        a = rng.binomial(1, p_a).astype(float)
        q_a = a.mean(axis=1)
        y = rng.binomial(1, np.clip(sigmoid(3 * q_a + 1.5 * u - 1.5), 0, 1)).astype(float)
        return {"Y": y, "A": a, "Z": z}
    raise ValueError("unknown curated gallery case: {!r}".format(cname))


def gallery_aligned_truth_ate(case, cgm, y_node, x_node, n_units, n_sub, n_mc, rng, identify_effect, estimate_causal_effect):
    cname = case[0]
    unobs_paper = set(case[9]) & set(cgm.dag.nodes)
    res_paper = identify_effect(cgm, Y=y_node, X=x_node, unobserved=unobs_paper)
    if not res_paper.identifiable:
        return float("nan")
    if cname in ("confounder_aug", "confounder_interferer_aug", "instrument_mar"):
        data_obs = curated_gallery_data_large(cname, n_units, n_sub, rng)
    else:
        hscm_sim = HSCMParametric(
            nodes=set(case[1]),
            edges=set(case[2]),
            unit_nodes=set(case[3]),
            subunit_nodes=set(case[4]),
            sizes=[n_sub] * n_units,
            node_functions={n: _empty_fun for n in case[1]},
            data={},
        )
        data_obs = simulate_binary_hscm(hscm_sim, n_units, n_sub, rng)
    fam = bern_families(data_obs)
    e1 = estimate_causal_effect(res_paper, data=data_obs, intervention={x_node: 1.0}, distribution_families=fam, random_seed=101, n_mc_samples=n_mc)
    e0 = estimate_causal_effect(res_paper, data=data_obs, intervention={x_node: 0.0}, distribution_families=fam, random_seed=202, n_mc_samples=n_mc)
    return float(e1 - e0)
