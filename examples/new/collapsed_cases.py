"""
Collapsed do-calculus graph cases from do_calculus.py.
Shared by collapsed_do_calculus_graphs_demo. build_cgm_for_case(dc, case) needs do_calculus module.
"""
from __future__ import annotations

from hierarchicalcausalmodels.models import HSCMParametric


def _empty_fun(*args, **kwargs):
    return None


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


# (name, nodes, edges, unit_nodes, subunit_nodes, augment: (q_hat, parents) or None,
#  marginalize: (q_hat, special_parents) or None, Y, X, unobserved, expected_identifiable)
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


def build_cgm_for_case(dc, case):
    """Build CGM for one case: HSCM -> collapse -> optional augment -> optional marginalize."""
    (name, nodes, edges, unit_nodes, subunit_nodes, augment, marginalize, Y, X, unobserved, expected_id) = case
    hscm = _make_hscm(nodes, edges, unit_nodes, subunit_nodes)
    cgm = dc.collapse(hscm)
    if augment is not None:
        q_hat, parents = augment
        cgm = dc.augment_collapsed_model(cgm, q_hat, parents)
    if marginalize is not None:
        q_hat, special_parents = marginalize
        cgm = dc.marginalize_augmented_model(cgm, q_hat, special_parents)
    return cgm, unobserved, Y, X, expected_id
