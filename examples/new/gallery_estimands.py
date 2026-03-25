"""
One estimand definition per collapsed-case gallery row (hcm_framework_test §4).

The plug-in estimator targets P(Y_outcome | do(X_intervention)) expressed on the collapsed CGM
(nodes Y, X from the case tuple). The §4 reference **`true_ATE`** is always the same operational
target as the estimate: `identify_effect` + `estimate_causal_effect` evaluated on a large
observational sample from the row DGP (`aligned_plugin_large_n` in truth_reference).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TruthReference = Literal["aligned_plugin_large_n"]


@dataclass(frozen=True)
class GalleryEstimand:
    case_name: str
    outcome_node: str
    intervention_node: str
    description: str
    truth_reference: TruthReference


# we one row per entry in COLLAPSED_DO_CALCULUS_CASES (same order as case[0])
GALLERY_ESTIMANDS: dict[str, GalleryEstimand] = {
    "confounder_aug": GalleryEstimand(
        case_name="confounder_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "Average treatment effect on the unit-level mean subunit outcome Q^y when the "
            "unit-level treatment summary Q^a is set by a sharp intervention on all subunit "
            "A indicators (do) relative to baseline do(Q^a=0) vs do(Q^a=1), under the confounded "
            "plate model; identified via back-door adjustment on the collapsed CGM using "
            "observable Q^{y|a}."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "confounder_interferer_aug": GalleryEstimand(
        case_name="confounder_interferer_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "Same estimand as confounder_aug on Q^y and Q^a, but the graph includes unit mediator Z "
            "and interference edges; the identified interventional distribution uses the front-door "
            "route through Z on the augmented collapsed CGM."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "instrument_mar": GalleryEstimand(
        case_name="instrument_mar",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "Effect of Q^a on unit-level Y on the marginalized instrument CGM (Q^z removed); "
            "identification uses the IV structure through Q^{a|z} and Q^z; estimand is "
            "E[Y | do(Q^a=1)] − E[Y | do(Q^a=0)] for that collapsed model."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_ex3_collapse": GalleryEstimand(
        case_name="ID_ex3_collapse",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on unit-level Y in the collapsed four-node CGM (U, Q^a, Q^{w|a}, Y); "
            "§4 truth is the aligned identified plug-in at large n (same as estimate_causal_effect)."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_ex2_aug": GalleryEstimand(
        case_name="ID_ex2_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "ATE on augmented unit outcome Q^y w.r.t. Q^a with latent plate structure "
            "(Q^{z|a}, Q^{y|a|z}); §4 truth is the aligned plug-in on the binary-plate simulator."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_ex6_collapse": GalleryEstimand(
        case_name="ID_ex6_collapse",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on unit-level Y in the multi-confounder collapse (U, Up, W, Z plates); "
            "§4 truth is the aligned plug-in on the binary-plate simulator."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_ex4_aug": GalleryEstimand(
        case_name="ID_ex4_aug",
        outcome_node="W",
        intervention_node="Q^a",
        description=(
            "Effect of Q^a on unit-level W (not Y) after augment/marginalize; §4 truth is the aligned "
            "plug-in with W as the outcome node."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_ex1_mar": GalleryEstimand(
        case_name="ID_ex1_mar",
        outcome_node="Y",
        intervention_node="Q^w",
        description=(
            "ATE of Q^w (augmented summary over W plate) on Y after marginalizing Q^z; "
            "§4 truth is the aligned plug-in on the binary-plate simulator."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_ex5_mar": GalleryEstimand(
        case_name="ID_ex5_mar",
        outcome_node="Y",
        intervention_node="Q^{a|x}",
        description=(
            "ATE of Q^{a|x} on Y after marginalizing Q^z; §4 truth is the aligned plug-in on the "
            "binary-plate simulator."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "ID_targeted_aug": GalleryEstimand(
        case_name="ID_targeted_aug",
        outcome_node="Q^y",
        intervention_node="Q^{a|x}",
        description=(
            "ATE on Q^y w.r.t. Q^{a|x} in the targeted-augment graph; §4 truth is the aligned plug-in."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "nonID_ex5_aug": GalleryEstimand(
        case_name="nonID_ex5_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "Causal contrast E[Q^y | do(Q^a=1)] − E[Q^y | do(Q^a=0)] on the stated SCM is the "
            "quantity of interest; the do-calculus row may still report non-identifiability while "
            "§4 truth is still the aligned plug-in for comparison when computable."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "nonID_ex4_aug": GalleryEstimand(
        case_name="nonID_ex4_aug",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on Y with extra latent Up; same aligned plug-in reference as other Q^a→Y "
            "plate cases; identification may fail even though the plug-in contrast is defined."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
    "nonID_ex1_aug": GalleryEstimand(
        case_name="nonID_ex1_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on Q^y in the non-ID variant; §4 truth is the aligned plug-in on the "
            "binary-plate simulator."
        ),
        truth_reference="aligned_plugin_large_n",
    ),
}


def get_estimand(case_name: str) -> GalleryEstimand:
    return GALLERY_ESTIMANDS[case_name]
