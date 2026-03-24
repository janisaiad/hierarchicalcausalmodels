"""
One estimand definition per collapsed-case gallery row (hcm_framework_test §4).

The plug-in estimator targets P(Y_outcome | do(X_intervention)) expressed on the collapsed CGM
(nodes Y, X from the case tuple). Reference truth is either the §1–3 curated DGP Monte Carlo
expectations or the structural binary-plate do() difference from the same plate HSCM as
simulate_binary_hscm / mc_truth_ate_binary_plate — see truth_reference on each record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TruthReference = Literal["curated_dgp_mc", "structural_binary_plate_do"]


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
        truth_reference="curated_dgp_mc",
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
        truth_reference="curated_dgp_mc",
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
        truth_reference="curated_dgp_mc",
    ),
    "ID_ex3_collapse": GalleryEstimand(
        case_name="ID_ex3_collapse",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on unit-level Y in the collapsed four-node CGM (U, Q^a, Q^{w|a}, Y); "
            "reference truth is the binary-plate structural do difference with all subunit A "
            "set to 0 vs 1 and Y aggregated as the unit node."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "ID_ex2_aug": GalleryEstimand(
        case_name="ID_ex2_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "ATE on augmented unit outcome Q^y w.r.t. Q^a with latent plate structure "
            "(Q^{z|a}, Q^{y|a|z}); reference truth pins subunit A to 0/1 and takes mean Q^y per unit."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "ID_ex6_collapse": GalleryEstimand(
        case_name="ID_ex6_collapse",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on unit-level Y in the multi-confounder collapse (U, Up, W, Z plates); "
            "reference truth from binary-plate do on A with Y as unit outcome."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "ID_ex4_aug": GalleryEstimand(
        case_name="ID_ex4_aug",
        outcome_node="W",
        intervention_node="Q^a",
        description=(
            "Effect of Q^a on unit-level W (not Y) after augment/marginalize; reference truth uses "
            "the same plate do on A and reads W as the outcome aggregate."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "ID_ex1_mar": GalleryEstimand(
        case_name="ID_ex1_mar",
        outcome_node="Y",
        intervention_node="Q^w",
        description=(
            "ATE of Q^w (augmented summary over W plate) on Y after marginalizing Q^z; "
            "structural reference pins subunit W to 0/1 when the mapping Q^w -> W exists."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "ID_ex5_mar": GalleryEstimand(
        case_name="ID_ex5_mar",
        outcome_node="Y",
        intervention_node="Q^{a|x}",
        description=(
            "ATE of Q^{a|x} on Y after marginalizing Q^z; reference do holds A and X subunit plates "
            "to common values 0 vs 1 when both are forced."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "ID_targeted_aug": GalleryEstimand(
        case_name="ID_targeted_aug",
        outcome_node="Q^y",
        intervention_node="Q^{a|x}",
        description=(
            "ATE on Q^y w.r.t. Q^{a|x} in the targeted-augment graph; reference truth uses do on "
            "subunits A and X together for the binary-plate simulator."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "nonID_ex5_aug": GalleryEstimand(
        case_name="nonID_ex5_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "Causal contrast E[Q^y | do(Q^a=1)] − E[Q^y | do(Q^a=0)] on the stated SCM is the "
            "quantity of interest; the do-calculus row may still report non-identifiability while "
            "the structural MC reference is computed for comparison only."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "nonID_ex4_aug": GalleryEstimand(
        case_name="nonID_ex4_aug",
        outcome_node="Y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on Y with extra latent Up; same structural reference rule as other Q^a→Y "
            "plate cases; identification may fail even though the MC contrast is defined."
        ),
        truth_reference="structural_binary_plate_do",
    ),
    "nonID_ex1_aug": GalleryEstimand(
        case_name="nonID_ex1_aug",
        outcome_node="Q^y",
        intervention_node="Q^a",
        description=(
            "ATE of Q^a on Q^y in the non-ID variant; reference truth from binary-plate do on A "
            "with Q^y as mean subunit Y per unit."
        ),
        truth_reference="structural_binary_plate_do",
    ),
}


def get_estimand(case_name: str) -> GalleryEstimand:
    return GALLERY_ESTIMANDS[case_name]
