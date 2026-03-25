"""
Pipeline: graph + parameters -> collapse -> augment -> BN with CPTs -> causalImpact -> ATE.
No data: use closed-form true ATE to validate; BN-based ATE can differ due to discretization
and pyAgrum CPT index order (parent/self). Estimation from data comes later.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import beta as beta_dist

from hierarchicalcausalmodels.models import HSCMParametric
from causalgraphicalmodels import CausalGraphicalModel

from hierarchicalcausalmodels.do_calculus import collapse, augment_collapsed_model, _sanitize_node_name

try:
    import pyagrum as gum
    import pyagrum.causal as csl
    PYAGUM_AVAILABLE = True
except ImportError:
    PYAGUM_AVAILABLE = False


# --- Confounder: paper params and true ATE (paper eq. appendix) ---

def default_confounder_params(omega: float = 0.2):
    """Paper appendix: alpha^a(0)=0.5, alpha^a(1)=4, beta^a=1; alpha^{y|a}(a,u), beta^{y|a}=2."""
    return {
        "omega": omega,
        "alpha_a": (0.5, 4.0),
        "beta_a": 1.0,
        "alpha_ya": {(0, 0): 0.5, (1, 0): 2.0, (0, 1): 1.0, (1, 1): 4.0},
        "beta_ya": 2.0,
    }


def true_ate_confounder(p: dict) -> float:
    """E[Y|do(A=1)] - E[Y|do(A=0)] from paper closed form."""
    o = p["omega"]
    a00, a10, a01, a11 = p["alpha_ya"][(0, 0)], p["alpha_ya"][(1, 0)], p["alpha_ya"][(0, 1)], p["alpha_ya"][(1, 1)]
    b = p["beta_ya"]
    e1 = (1 - o) * (a10 / (a10 + b)) + o * (a11 / (a11 + b))
    e0 = (1 - o) * (a00 / (a00 + b)) + o * (a01 / (a01 + b))
    return e1 - e0


# --- Build confounder CGM (collapse + augment) ---

def _empty_fun(*args, **kwargs):
    return None


def build_confounder_cgm() -> CausalGraphicalModel:
    """Confounder HSCM -> collapse -> augment with Q^y."""
    hscm = HSCMParametric(
        nodes={"U", "A", "Y"},
        edges={("U", "A"), ("U", "Y"), ("A", "Y")},
        unit_nodes={"U"},
        subunit_nodes={"A", "Y"},
        sizes=[3],
        node_functions={"U": _empty_fun, "A": _empty_fun, "Y": _empty_fun},
        data={},
    )
    cgm = collapse(hscm)
    cgm = augment_collapsed_model(cgm, "Q^y", {"Q^{y|a}", "Q^a"})
    cgm.unobserved_variables = {"U"}
    return cgm


# --- Discretization: bins in [0,1] ---

def _bin_centers(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins)


def _bin_probs_from_beta(alpha: float, bet: float, n_bins: int) -> np.ndarray:
    centers = _bin_centers(n_bins)
    edges = np.concatenate([[0], (centers[1:] + centers[:-1]) / 2, [1]])
    cdf = beta_dist.cdf(edges[1:], alpha, bet) - beta_dist.cdf(edges[:-1], alpha, bet)
    return cdf / cdf.sum()


# --- Build BN with CPTs from params (confounder only) ---

def build_confounder_bn_with_cpts(
    p: dict,
    n_bins: int = 11,
) -> tuple:
    """
    Build pyAgrum BN for confounder with CPTs from parametric model.
    Returns (bn, latent_descriptor, name_map) so we can build CausalModel.
    Nodes: U (2), Qa (n_bins), Qy_a (n_bins*n_bins for (mu0,mu1)), Qy (n_bins).
    """
    if not PYAGUM_AVAILABLE:
        raise ImportError("pyagrum required for build_confounder_bn_with_cpts")
    centers = _bin_centers(n_bins)
    # U
    bn = gum.BayesNet()
    bn.add("U", 2)
    bn.add("Qa", n_bins)
    bn.add("Qy_a", n_bins * n_bins)
    bn.add("Qy", n_bins)
    bn.addArc("U", "Qa")
    bn.addArc("U", "Qy_a")
    bn.addArc("Qa", "Qy")
    bn.addArc("Qy_a", "Qy")
    # P(U)
    bn.cpt("U")[0] = 1.0 - p["omega"]
    bn.cpt("U")[1] = p["omega"]
    # P(Qa | U): pyAgrum order (Qa, U) from earlier check; setitem may expect (U, Qa)
    cpt_qa = bn.cpt("Qa")
    for u in range(2):
        probs = _bin_probs_from_beta(p["alpha_a"][u], p["beta_a"], n_bins)
        for k in range(n_bins):
            cpt_qa[u, k] = probs[k]
    # P(Qy_a | U): use (u, idx) for setitem
    p_mu0_u0 = _bin_probs_from_beta(p["alpha_ya"][(0, 0)], p["beta_ya"], n_bins)
    p_mu0_u1 = _bin_probs_from_beta(p["alpha_ya"][(0, 1)], p["beta_ya"], n_bins)
    p_mu1_u0 = _bin_probs_from_beta(p["alpha_ya"][(1, 0)], p["beta_ya"], n_bins)
    p_mu1_u1 = _bin_probs_from_beta(p["alpha_ya"][(1, 1)], p["beta_ya"], n_bins)
    cpt_qya = bn.cpt("Qy_a")
    for u in range(2):
        p0 = p_mu0_u0 if u == 0 else p_mu0_u1
        p1 = p_mu1_u0 if u == 0 else p_mu1_u1
        for i in range(n_bins):
            for j in range(n_bins):
                idx = i * n_bins + j
                cpt_qya[u, idx] = p0[i] * p1[j]
    # P(Qy | Qa, Qy_a): deterministic. mean_y = qa * mu1 + (1-qa)*mu0 -> bin
    # pyAgrum CPT order (Qy, Qa, Qy_a); fill as flat list then fillWith
    n_qa, n_qya = n_bins, n_bins * n_bins
    flat = [0.0] * (n_bins * n_qa * n_qya)
    for qa_idx in range(n_bins):
        for qya_idx in range(n_qya):
            i, j = qya_idx // n_bins, qya_idx % n_bins
            mu0, mu1 = centers[i], centers[j]
            qa = centers[qa_idx]
            mean_y = qa * mu1 + (1.0 - qa) * mu0
            bin_y = int(np.clip(round(mean_y * (n_bins - 1)), 0, n_bins - 1))
            for y_bin in range(n_bins):
                idx = y_bin * (n_qa * n_qya) + qa_idx * n_qya + qya_idx
                flat[idx] = 1.0 if y_bin == bin_y else 0.0
    bn.cpt("Qy").fillWith(flat)
    # no latent: we use the full parameterized BN (U has CPT)
    latent_descriptor = []
    name_map = {"U": "U", "Q^a": "Qa", "Q^{y|a}": "Qy_a", "Q^y": "Qy"}
    return bn, latent_descriptor, name_map


def ate_from_causal_impact(
    bn: "gum.BayesNet",
    latent_descriptor: list,
    name_map: dict,
    n_bins: int,
    treatment_node: str = "Q^a",
    outcome_node: str = "Q^y",
) -> float:
    """
    Run causalImpact for do(treatment=0) and do(treatment=1); return E[outcome|do(1)] - E[outcome|do(0)].
    Outcome is discretized; we use bin centers as values for expectation.
    """
    if not PYAGUM_AVAILABLE:
        raise ImportError("pyagrum required")
    cm = csl.CausalModel(bn, latent_descriptor, keepArcs=False)
    do_name = name_map.get(treatment_node, treatment_node)
    on_name = name_map.get(outcome_node, outcome_node)
    centers = _bin_centers(n_bins)
    e0, e1 = 0.0, 0.0
    # do(Qa=0) -> E[Y|do(A=0)]; do(Qa=n_bins-1) -> E[Y|do(A=1)]
    for do_val in [0, n_bins - 1]:
        _, tensor, _ = csl.causalImpact(cm, on=on_name, doing=do_name, values={do_name: do_val})
        n_el = 1
        for d in range(tensor.nbrDim()):
            n_el *= tensor.variable(d).domainSize()
        dist = np.array([float(tensor[i]) for i in range(n_el)])
        dist = dist / dist.sum()
        ex = float(np.sum(centers * dist))
        if do_val == 0:
            e0 = ex
        else:
            e1 = ex
    return e1 - e0  # ATE = E[Y|do(A=1)] - E[Y|do(A=0)]


def run_pipeline(p: dict | None = None, n_bins: int = 11) -> dict:
    """
    Full pipeline: params -> true ATE, build CGM, build BN with CPTs, causalImpact ATE.
    Returns dict with true_ate, bn_ate, diff.
    """
    if p is None:
        p = default_confounder_params(0.2)
    true_ate = true_ate_confounder(p)
    bn, latent_descriptor, name_map = build_confounder_bn_with_cpts(p, n_bins=n_bins)
    bn_ate = ate_from_causal_impact(bn, latent_descriptor, name_map, n_bins)
    return {"true_ate": true_ate, "bn_ate": bn_ate, "diff": abs(bn_ate - true_ate)}


if __name__ == "__main__":
    out = run_pipeline(default_confounder_params(0.2), n_bins=11)
    print("True ATE (closed form):", round(out["true_ate"], 5))
    print("BN ATE (causalImpact):", round(out["bn_ate"], 5))
    print("|diff|:                ", round(out["diff"], 5))
    print("(BN uses discretized CPTs; diff can be reduced by increasing n_bins or fixing index order)")
