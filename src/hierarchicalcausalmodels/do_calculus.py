"""Do-calculus helpers for collapsed and augmented causal graphical models (pyAgrum).

Utilities to collapse hierarchical SCMs to flat CGMs, suggest and apply augmentations,
and run identification via pyAgrum causal do-calculus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hierarchicalcausalmodels.models import HSCMParametric
from causalgraphicalmodels import CausalGraphicalModel
import copy

try:
    import pyagrum as gum
    import pyagrum.causal as csl
    PYAGNUM_AVAILABLE = True
except ImportError:
    PYAGNUM_AVAILABLE = False


def _sanitize_node_name(name: str) -> str:
    """Sanitize CGM node name for pyAgrum (no ^{}|). Paper Q^{y|a} -> Qy_a."""
    return name.replace("^", "").replace("{", "").replace("}", "").replace("|", "_")


def cgm_to_pyagrum_causal(cgm: CausalGraphicalModel, unobserved=None, domain_size: int = 2):
    """
    Build a pyAgrum BayesNet and latent descriptor from a (collapsed/augmented) CGM.

    Use this to run do-calculus on the flat graph with Q-variables. The BN has only
    observed nodes; unobserved nodes become latent confounder groups in CausalModel.

    Parameters
    ----------
    cgm : CausalGraphicalModel
        Collapsed (and optionally augmented/marginalized) CGM with nodes like Q^a, Q^{y|a}, U, etc.
    unobserved : set[str] or None
        Node names to treat as unobserved (e.g. {"U"}). If None, uses cgm.unobserved_variables
        when it is a set of node names in the graph; otherwise no latent.
    domain_size : int
        Number of states per variable in the BN (pyAgrum is discrete). 2 = binary.

    Returns
    -------
    bn : pyagrum.BayesNet
        Observational BN over observed nodes with uniform CPTs (for identification only).
    latent_descriptor : list[tuple[str, list[str]]]
        For CausalModel: list of (latent_name, list of observed children).
    name_map : dict[str, str]
        Paper name -> sanitized name used in the BN (for calling identifyingIntervention).
    """
    if not PYAGNUM_AVAILABLE:
        raise ImportError("pyagrum is required for cgm_to_pyagrum_causal. Install with: pip install pyagrum")
    nodes_all = set(cgm.dag.nodes)
    if unobserved is not None:
        unobs = set(unobserved) & nodes_all
    else:
        u = getattr(cgm, "unobserved_variables", None)
        unobs = (set(u) & nodes_all) if u else set()
    observed = nodes_all - unobs
    name_map = {n: _sanitize_node_name(n) for n in observed}
    name_map.update({n: _sanitize_node_name(n) for n in unobs})
    obs_sanitized = [name_map[n] for n in sorted(observed)]
    bn = gum.BayesNet()
    for n in obs_sanitized:
        bn.add(n, domain_size)
    for (u, v) in cgm.dag.edges:
        if u in observed and v in observed:
            bn.addArc(name_map[u], name_map[v])
    for node in obs_sanitized:
        bn.cpt(node).fillWith(1.0).normalize()
    latent_descriptor = []
    for u in unobs:
        children_obs = [name_map[v] for v in cgm.dag.successors(u) if v in observed]
        if children_obs:
            latent_descriptor.append((name_map[u], children_obs))
    return bn, latent_descriptor, name_map


def run_do_calculus(cgm: CausalGraphicalModel, Y, X, unobserved=None, method: str = "identifyingIntervention"):
    """
    Run pyAgrum do-calculus on a collapsed/augmented CGM: get identification formula (and optionally impact).

    Y = set of outcome variable names (paper notation, e.g. {"Q^y"}).
    X = set of intervention variable names (e.g. {"Q^a"}).
    Soft intervention on subunit A corresponds to hard intervention on Q^a in the collapsed model.

    Parameters
    ----------
    cgm : CausalGraphicalModel
        Flat CGM with Q-variables (after collapse and optional augment/marginalize).
    Y : set[str] or str
        Outcome variable(s), e.g. {"Q^y"} or "Q^y".
    X : set[str] or str
        Intervention variable(s), e.g. {"Q^a"} or "Q^a".
    unobserved : set[str] or None
        Unobserved nodes (e.g. {"U"}). If None, inferred from cgm when possible.
    method : str
        "identifyingIntervention" (default) or "causalImpact".

    Returns
    -------
    If method == "identifyingIntervention": ASTtree with .toLatex().
    If method == "causalImpact": (formula, tensor, explanation).
    """
    if not PYAGNUM_AVAILABLE:
        raise ImportError("pyagrum is required for run_do_calculus. Install with: pip install pyagrum")
    Y_set = {Y} if isinstance(Y, str) else set(Y)
    X_set = {X} if isinstance(X, str) else set(X)
    bn, latent_descriptor, name_map = cgm_to_pyagrum_causal(cgm, unobserved=unobserved)
    Y_sanitized = [name_map[n] for n in Y_set if n in name_map]
    X_sanitized = [name_map[n] for n in X_set if n in name_map]
    if len(Y_sanitized) != len(Y_set) or len(X_sanitized) != len(X_set):
        missing = (Y_set | X_set) - set(name_map.keys())
        raise ValueError("Y or X refer to nodes not in the CGM or not observed: " + str(missing))
    cm = csl.CausalModel(bn, latent_descriptor, keepArcs=False)
    if method == "causalImpact":
        return csl.causalImpact(cm, on=Y_sanitized[0] if len(Y_sanitized) == 1 else Y_sanitized,
                                doing=X_sanitized[0] if len(X_sanitized) == 1 else X_sanitized)
    ast = csl.identifyingIntervention(cm, Y=set(Y_sanitized), X=set(X_sanitized))
    return ast


@dataclass
class DoCalculusResult:
    """Result of running the do-calculus identification engine on a collapsed/augmented CGM."""
    identifiable: bool
    formula_latex: Optional[str]
    explanation: str
    ast: Any = None
    impact_tensor: Any = None
    error: Optional[str] = None

    def __repr__(self) -> str:
        s = "DoCalculusResult(identifiable={}".format(self.identifiable)
        if self.formula_latex:
            s += ", formula_latex=<{} chars>".format(len(self.formula_latex))
        if self.error:
            s += ", error={!r}".format(self.error[:80])
        return s + ")"


def identify_effect(
    cgm: CausalGraphicalModel,
    Y: str | set[str],
    X: str | set[str],
    unobserved: Optional[set[str]] = None,
    with_impact: bool = False,
) -> DoCalculusResult:
    """
    Do-calculus engine: identify P(Y | do(X)) on a collapsed/augmented CGM.
    Single entry point for running pyAgrum do-calculus. Catches HedgeException
    and UnidentifiableException and returns a structured result.
    """
    if not PYAGNUM_AVAILABLE:
        return DoCalculusResult(
            identifiable=False,
            formula_latex=None,
            explanation="pyagrum not installed.",
            error="ImportError: pyagrum required",
        )
    try:
        ast = run_do_calculus(cgm, Y=Y, X=X, unobserved=unobserved, method="identifyingIntervention")
        formula_latex = ast.toLatex()
        impact_tensor = None
        if with_impact:
            _, impact_tensor, _ = run_do_calculus(
                cgm, Y=Y, X=X, unobserved=unobserved, method="causalImpact"
            )
        return DoCalculusResult(
            identifiable=True,
            formula_latex=formula_latex,
            explanation="Do-calculus identification succeeded.",
            ast=ast,
            impact_tensor=impact_tensor,
        )
    except (csl.HedgeException, csl.UnidentifiableException) as e:
        msg = getattr(e, "message", str(e))
        return DoCalculusResult(
            identifiable=False,
            formula_latex=None,
            explanation=msg,
            error=msg,
        )
    except Exception as e:
        return DoCalculusResult(
            identifiable=False,
            formula_latex=None,
            explanation="Engine error: {}".format(e),
            error=str(e),
        )


def _q_node_name_paper(subunit: str, original_edges, subunit_nodes: set) -> str:
    """Name for the Q variable per paper.tex: Q^{v|pa_S(v)} with pa_S = subunit-level parents.
    We use original_edges (not the mutated copy) so pa_S is correct. We strip a leading '_'
    so that _A -> a (HSCMParametric uses _ prefix for subunit nodes)."""
    pa_S = {p for (p, c) in original_edges if c == subunit and p in subunit_nodes}
    v = subunit.lstrip("_").lower()
    if not pa_S:
        return f"Q^{v}"
    return "Q^{" + v + "|" + "|".join(sorted(p.lstrip("_").lower() for p in pa_S)) + "}"


def _subunit_ancestors_of_unit(unit_node, original_edges, subunit_nodes, unit_nodes):
    """Subunit nodes v for which there is a directed path v -> ... -> unit_node in the original graph."""
    from collections import deque
    in_edges = {}
    for a, b in original_edges:
        in_edges.setdefault(b, set()).add(a)
    seen = set()
    queue = deque([unit_node])
    while queue:
        n = queue.popleft()
        if n in seen:
            continue
        seen.add(n)
        for p in in_edges.get(n, ()):
            if p not in seen:
                queue.append(p)
    return seen & subunit_nodes


def collapse(HSCMParametric: HSCMParametric) -> CausalGraphicalModel:
    """
    Collapse a hierarchical structural causal model (HSCM) into a non-hierarchical CGM.

    This function takes an HSCMParametric object and collapses its hierarchical structure,
    resulting in a collapsed flat causal graphical model that retains the causal relationships of the original model.
    Q-node names follow paper.tex: Q^v when subunit v has no subunit-level parents,
    and Q^{v|pa_S(v)} otherwise (e.g. Q^a, Q^{y|a}, Q^z, Q^{a|z}).
    Follows paper Alg. 1: unit parents -> Q, Q -> unit descendants (and Q -> Q when subunit -> subunit).

    Parameters
    ----------
    HSCMParametric : HSCMParametric
        The hierarchical structural causal model to be collapsed.

    Returns
    -------
    CausalGraphicalModel
        A collapsed flat causal graphical model.
    """
    nodes = HSCMParametric.unit_nodes.copy()
    edges = HSCMParametric.edges.copy()
    subunit_nodes = HSCMParametric.subunit_nodes
    unit_nodes = HSCMParametric.unit_nodes
    original_edges = HSCMParametric.edges
    subunit_to_q = {}
    for subunit in sorted(subunit_nodes):
        q_node = _q_node_name_paper(subunit, original_edges, subunit_nodes)
        subunit_to_q[subunit] = q_node
        nodes.add(q_node)
        for parent, child in list(HSCMParametric.edges):
            if child == subunit and parent in nodes:
                edges.discard((parent, child))
                edges.add((parent, q_node))
            elif parent == subunit and child in nodes:
                edges.discard((parent, child))
                edges.add((q_node, child))
            elif parent == subunit:
                edges.discard((parent, child))  # subunit->subunit: no Q->Q edge (paper collapsed has Q^{a|z} ~ pr(·|u) only)
        collapsed_model = CausalGraphicalModel(nodes=list(nodes), edges=list(edges))
    # paper Alg. 1: connect Q^{v|pa_S(v)} to X^w for each unit descendant w of v
    for w in unit_nodes:
        for v in _subunit_ancestors_of_unit(w, original_edges, subunit_nodes, unit_nodes):
            edges.add((subunit_to_q[v], w))
    
    # keeping observed variables
    temp_cgm = CausalGraphicalModel(nodes=list(nodes), edges=list(edges))
    temp_cgm.observed_variables = HSCMParametric.cgm.observed_variables
    temp_cgm.unobserved_variables = HSCMParametric.cgm.unobserved_variables
    return temp_cgm


def _subunit_ancestors_of_subunit(node: str, edges: set, subunit_nodes: set) -> set:
    """Subunit nodes that are ancestors of the given subunit node (including itself)."""
    from collections import deque
    in_edges = {}
    for a, b in edges:
        in_edges.setdefault(b, set()).add(a)
    seen = set()
    queue = deque([node])
    while queue:
        n = queue.popleft()
        if n in seen or n not in subunit_nodes:
            continue
        seen.add(n)
        for p in in_edges.get(n, ()):
            if p not in seen:
                queue.append(p)
    return seen


def suggest_augment_for_outcome(hscm: HSCMParametric, outcome_subunit: str) -> tuple[str, set[str]]:
    """
    Suggest augmentation (q_hat, parents) so that the estimand E[outcome | do(...)] can be identified.

    Paper: the estimand for subunit outcome Y involves the within-unit marginal Q^y. That marginal
    is a deterministic function of the conditionals for Y and its subunit ancestors (eq. 4140 with
    L = {Y}, R = empty). So we augment with Q^y with parents = { Q^{v|pa_S(v)} : v in an_S(Y) }.
    The augmentation is observed iff all those Q nodes are observed (Alg 2: computable from
    q(x^{S_obs})).

    Parameters
    ----------
    hscm : HSCMParametric
        The hierarchical SCM (used for edges and subunit_nodes).
    outcome_subunit : str
        Name of the subunit outcome variable (e.g. "Y").

    Returns
    -------
    q_hat : str
        Augmentation variable name (e.g. "Q^y").
    q_hat_parents : set[str]
        Parent Q-node names (e.g. {"Q^a", "Q^{y|a}"}).
    """
    edges = set(hscm.edges)
    subunit_nodes = set(hscm.subunit_nodes)
    names_no_prefix = getattr(hscm, "subunit_nodes_names", set())
    # HSCMParametric uses "_" prefix for subunit nodes; accept "Y" or "_Y"
    if outcome_subunit in subunit_nodes:
        out = outcome_subunit
    elif outcome_subunit in names_no_prefix:
        out = "_" + outcome_subunit.lstrip("_")
    else:
        raise ValueError("outcome_subunit must be a subunit node (e.g. 'Y' or '_Y'): {}".format(outcome_subunit))
    if out not in subunit_nodes:
        raise ValueError("outcome_subunit must be a subunit node: {}".format(outcome_subunit))
    an_s = _subunit_ancestors_of_subunit(out, edges, subunit_nodes)
    parents = set()
    for v in an_s:
        q_node = _q_node_name_paper(v, edges, subunit_nodes)
        parents.add(q_node)
    # q_hat = marginal of outcome -> Q^{outcome}; paper notation Q^y for variable Y
    outcome_lower = out.lstrip("_").lower()
    q_hat = "Q^{}".format(outcome_lower)
    return q_hat, parents


def augment_collapsed_model(hscm_collapsed, q_hat, q_hat_parents) -> CausalGraphicalModel:
    """
    Augment a collapsed causal graphical model with a new node (paper Step 2).

    we add ``q_hat`` as a deterministic function of ``q_hat_parents`` only: we add edges
    parent -> q_hat for each parent in ``q_hat_parents``. we do **not** remove or replace
    other outgoing edges from those parents (paper fig. augment_interfere keeps e.g. Q^a -> Z).

    Parameters
    ----------
    hscm_collapsed : CausalGraphicalModel
        The collapsed causal graphical model to be augmented.
    q_hat : str
        The name of the new node (format: Q^{v} or Q^{v|p1|p2|...} per _q_node_name_paper).
    q_hat_parents : set
        The set of parent nodes for the new node.

    Returns
    -------
    CausalGraphicalModel
        The augmented causal graphical model.
    """

    collapsed_model = copy.deepcopy(hscm_collapsed)
    node = q_hat
    collapsed_model.add_node(node)

    # checking if f(q_hat) can be computed from the data
    can_be_computed = True
    for parent in q_hat_parents:
        if parent not in collapsed_model.observed_variables:
            can_be_computed = False
            break
    if can_be_computed:
        collapsed_model.observed_variables.add(q_hat)  # mark q as observed

    for parent in q_hat_parents:  # deterministic q_hat = m(parents); paper double-arrow
        collapsed_model.add_edge(parent, node)

    return collapsed_model


def marginalize_augmented_model(augmented_collapsed_model, q_hat, q_hat_special_parents) -> CausalGraphicalModel:
    """
    Marginalize out the augmented node from the collapsed model.

    This function removes the augmented node from the model and updates the edges and mechanisms
    to reflect the marginalization.

    Parameters
    ----------
    CausalGraphicalModel : augmented_collapsed_model
        The augmented collapsed causal graphical model to be marginalized.
    str : q_hat
        The name of the node to be marginalized out.
    set : q_hat_special_parents
        The set of special parent nodes that may require additional handling during marginalization.

    Returns
    -------
    CausalGraphicalModel
        The marginalized causal graphical model.
    """
    temp_cgm = copy.deepcopy(augmented_collapsed_model)
    for variable in q_hat_special_parents:
        for parent, child in list(temp_cgm.dag.edges):
            if child == variable:
                temp_cgm.remove_edge(parent, child)
                if parent != q_hat:  # avoid self-loop when parent is q_hat (edge from augment)
                    temp_cgm.add_edge(parent, q_hat)
        temp_cgm.remove_node(variable)
    marginalized_augmented_model = CausalGraphicalModel(nodes=list(temp_cgm.dag.nodes), edges=list(temp_cgm.dag.edges))
    return marginalized_augmented_model


__all__ = [
    "PYAGNUM_AVAILABLE",
    "DoCalculusResult",
    "cgm_to_pyagrum_causal",
    "run_do_calculus",
    "identify_effect",
    "collapse",
    "augment_collapsed_model",
    "marginalize_augmented_model",
    "suggest_augment_for_outcome",
]
