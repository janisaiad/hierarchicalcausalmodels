# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from hierarchicalcausalmodels.models import HSCMParametric
from causalgraphicalmodels import CausalGraphicalModel
import copy
import re

try:
    import pyagrum as gum
    import pyagrum.causal as csl
    PYAGNUM_AVAILABLE = True
except ImportError:
    PYAGNUM_AVAILABLE = False


# %%
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


# %%
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



# %%
# Build three example HSCMs (confounder, confounder + interferer, instrument)
# and collapse each using the collapse() function.

def _empty_fun(*args, **kwargs):
    return None

# (a) Confounder: U -> A, U -> Y, A -> Y (A, Y are subunit-level)
hscm_confounder = HSCMParametric(
    nodes={"U", "A", "Y"},
    edges={("U", "A"), ("U", "Y"), ("A", "Y")},
    unit_nodes={"U"},
    subunit_nodes={"A", "Y"},
    sizes=[3],
    node_functions={"U": _empty_fun, "A": _empty_fun, "Y": _empty_fun},
    data={},
)
confounder_cgm = collapse(hscm_confounder)
print("Confounder CGM nodes:", confounder_cgm.dag.nodes)
print("Confounder CGM edges:", confounder_cgm.dag.edges)

# (b) Confounder & Interferer: U -> A, U -> Y, A -> Y, A -> Z, Z -> Y
# A, Y are subunit-level; Z, U are unit-level.
hscm_confounder_interferer = HSCMParametric(
    nodes={"U", "Z", "A", "Y"},
    edges={("U", "A"), ("U", "Y"), ("A", "Y"), ("A", "Z"), ("Z", "Y")},
    unit_nodes={"U", "Z"},
    subunit_nodes={"A", "Y"},
    sizes=[3],
    node_functions={"U": _empty_fun, "Z": _empty_fun, "A": _empty_fun, "Y": _empty_fun},
    data={},
)
confounder_interferer_cgm = collapse(hscm_confounder_interferer)
print("Confounder Interferer CGM nodes:", confounder_interferer_cgm.dag.nodes)
print("Confounder Interferer CGM edges:", confounder_interferer_cgm.dag.edges)


# (c) Instrument: U -> A, U -> Y, Z -> A, A -> Y
# Z, A are subunit-level; U, Y are unit-level.
hscm_instrument = HSCMParametric(
    nodes={"U", "Y", "Z", "A"},
    edges={("U", "A"), ("U", "Y"), ("Z", "A"), ("A", "Y")},
    unit_nodes={"U", "Y"},
    subunit_nodes={"Z", "A"},
    sizes=[3],
    node_functions={"U": _empty_fun, "Y": _empty_fun, "Z": _empty_fun, "A": _empty_fun},
    data={},
)
instrument_cgm = collapse(hscm_instrument)
print("Instrument CGM nodes:", instrument_cgm.dag.nodes)
print("Instrument CGM edges:", instrument_cgm.dag.edges) 



# %%
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


# %%
# I'll just write mechanisms but we can get it from HSCMParametric.functions by restraining to unit_level variables
# we can represent each mechanism as a dictionnary containing the function for sampling, the parents in the order the function should take them and finally a string expressing whether or not another function is present in there

def augment_collapsed_model(hscm_collapsed, q_hat, q_hat_parents) -> CausalGraphicalModel:
    """
    Augment a collapsed causal graphical model with a new node representing a subunit variable.

    This function adds a new node to the collapsed model, representing a subunit variable
    conditioned on its parents. It also updates the edges and mechanisms accordingly.

    Parameters
    ----------
    CausalGraphicalModel : collapsed_model
        The collapsed causal graphical model to be augmented.
    str : q_hat
        The name of the new node (format: Q^{v} or Q^{v|p1|p2|...} per _q_node_name_paper).
    set : q_hat_parents
        The set of parent nodes for the new node.
    function : q_hat_expr
        The functional expression defining the new node.
    dict : mechanisms
        A dictionary of mechanisms for the nodes in the model.

    Returns
    -------
    CausalGraphicalModel
        The augmented causal graphical model.
    """
    
    collapsed_model=copy.deepcopy(hscm_collapsed)
    node = q_hat
    collapsed_model.add_node(node)
    
    # we suppose we already have the parents of q_hat that are q_variables
    
    # we extract the subunit variable from q_hat (format Q^{v} or Q^{v|p1|p2|...})
    match = re.search(r"\^\{?([a-zA-Z0-9_]+)(?:\||\})?", q_hat)
    variable = match.group(1) if match else None

    # checking if f(q_hat) can be computed from the data
    can_be_computed = True
    for parent in q_hat_parents:
        if parent not in collapsed_model.observed_variables:
            can_be_computed = False
            break
    if can_be_computed:
        collapsed_model.observed_variables.add(q_hat) # mark q as observed
        
        
    for parent in q_hat_parents: # double arrow
        collapsed_model.add_edge(parent, node)
    # we redirect edges from q_hat_parents to their non-q_hat children: parent->child becomes q_hat->child
    # we skip edges parent->q_hat (the ones we just added) to avoid self-loops
    for parent, child in list(collapsed_model.dag.edges):
        if parent in q_hat_parents and child != q_hat:
            collapsed_model.remove_edge(parent, child)
            collapsed_model.add_edge(q_hat, child)
                    
    return collapsed_model

    
    

# %% [markdown]
# Testing the augmented model

# %%
# we suppose we already infered that we want to augment the model with q_hat = Q^y
augmented_confounder_cgm = augment_collapsed_model(confounder_cgm, 'Q^y', {'Q^{y|a}',"Q^a"})
print(augmented_confounder_cgm.dag.nodes)
print(augmented_confounder_cgm.dag.edges)

# %%
# Do-calculus via pyAgrum: set U as unobserved, then identify P(Q^y | do(Q^a))
if PYAGNUM_AVAILABLE:
    augmented_confounder_cgm.unobserved_variables = {"U"}
    ast_confounder = run_do_calculus(augmented_confounder_cgm, Y={"Q^y"}, X={"Q^a"}, unobserved={"U"})
    print("Confounder: P(Q^y | do(Q^a)) identified as:")
    print(ast_confounder.toLatex())

# %%
augmented_cofounder_interferer_cgm = augment_collapsed_model(confounder_interferer_cgm, 'Q^y', {'Q^{y|a}',"Q^a"})
print(augmented_cofounder_interferer_cgm.dag.nodes)
print(augmented_cofounder_interferer_cgm.dag.edges)

# %%
augmented_instrument_cgm = augment_collapsed_model(instrument_cgm, 'Q^y', {'Q^{y|a}',"Q^a"})
print(augmented_instrument_cgm.dag.nodes)
print(augmented_instrument_cgm.dag.edges)


# %%
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


# %%
# (c) Instrument: U -> A, U -> Y, Z -> A, A -> Y
# Z, A are subunit-level; U, Y are unit-level.
hscm_instrument = HSCMParametric(
    nodes={"U", "Y", "Z", "A"},
    edges={("U", "A"), ("U", "Y"), ("Z", "A"), ("A", "Y")},
    unit_nodes={"U", "Y"},
    subunit_nodes={"Z", "A"},
    sizes=[3],
    node_functions={"U": _empty_fun, "Y": _empty_fun, "Z": _empty_fun, "A": _empty_fun},
    data={},
)

collapsed_instrument_cgm = collapse(hscm_instrument)
augmented_instrument_cgm = augment_collapsed_model(collapsed_instrument_cgm, 'Q^a', {'Q^z','Q^{a|z}'}) # we deduced Q^y|a from the data	
print("Instrument CGM nodes:", augmented_instrument_cgm.dag.nodes)
print("Instrument CGM edges:", augmented_instrument_cgm.dag.edges) 



# %%
instrument_cgm = marginalize_augmented_model(augmented_instrument_cgm,'Y',{'Q^z'})
print("Instrument CGM nodes:", instrument_cgm.dag.nodes)
print("Instrument CGM edges:", instrument_cgm.dag.edges)

# %%
# Do-calculus on instrument (marginalized) model: P(Y | do(Q^a))
if PYAGNUM_AVAILABLE:
    instrument_cgm.unobserved_variables = {"U"}
    ast_instrument = run_do_calculus(instrument_cgm, Y={"Y"}, X={"Q^a"}, unobserved={"U"})
    print("Instrument: P(Y | do(Q^a)) identified as:")
    print(ast_instrument.toLatex())
    # Optional: causalImpact returns (formula, tensor, explanation); use method="causalImpact" for numeric eval
    # formula, impact_tensor, expl = run_do_calculus(augmented_confounder_cgm, "Q^y", "Q^a", unobserved={"U"}, method="causalImpact")

# %% [markdown]
# ## Figure A3: Additional ID examples (effect of A on Y identified)
#
# Examples from paper Fig. A3 – each would not be identifiable if inner plate were erased.

# %%
# ID_ex3 (paper Fig. A3): U->A, U->W, A->W, A->Y, W->Y. Subunit: A,W. Unit: U,Y.
# Collapse only – no augment needed for identification (condition 1: no bi-directed path).
hscm_id_ex3 = HSCMParametric(
    nodes={"U", "A", "W", "Y"},
    edges={("U", "A"), ("U", "W"), ("A", "W"), ("A", "Y"), ("W", "Y")},
    unit_nodes={"U", "Y"},
    subunit_nodes={"A", "W"},
    sizes=[3],
    node_functions={"U": _empty_fun, "A": _empty_fun, "W": _empty_fun, "Y": _empty_fun},
    data={},
)
id_ex3_cgm = collapse(hscm_id_ex3)
print("ID_ex3 (collapse) nodes:", sorted(id_ex3_cgm.dag.nodes))
print("ID_ex3 (collapse) edges:", sorted(id_ex3_cgm.dag.edges))

# %%
# ID_ex2 (paper Fig. A3): U->A,Z,Y; A->Z,Y; Z->Y. All A,Z,Y subunit. Unit: U.
# Augment with Q^y, parents Q^a, Q^{z|a}, Q^{y|z,a}.
hscm_id_ex2 = HSCMParametric(
    nodes={"U", "A", "Z", "Y"},
    edges={("U", "A"), ("U", "Z"), ("U", "Y"), ("A", "Z"), ("A", "Y"), ("Z", "Y")},
    unit_nodes={"U"},
    subunit_nodes={"A", "Z", "Y"},
    sizes=[3],
    node_functions={"U": _empty_fun, "A": _empty_fun, "Z": _empty_fun, "Y": _empty_fun},
    data={},
)
id_ex2_cgm = collapse(hscm_id_ex2)
id_ex2_aug = augment_collapsed_model(id_ex2_cgm, "Q^y", {"Q^a", "Q^{z|a}", "Q^{y|a|z}"})
print("ID_ex2 (collapse) nodes:", sorted(id_ex2_cgm.dag.nodes))
print("ID_ex2 (collapse) edges:", sorted(id_ex2_cgm.dag.edges))
print("ID_ex2 (augment Q^y) nodes:", sorted(id_ex2_aug.dag.nodes))
print("ID_ex2 (augment Q^y) edges:", sorted(id_ex2_aug.dag.edges))

# %%
# ID_ex6 (paper Fig. A3): A,U->Z; U->A; A->W; W->Z; A->Y; W->Y; Z->Y; U'->W,Y. Subunit: A,Z. Unit: U,W,Y,Up.
hscm_id_ex6 = HSCMParametric(
    nodes={"A", "Z", "U", "W", "Y", "Up"},
    edges={("A", "Z"), ("U", "Z"), ("U", "A"), ("A", "W"), ("W", "Z"), ("A", "Y"), ("W", "Y"), ("Z", "Y"), ("Up", "W"), ("Up", "Y")},
    unit_nodes={"U", "W", "Y", "Up"},
    subunit_nodes={"A", "Z"},
    sizes=[3],
    node_functions={"A": _empty_fun, "Z": _empty_fun, "U": _empty_fun, "W": _empty_fun, "Y": _empty_fun, "Up": _empty_fun},
    data={},
)
id_ex6_cgm = collapse(hscm_id_ex6)
print("ID_ex6 (collapse) nodes:", sorted(id_ex6_cgm.dag.nodes))
print("ID_ex6 (collapse) edges:", sorted(id_ex6_cgm.dag.edges))

# %%
# ID_ex4 (paper Fig. A3): Z->A, U->A,W,Y, A->W,Y, W->Y. Subunit: A,Z,Y. Unit: U,W.
# Augment with Q^a (parents Q^z, Q^{a|z}) – second augment Q^y can create cycles, skip marginalize.
hscm_id_ex4 = HSCMParametric(
    nodes={"Z", "A", "U", "W", "Y"},
    edges={("Z", "A"), ("U", "A"), ("U", "W"), ("U", "Y"), ("A", "W"), ("A", "Y"), ("W", "Y")},
    unit_nodes={"U", "W"},
    subunit_nodes={"Z", "A", "Y"},
    sizes=[3],
    node_functions={"Z": _empty_fun, "A": _empty_fun, "U": _empty_fun, "W": _empty_fun, "Y": _empty_fun},
    data={},
)
id_ex4_cgm = collapse(hscm_id_ex4)
id_ex4_aug = augment_collapsed_model(id_ex4_cgm, "Q^a", {"Q^z", "Q^{a|z}"})
print("ID_ex4 (collapse) nodes:", sorted(id_ex4_cgm.dag.nodes))
print("ID_ex4 (collapse) edges:", sorted(id_ex4_cgm.dag.edges))
print("ID_ex4 (augment Q^a) nodes:", sorted(id_ex4_aug.dag.nodes))
print("ID_ex4 (augment Q^a) edges:", sorted(id_ex4_aug.dag.edges))

# %%
# ID_ex1 (paper Fig. A3): Z->W, U->A,W,Y, A->W, W->Y. Subunit: A,W,Z. Unit: U,Y. Augment Q^w, marginalize Q^z.
hscm_id_ex1 = HSCMParametric(
    nodes={"Z", "W", "A", "U", "Y"},
    edges={("Z", "W"), ("U", "A"), ("U", "W"), ("U", "Y"), ("A", "W"), ("W", "Y")},
    unit_nodes={"U", "Y"},
    subunit_nodes={"Z", "A", "W"},
    sizes=[3],
    node_functions={"Z": _empty_fun, "W": _empty_fun, "A": _empty_fun, "U": _empty_fun, "Y": _empty_fun},
    data={},
)
id_ex1_cgm = collapse(hscm_id_ex1)
id_ex1_aug = augment_collapsed_model(id_ex1_cgm, "Q^w", {"Q^a", "Q^{w|a|z}", "Q^z"})
id_ex1_mar = marginalize_augmented_model(id_ex1_aug, "Q^w", {"Q^z"})
print("ID_ex1 (collapse) nodes:", sorted(id_ex1_cgm.dag.nodes))
print("ID_ex1 (collapse) edges:", sorted(id_ex1_cgm.dag.edges))
print("ID_ex1 (augment Q^w) nodes:", sorted(id_ex1_aug.dag.nodes))
print("ID_ex1 (augment Q^w) edges:", sorted(id_ex1_aug.dag.edges))
print("ID_ex1 (marginalize Q^z) nodes:", sorted(id_ex1_mar.dag.nodes))
print("ID_ex1 (marginalize Q^z) edges:", sorted(id_ex1_mar.dag.edges))

# %%
# ID_ex5 (paper Fig. A3): Z->A, U->A,X,Y, X->A,Y, A->Y. Subunit: Z,A,X. Unit: U,Y. Augment with Q^{a|x}, parents Q^z, Q^{a|z,x}.
hscm_id_ex5 = HSCMParametric(
    nodes={"Z", "A", "X", "Y", "U"},
    edges={("Z", "A"), ("U", "A"), ("U", "X"), ("U", "Y"), ("X", "A"), ("X", "Y"), ("A", "Y")},
    unit_nodes={"U", "Y"},
    subunit_nodes={"Z", "A", "X"},
    sizes=[3],
    node_functions={"Z": _empty_fun, "A": _empty_fun, "X": _empty_fun, "Y": _empty_fun, "U": _empty_fun},
    data={},
)
id_ex5_cgm = collapse(hscm_id_ex5)
id_ex5_aug = augment_collapsed_model(id_ex5_cgm, "Q^{a|x}", {"Q^z", "Q^{a|x|z}"})
id_ex5_mar = marginalize_augmented_model(id_ex5_aug, "Q^{a|x}", {"Q^z"})
print("ID_ex5 (collapse) nodes:", sorted(id_ex5_cgm.dag.nodes))
print("ID_ex5 (collapse) edges:", sorted(id_ex5_cgm.dag.edges))
print("ID_ex5 (augment Q^{a|x}) nodes:", sorted(id_ex5_aug.dag.nodes))
print("ID_ex5 (augment Q^{a|x}) edges:", sorted(id_ex5_aug.dag.edges))
print("ID_ex5 (marginalize Q^z) nodes:", sorted(id_ex5_mar.dag.nodes))
print("ID_ex5 (marginalize Q^z) edges:", sorted(id_ex5_mar.dag.edges))

# %%
# ID_targeted (paper Fig. A3, fig:hcm_obs_confound): A,X,Y subunit; U,Z unit. U->A,X,Y; X->A,Y,Z; A->Z,Y; Z->Y.
# Conditional soft intervention on A|X. Augment with Q^y (parents Q^x, Q^{a|x}, Q^{y|a|x}).
hscm_id_targeted = HSCMParametric(
    nodes={"A", "X", "Y", "U", "Z"},
    edges={("U", "A"), ("U", "Y"), ("U", "X"), ("A", "Z"), ("A", "Y"), ("Z", "Y"), ("X", "A"), ("X", "Y"), ("X", "Z")},
    unit_nodes={"U", "Z"},
    subunit_nodes={"A", "X", "Y"},
    sizes=[3],
    node_functions={"A": _empty_fun, "X": _empty_fun, "Y": _empty_fun, "U": _empty_fun, "Z": _empty_fun},
    data={},
)
id_targeted_cgm = collapse(hscm_id_targeted)
id_targeted_aug = augment_collapsed_model(id_targeted_cgm, "Q^y", {"Q^x", "Q^{a|x}", "Q^{y|a|x}"})
print("ID_targeted (collapse) nodes:", sorted(id_targeted_cgm.dag.nodes))
print("ID_targeted (collapse) edges:", sorted(id_targeted_cgm.dag.edges))
print("ID_targeted (augment Q^y) nodes:", sorted(id_targeted_aug.dag.nodes))
print("ID_targeted (augment Q^y) edges:", sorted(id_targeted_aug.dag.edges))

# %% [markdown]
# ## Figure A4: Non-ID examples (effect of A on Y not identified)
#
# Examples from paper Fig. A4 – effect not identified via the collapse/augment/marginalize method.

# %%
# nonID_ex5 (paper Fig. A4): U->A, U->Y, A->Y, A->Z, Z->Y, U->Z. Subunit: A,Y. Unit: U,Z.
hscm_nonid_ex5 = HSCMParametric(
    nodes={"U", "A", "Y", "Z"},
    edges={("U", "A"), ("U", "Y"), ("A", "Y"), ("A", "Z"), ("Z", "Y"), ("U", "Z")},
    unit_nodes={"U", "Z"},
    subunit_nodes={"A", "Y"},
    sizes=[3],
    node_functions={"U": _empty_fun, "A": _empty_fun, "Y": _empty_fun, "Z": _empty_fun},
    data={},
)
nonid_ex5_cgm = collapse(hscm_nonid_ex5)
nonid_ex5_aug = augment_collapsed_model(nonid_ex5_cgm, "Q^y", {"Q^a", "Q^{y|a}"})
print("nonID_ex5 (collapse) nodes:", sorted(nonid_ex5_cgm.dag.nodes))
print("nonID_ex5 (collapse) edges:", sorted(nonid_ex5_cgm.dag.edges))
print("nonID_ex5 (augment Q^y) nodes:", sorted(nonid_ex5_aug.dag.nodes))
print("nonID_ex5 (augment Q^y) edges:", sorted(nonid_ex5_aug.dag.edges))

# %%
# nonID_ex4 (paper Fig. A4): U->A, U->Y, U'->A, U'->Z, Z->A, A->Y. Subunit: A,Z. Unit: U,Up,Y.
hscm_nonid_ex4 = HSCMParametric(
    nodes={"U", "Up", "A", "Z", "Y"},
    edges={("U", "A"), ("U", "Y"), ("Up", "A"), ("Up", "Z"), ("Z", "A"), ("A", "Y")},
    unit_nodes={"U", "Up", "Y"},
    subunit_nodes={"A", "Z"},
    sizes=[3],
    node_functions={"U": _empty_fun, "Up": _empty_fun, "A": _empty_fun, "Z": _empty_fun, "Y": _empty_fun},
    data={},
)
nonid_ex4_cgm = collapse(hscm_nonid_ex4)
nonid_ex4_aug = augment_collapsed_model(nonid_ex4_cgm, "Q^a", {"Q^z", "Q^{a|z}"})
print("nonID_ex4 (collapse) nodes:", sorted(nonid_ex4_cgm.dag.nodes))
print("nonID_ex4 (collapse) edges:", sorted(nonid_ex4_cgm.dag.edges))
print("nonID_ex4 (augment Q^a) nodes:", sorted(nonid_ex4_aug.dag.nodes))
print("nonID_ex4 (augment Q^a) edges:", sorted(nonid_ex4_aug.dag.edges))

# %%
# nonID_ex1 (paper Fig. A4): U->A, U->W, A->W, A->Y, W->Y. Subunit: A,Y. Unit: U,W.
hscm_nonid_ex1 = HSCMParametric(
    nodes={"U", "A", "W", "Y"},
    edges={("U", "A"), ("U", "W"), ("A", "W"), ("A", "Y"), ("W", "Y")},
    unit_nodes={"U", "W"},
    subunit_nodes={"A", "Y"},
    sizes=[3],
    node_functions={"U": _empty_fun, "A": _empty_fun, "W": _empty_fun, "Y": _empty_fun},
    data={},
)
nonid_ex1_cgm = collapse(hscm_nonid_ex1)
nonid_ex1_aug = augment_collapsed_model(nonid_ex1_cgm, "Q^y", {"Q^a", "Q^{y|a}"})
print("nonID_ex1 (collapse) nodes:", sorted(nonid_ex1_cgm.dag.nodes))
print("nonID_ex1 (collapse) edges:", sorted(nonid_ex1_cgm.dag.edges))
print("nonID_ex1 (augment Q^y) nodes:", sorted(nonid_ex1_aug.dag.nodes))
print("nonID_ex1 (augment Q^y) edges:", sorted(nonid_ex1_aug.dag.edges))


# %%
# Tests: collapse, augment, marginalize produce expected nodes and valid DAGs
def _nodes(cgm):
    return set(cgm.dag.nodes)

def _edges(cgm):
    return set(cgm.dag.edges)

# confounder: collapse + augment
assert _nodes(confounder_cgm) == {"U", "Q^a", "Q^{y|a}"}
assert _nodes(augmented_confounder_cgm) == {"U", "Q^a", "Q^{y|a}", "Q^y"}
assert ("Q^a", "Q^y") in _edges(augmented_confounder_cgm)
assert ("Q^{y|a}", "Q^y") in _edges(augmented_confounder_cgm)

# instrument: collapse + augment + marginalize
assert "Q^a" in _nodes(augmented_instrument_cgm)
assert ("Q^a", "Y") in _edges(augmented_instrument_cgm)
assert "Q^z" not in _nodes(instrument_cgm)
assert ("Q^{a|z}", "Q^a") in _edges(instrument_cgm)

# ID_ex3: collapse only
assert _nodes(id_ex3_cgm) == {"U", "Y", "Q^a", "Q^{w|a}"}

# ID_ex2: collapse + augment
assert "Q^y" in _nodes(id_ex2_aug)
assert all(p in _nodes(id_ex2_aug) for p in ["Q^a", "Q^{z|a}", "Q^{y|a|z}"])

# ID_ex5: collapse + augment + marginalize
assert "Q^{a|x}" in _nodes(id_ex5_aug)
assert "Q^z" not in _nodes(id_ex5_mar)

# ID_ex4: collapse + augment Q^a
assert "Q^a" in _nodes(id_ex4_aug)

# ID_ex1: collapse + augment Q^w + marginalize Q^z
assert "Q^w" in _nodes(id_ex1_aug)
assert "Q^z" not in _nodes(id_ex1_mar)

# nonID: collapse + augment (effect not identified, but graph ops should run)
assert "Q^y" in _nodes(nonid_ex5_aug)
assert "Q^a" in _nodes(nonid_ex4_aug)
assert "Q^y" in _nodes(nonid_ex1_aug)

print("All collapse/augment/marginalize tests passed.")
