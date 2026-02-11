# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
    temp_cgm.observed_variables = HSCMParametric.observed_nodes
    temp_cgm.unobserved_variables = HSCMParametric.unobserved_nodes
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
        The name of the new node to be added.
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
    #q_hat = Q^a|y	
    node = q_hat
    collapsed_model.add_node(node)
    
    # we suppose we already have the parents of q_hat that are q_variables
    
    # Extract 'a' from q_hat (expected format 'Q^a|y')
    match = re.search(r"\^([a-zA-Z0-9_]+)\|", q_hat)
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
        collapsed_model.add_edge(node,parent)
        
    # for each children x we connect q_hat to X    
    for parent,children in collapsed_model.dag.edges:
        if parent == variable:  # if a is the variable we want to augment
            collapsed_model.add_edge(q_hat, children)
            for q_parent in q_hat_parents: # we suppose we already have the parents of q_hat
                if (q_parent, children) in collapsed_model.dag.edges:
                    collapsed_model.remove_edge(parent, unit)
    return collapsed_model

    
    

# %% [markdown]
# Testing the augmented model

# %%
# we suppose we already infered that we want to augment the model with q_hat = Q^y

augmented_confounder_cgm = augment_collapsed_model(confounder_cgm, 'Q^y', {'Q^y|a',"Q^a"})



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
    nodes = augmented_collapsed_model.dag.nodes.copy()
    edges = augmented_collapsed_model.dag.edges.copy()
    for variable in q_hat_special_parents:
        for  parent, child in augmented_collapsed_model.dag.edges:
            if child == variable :
                edges.remove((parent, child))
                edges.add((parent, q_hat))
        nodes.remove(variable)
    marginalized_augmented_model = CausalGraphicalModel(nodes=list(nodes), edges=list(edges))
    return marginalized_augmented_model


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
confounder_cgm = marginalize_augmented_model(hscm_confounder)
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
confounder_interferer_cgm = marginalize_augmented_model(hscm_confounder_interferer)
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
instrument_cgm = marginalize_augmented_model(hscm_instrument)
print("Instrument CGM nodes:", instrument_cgm.dag.nodes)
print("Instrument CGM edges:", instrument_cgm.dag.edges) 



# %%
