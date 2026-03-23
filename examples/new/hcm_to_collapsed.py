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
# I'll just write mechanisms but we can get it from HSCMParametric.functions by restraining to unit_level variables
# we can represent each mechanism as a dictionnary containing the function for sampling, the parents in the order the function should take them and finally a string expressing whether or not another function is present in there

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



# %% [markdown]
# Testing the augmented model

# %%
# we suppose we already infered that we want to augment the model with q_hat = Q^y
augmented_confounder_cgm = augment_collapsed_model(confounder_cgm, 'Q^y', {'Q^{y|a}',"Q^a"})
print(augmented_confounder_cgm.dag.nodes)
print(augmented_confounder_cgm.dag.edges)


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
assert ("Q^z", "Q^a") in _edges(augmented_instrument_cgm)
assert ("Q^{a|z}", "Q^a") in _edges(augmented_instrument_cgm)
assert ("Q^{a|z}", "Y") in _edges(augmented_instrument_cgm)
assert ("Q^z", "Y") in _edges(augmented_instrument_cgm)
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
