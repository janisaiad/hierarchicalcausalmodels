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
from __future__ import annotations

from hierarchicalcausalmodels.models import HSCMParametric
from causalgraphicalmodels import CausalGraphicalModel

from hierarchicalcausalmodels.do_calculus import (
    PYAGNUM_AVAILABLE,
    DoCalculusResult,
    cgm_to_pyagrum_causal,
    run_do_calculus,
    identify_effect,
    collapse,
    augment_collapsed_model,
    marginalize_augmented_model,
    suggest_augment_for_outcome,
    _sanitize_node_name,
    _q_node_name_paper,
    _subunit_ancestors_of_unit,
    _subunit_ancestors_of_subunit,
)


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
