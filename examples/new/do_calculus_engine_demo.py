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

# %% [markdown]
# # Do-calculus engine demo
#
# Run the engine (examples/new/do_calculus.py) on confounder, instrument, and nonID
# examples. Software tests are in `tests/`.

# %%
import do_calculus as dc
from hierarchicalcausalmodels.models import HSCMParametric

def _empty_fun(*args, **kwargs):
    return None

# %%
if not dc.PYAGNUM_AVAILABLE:
    print("pyagrum not installed; skip do-calculus demos.")
else:
    # Confounder
    h_conf = HSCMParametric(
        nodes={"U", "A", "Y"}, edges={("U", "A"), ("U", "Y"), ("A", "Y")},
        unit_nodes={"U"}, subunit_nodes={"A", "Y"},
        sizes=[3], node_functions={"U": _empty_fun, "A": _empty_fun, "Y": _empty_fun}, data={},
    )
    conf_cgm = dc.collapse(h_conf)
    aug_conf = dc.augment_collapsed_model(conf_cgm, "Q^y", {"Q^{y|a}", "Q^a"})
    aug_conf.unobserved_variables = {"U"}
    res_conf = dc.identify_effect(aug_conf, Y="Q^y", X="Q^a", unobserved={"U"})
    print("Confounder: identifiable =", res_conf.identifiable)
    if res_conf.formula_latex:
        print("  Formula:", res_conf.formula_latex[:80], "...")

# %%
# All graphs: full LaTeX formula (rendered as math) and pyAgrum AST
import re
from IPython.display import display, Math, Markdown
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, build_cgm_for_case

def _latex_for_katex(s):
    """Escape underscores in variable names so KaTeX does not parse them as double subscript."""
    return re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", r"\\_", s)

if dc.PYAGNUM_AVAILABLE:
    for case in COLLAPSED_DO_CALCULUS_CASES:
        name = case[0]
        cgm, unobs, Y_var, X_var, _ = build_cgm_for_case(dc, case)
        display(Markdown("---\n### **{}**  \nOutcome \\(Y\\): {}, Intervention \\(X\\): {}".format(name, Y_var, X_var)))
        cgm.unobserved_variables = unobs
        res = dc.identify_effect(cgm, Y=Y_var, X=X_var, unobserved=unobs)
        if res.identifiable and res.formula_latex:
            display(Math(_latex_for_katex(res.formula_latex)))
            if res.ast is not None:
                print("AST:", res.ast)
        else:
            print("Not identified or error:", (res.error or res.explanation or "unknown")[:200])
        print()

# %%
if dc.PYAGNUM_AVAILABLE:
    # Instrument
    h_inst = HSCMParametric(
        nodes={"U", "Y", "Z", "A"}, edges={("U", "A"), ("U", "Y"), ("Z", "A"), ("A", "Y")},
        unit_nodes={"U", "Y"}, subunit_nodes={"Z", "A"},
        sizes=[3], node_functions={n: _empty_fun for n in ["U", "Y", "Z", "A"]}, data={},
    )
    inst_cgm = dc.collapse(h_inst)
    inst_cgm = dc.augment_collapsed_model(inst_cgm, "Q^a", {"Q^z", "Q^{a|z}"})
    inst_cgm = dc.marginalize_augmented_model(inst_cgm, "Q^a", {"Q^z"})
    inst_cgm.unobserved_variables = {"U"}
    res_inst = dc.identify_effect(inst_cgm, Y="Y", X="Q^a", unobserved={"U"})
    print("Instrument: identifiable =", res_inst.identifiable)
    if res_inst.formula_latex:
        print("  Formula:", res_inst.formula_latex[:80], "...")

# %%
if dc.PYAGNUM_AVAILABLE:
    # NonID
    h_nonid = HSCMParametric(
        nodes={"U", "A", "W", "Y"}, edges={("U", "A"), ("U", "W"), ("A", "W"), ("A", "Y"), ("W", "Y")},
        unit_nodes={"U", "W"}, subunit_nodes={"A", "Y"},
        sizes=[3], node_functions={n: _empty_fun for n in ["U", "A", "W", "Y"]}, data={},
    )
    nonid_cgm = dc.collapse(h_nonid)
    nonid_aug = dc.augment_collapsed_model(nonid_cgm, "Q^y", {"Q^a", "Q^{y|a}"})
    nonid_aug.unobserved_variables = {"U"}
    res_nonid = dc.identify_effect(nonid_aug, Y="Q^y", X="Q^a", unobserved={"U"})
    print("nonID_ex1: identifiable =", res_nonid.identifiable)
    if res_nonid.error:
        print("  Error:", res_nonid.error[:60], "...")
