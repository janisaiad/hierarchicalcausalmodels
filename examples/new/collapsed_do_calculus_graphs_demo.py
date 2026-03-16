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
# # Do-calculus on all collapsed graphs (from hcm_to_collapsed.ipynb)
#
# Build HSCM -> collapse -> (optional) augment -> (optional) marginalize; run
# identify_effect for each case and show a summary table. Software tests are in `tests/`.

# %%
import do_calculus as dc
from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, build_cgm_for_case

# %%
if not dc.PYAGNUM_AVAILABLE:
    print("pyagrum not installed.")
else:
    results = []
    for case in COLLAPSED_DO_CALCULUS_CASES:
        name = case[0]
        expected_id = case[10]
        cgm, unobserved, Y, X, _ = build_cgm_for_case(dc, case)
        res = dc.identify_effect(cgm, Y=Y, X=X, unobserved=unobserved)
        ok = res.identifiable == expected_id
        results.append((name, expected_id, res.identifiable, ok, res.formula_latex or res.error or ""))
    for name, exp, got, ok, detail in results:
        status = "ok" if ok else "MISMATCH"
        short = (detail[:60] + "...") if detail and len(detail) > 60 else (detail or "")
        print("{}: expected_id={} got={} {} | {}".format(name, exp, got, status, short))
