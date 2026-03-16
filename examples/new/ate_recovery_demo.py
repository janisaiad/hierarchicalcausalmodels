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
# # ATE recovery demo (confounder HCM)
#
# Paper DGP: binary U, A, Y; Beta-Bernoulli q^a and q^{y|a}. True ATE in closed form;
# HCM estimator = per-unit E[Y|A=a] with pseudocounts then average. Demos live here;
# software tests are in `tests/`.

# %%
import numpy as np
from confounder_model import ConfounderModel, default_confounder_params

# %%
rng = np.random.default_rng(42)
params = default_confounder_params(omega=0.2)
model = ConfounderModel(**params)
true_ate = model.true_ate()
print("True ATE (closed form):", round(true_ate, 4))

# %%
n, m = 600, 600
A, Y = model.sample(n, m, rng)
est_ate = model.estimate_ate(A, Y)
print("Estimated ATE:", round(est_ate, 4))
print("|est - true|:", round(abs(est_ate - true_ate), 4))

# %%
print("Multi-seed (n=m=500, omega=0.3):")
params = default_confounder_params(omega=0.3)
model = ConfounderModel(**params)
true_ate = model.true_ate()
errors = []
for seed in [0, 1, 2, 3, 4]:
    A, Y = model.sample(500, 500, np.random.default_rng(seed))
    errors.append(abs(model.estimate_ate(A, Y) - true_ate))
print("  mean |est - true|:", round(float(np.mean(errors)), 4))

# %%
print("No confounding (omega=0):")
params = default_confounder_params(omega=0.0)
model = ConfounderModel(**params)
A, Y = model.sample(400, 400, rng)
print("  true ATE:", round(model.true_ate(), 4), "  est ATE:", round(model.estimate_ate(A, Y), 4))
