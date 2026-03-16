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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear hierarchical causal model: estimate vs reality
#
# Generate data from a **linear** confounder HCM (Gaussian unit/subunit noise), compute the **true ATE** in closed form, and compare with the **per-unit linear regression** estimator (paper-style backdoor via units).

# %%
import numpy as np
from linear_hcm_data import (
    LinearConfounderHCM,
    default_linear_confounder_params,
    estimate_ate_linear_per_unit,
)

# %%
rng = np.random.default_rng(42)
params = default_linear_confounder_params(beta_a=1.0, beta_u=0.8, gamma_u=0.6)
model = LinearConfounderHCM(**params)
true_ate = model.true_ate()
print("True ATE (closed form):", round(true_ate, 4))

# %%
n, m = 200, 100
U, A, Y = model.sample(n, m, rng)
est_ate = estimate_ate_linear_per_unit(A, Y)
print("Estimated ATE (per-unit linear):", round(est_ate, 4))
print("|est - true|:", round(abs(est_ate - true_ate), 4))

# %%
print("Multi-seed (n=200, m=100, 20 seeds):")
true_ate = model.true_ate()
errors = []
for seed in range(20):
    U, A, Y = model.sample(n, m, np.random.default_rng(seed))
    est = estimate_ate_linear_per_unit(A, Y)
    errors.append(est - true_ate)
errors = np.array(errors)
print("  bias (mean est - true):", round(float(np.mean(errors)), 4))
print("  RMSE:", round(float(np.sqrt(np.mean(errors**2))), 4))
