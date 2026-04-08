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
# # Parallel per-unit estimators demo
#
# Fit one regressor per unit (sequential or parallel), aggregate for ATE. Device kwargs
# for multi-GPU; MLP on CPU. Software tests are in `tests/`.

# %%
import numpy as np
from hierarchicalcausalmodels.estimation import (
    fit_per_unit_estimators,
    aggregate_per_unit_outputs,
    fit_regressors_per_unit,
    estimate_ate_confounder,
    estimate_ate_confounder_torch_batched,
    device_kwargs_for_workers,
)

def _fit_unit_confounder_style(data):
    a, y = data
    match_0, match_1 = (a == 0), (a == 1)
    n0, n1 = max(int(match_0.sum()), 1), max(int(match_1.sum()), 1)
    mu_0 = (float(y[match_0].sum()) + 1.0) / (n0 + 2.0)
    mu_1 = (float(y[match_1].sum()) + 1.0) / (n1 + 2.0)
    return (mu_0, mu_1)

# %%
rng = np.random.default_rng(42)
n, m = 10, 30
A = rng.binomial(1, 0.5, (n, m))
Y = rng.binomial(1, 0.3 + 0.4 * A, (n, m))
unit_data = [(A[i], Y[i]) for i in range(n)]
fits = fit_per_unit_estimators(
    unit_data,
    _fit_unit_confounder_style,
    n_jobs=2,
    parallel_backend="threads",
)
ate = aggregate_per_unit_outputs(fits, "mean", extract_fn=lambda x: x[1] - x[0])
print("Per-unit fit + aggregate ATE:", round(ate, 4))

# %%
from sklearn.linear_model import LinearRegression
regs = fit_regressors_per_unit(A, Y, LinearRegression, n_jobs=2, parallel_backend="threads")
ate2 = estimate_ate_confounder(A, Y, LinearRegression, n_jobs=2, parallel_backend="threads")
print("fit_regressors_per_unit: {} regressors".format(len(regs)))
print("estimate_ate_confounder ATE:", round(ate2, 4))

# %%
ate_torch = estimate_ate_confounder_torch_batched(
    A,
    Y,
    family="bernoulli",
    device="cpu",
    max_iter=100,
)
print("estimate_ate_confounder_torch_batched (CPU) ATE:", round(ate_torch, 4))

# %%
kwargs_list = device_kwargs_for_workers(4, backend="torch", use_cuda=False)
print("device_kwargs_for_workers(torch, use_cuda=False):", kwargs_list)

# %%
kwargs_gpu = device_kwargs_for_workers(5, backend="torch", n_gpus=2, use_cuda=True)
print("device_kwargs_for_workers(torch, n_gpus=2):", [d["device"] for d in kwargs_gpu])

# %%
try:
    from hierarchicalcausalmodels.estimation.torch_estimators import MLPRegressorPerUnit
    regs_mlp = fit_regressors_per_unit(
        A, Y, MLPRegressorPerUnit, n_jobs=2, parallel_backend="threads",
        regressor_kwargs={"device": "cpu", "max_epochs": 10, "hidden_sizes": (8,)},
    )
    ate_mlp = estimate_ate_confounder(
        A,
        Y,
        MLPRegressorPerUnit,
        n_jobs=2,
        parallel_backend="threads",
        regressor_kwargs={"device": "cpu", "max_epochs": 10},
    )
    print("MLPRegressorPerUnit (CPU) ATE:", round(ate_mlp, 4))
except Exception as e:
    print("MLP demo skipped:", e)
