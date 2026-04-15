"""
HCM Causal Estimand Numerical Evaluation
==========================================

Parse a pyAgrum identification formula (AST) and evaluate it numerically
from observed data.

The main entry point is ``ast_to_estimator``, which:

1. Parses the symbolic identification formula returned by
   ``csl.identifyingIntervention()`` (or ``DoCalculusResult.ast``).
2. Discovers which conditional densities P(Y|X_1,...,X_k) appear in the formula.
3. Estimates those densities from observed data using a requested parametric
   family (``"gaussian"``, ``"bernoulli"``) or falls back to histogram /
   Gaussian KDE (``"nonparametric"``).
4. Evaluates the formula numerically via per-unit averaging, Monte Carlo, or
   exact discrete enumeration.

For Q-variables (unit-level random distributions) whose raw subunit-level data
are provided, per-unit conditional estimators are automatically built from the
within-unit observations — the approach used in Appendix D of the HCM paper.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np

from .numba_kernels import (
    expit_array,
    linear_predict_batch,
    logistic_positive_proba_batch,
)
from .device_defaults import resolve_torch_device_from_mapping
from .parallel import ParallelBackend, parallel_map
from .torch_estimators import (
    torch_fit_batched_beta,
    torch_compute_subunit_params,
    torch_conditional_expectations_per_unit,
    torch_fit_batched_bernoulli,
    torch_fit_batched_gaussian,
    torch_fit_batched_gamma,
    torch_fit_batched_poisson,
    torch_predict_batched_beta_mean,
    torch_predict_batched_bernoulli,
    torch_predict_batched_gaussian,
    torch_predict_batched_gamma_mean,
    torch_predict_batched_poisson,
)
from .variational_estimators import (
    VariationalConditionalState,
    fit_variational_conditional_estimator,
    variational_credible_interval,
    variational_density_samples,
    variational_mean_prediction,
)

try:
    from scipy import stats as _sp_stats
    SCIPY_AVAILABLE = True
except ImportError:
    _sp_stats = None  # type: ignore
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_MIXTURE_AVAILABLE = True
except ImportError:
    GaussianMixture = None  # type: ignore
    SKLEARN_MIXTURE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Low-level conditional density estimator
# ─────────────────────────────────────────────────────────────────────────────

#: All supported parametric family names (aliases are normalised in __init__).
SUPPORTED_FAMILIES = {
    # Binary / count
    "bernoulli",
    "poisson",
    # Continuous unbounded
    "gaussian", "normal",
    "gaussian_mixture", "gmm",
    "laplace",
    "student_t", "t",
    # Continuous positive
    "exponential",
    "gamma",
    "lognormal", "log_normal",
    "weibull",
    "inverse_gaussian", "wald",
    # Bounded [0, 1]
    "beta",
    # Nominal / ordinal finite support (unit-level codes, e.g. urbanicity)
    "categorical",
    "multinomial",
    "nominal",
    # Positive heavy-tailed (variance/scale priors in HCM simulations)
    "half_cauchy", "halfcauchy",
    # Non-parametric fallback
    "nonparametric",
}

#: Families valid only for :class:`SubunitParamEstimator` (per-unit Q summaries), not pooled conditionals.
SUBUNIT_ONLY_FAMILIES: frozenset[str] = frozenset({"beta_unit_minmax"})

DEFAULT_GAUSSIAN_MIXTURE_COMPONENTS = 2


def _fit_q_from_subunit_task(
    task: tuple[str, np.ndarray, str, Optional[Dict[str, Any]]],
) -> tuple[str, np.ndarray]:
    """Fit one Q-summary array from one raw subunit matrix."""
    q_key, arr_np, family, estimator_kwargs = task
    est = SubunitParamEstimator(family=family, estimator_kwargs=estimator_kwargs)
    return q_key, est.fit(arr_np)


def _fit_conditional_q_row_task(
    task: tuple[np.ndarray, np.ndarray, np.ndarray, str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]],
) -> np.ndarray:
    """Fit one per-unit conditional Q profile."""
    y_i, a_i, eval_vals, family_outcome, estimator_backend, torch_kwargs, estimator_kwargs = task
    est = ConditionalDensityEstimator(
        family=family_outcome,
        backend=estimator_backend,
        torch_kwargs=torch_kwargs,
        estimator_kwargs=estimator_kwargs,
    )
    est.fit(y_i, a_i.reshape(-1, 1))
    return np.array(
        [
            float(est.expectation(np.array([[cval]], dtype=float)))
            for cval in eval_vals
        ],
        dtype=float,
    )


def _find_subunit_matrix_for_letter(
    data: Dict[str, np.ndarray],
    letter: str,
) -> Optional[tuple[str, np.ndarray]]:
    """Return (key, arr) for the first 2-D subunit array whose key starts with ``letter``."""
    lt = letter.lower().strip()
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim == 2 and k.lower().strip().startswith(lt):
            return k, arr
    return None


def _fit_conditional_q_multiparent_row_task(
    task: tuple[np.ndarray, np.ndarray, str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]],
) -> np.ndarray:
    """
    Fit Y | (X_1,…,X_p) on one unit's subunits; return one scalar = mean_j E[Y | X_ij].

    Used for sanitized names like ``Qy_g_l_m`` (multi-parent Q); avoids injecting
    ``iv_val`` on non-treatment parents (see ``_precompute_conditional_q_vars`` docstring).
    """
    y_i, X_i, family_outcome, estimator_backend, torch_kwargs, estimator_kwargs = task
    est = ConditionalDensityEstimator(
        family=family_outcome,
        backend=estimator_backend,
        torch_kwargs=torch_kwargs,
        estimator_kwargs=estimator_kwargs,
    )
    X2 = np.asarray(X_i, dtype=float).reshape(len(y_i), -1)
    est.fit(np.asarray(y_i, dtype=float).ravel(), X2)
    preds = [float(est.expectation(X2[j : j + 1, :])) for j in range(X2.shape[0])]
    return np.array([float(np.mean(preds))], dtype=float)


def _log_gaussian_density(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Stable log-density for a multivariate Gaussian."""
    x_v = np.asarray(x, dtype=float).reshape(-1)
    mean_v = np.asarray(mean, dtype=float).reshape(-1)
    cov_m = np.asarray(cov, dtype=float)
    cov_m = cov_m + 1e-8 * np.eye(cov_m.shape[0], dtype=float)
    diff = x_v - mean_v
    sign, logdet = np.linalg.slogdet(cov_m)
    if sign <= 0:
        cov_m = cov_m + 1e-6 * np.eye(cov_m.shape[0], dtype=float)
        sign, logdet = np.linalg.slogdet(cov_m)
    inv = np.linalg.pinv(cov_m)
    quad = float(diff.T @ inv @ diff)
    dim = len(x_v)
    return float(-0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad))

_FAMILY_ALIASES: Dict[str, str] = {
    "normal": "gaussian",
    "gmm": "gaussian_mixture",
    "t": "student_t",
    "log_normal": "lognormal",
    "wald": "inverse_gaussian",
    "halfcauchy": "half_cauchy",
    "multinomial": "categorical",
    "nominal": "categorical",
    "beta_scaled_unit": "beta_unit_minmax",
    "beta_minmax_unit": "beta_unit_minmax",
}


def _canonical_family_name(family: str) -> str:
    family_l = family.lower().strip()
    return _FAMILY_ALIASES.get(family_l, family_l)


def _resolve_estimator_kwargs(
    estimator_kwargs: Optional[Dict[str, Any]],
    *,
    variable_name: Optional[str] = None,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    if not estimator_kwargs:
        return {}
    if not any(isinstance(value, dict) for value in estimator_kwargs.values()):
        return dict(estimator_kwargs)
    merged: Dict[str, Any] = {}
    default_kwargs = estimator_kwargs.get("__default__")
    if isinstance(default_kwargs, dict):
        merged.update(default_kwargs)
    if family is not None:
        family_kwargs = estimator_kwargs.get(_canonical_family_name(family))
        if isinstance(family_kwargs, dict):
            merged.update(family_kwargs)
    if variable_name is not None:
        var_kwargs = estimator_kwargs.get(variable_name)
        if isinstance(var_kwargs, dict):
            merged.update(var_kwargs)
    return merged


class ConditionalDensityEstimator:
    """
    Estimate P(Y | X_1, ..., X_k) from observed data.

    The *family* argument selects the parametric model.  Supported values:

    **Binary / count**

    ``"bernoulli"``
        Y ∈ {0, 1}.  Unconditional: sample proportion.
        Conditional: logistic regression.

    ``"poisson"``
        Y ∈ {0, 1, 2, …}.  Unconditional: MLE λ = ȳ.
        Conditional: log-linear (Poisson GLM via sklearn or scipy).

    **Continuous unbounded**

    ``"gaussian"`` / ``"normal"``
        Y ∈ ℝ.  Parameters (μ, σ).  Conditional: linear regression for μ.

    ``"laplace"``
        Y ∈ ℝ.  Parameters (loc, scale).  Conditional: LAD (L1) regression for loc.

    ``"student_t"`` / ``"t"``
        Y ∈ ℝ.  Parameters (df, loc, scale) via scipy MLE.
        Conditional: linear regression for loc, fixed df/scale from marginal fit.

    **Continuous positive**

    ``"exponential"``
        Y > 0.  Parameter λ = 1/ȳ.  Conditional: 1/E[Y|X] via linear regression.

    ``"gamma"``
        Y > 0.  Parameters (α, β) via MLE.
        Conditional: linear regression for mean, moment-match for shape.

    ``"lognormal"`` / ``"log_normal"``
        Y > 0.  Parameters (μ_log, σ_log) on log scale.
        Conditional: linear regression on log Y.

    ``"weibull"``
        Y > 0.  Parameters (k, λ) via scipy MLE.
        Conditional: linear regression for scale, fixed shape from marginal fit.

    ``"inverse_gaussian"`` / ``"wald"``
        Y > 0.  Parameters (μ, λ) via MLE.
        Conditional: linear regression for μ.

    **Bounded [0, 1]**

    ``"beta"``
        Y ∈ (0, 1).  Parameters (α, β) via MOM.
        Conditional: logit-linear regression for μ, fixed concentration.

    **Positive heavy-tailed (variance/scale priors)**

    ``"half_cauchy"`` / ``"halfcauchy"``
        Y > 0.  Parameter γ (scale).  Half-Cauchy(0, γ).
        Unconditional: γ estimated via scipy MLE or median heuristic.
        Conditional: linear regression for scale.
        Used in HCM paper simulations as prior on variance parameters τ.

    **Finite categorical (urbanicity codes, etc.)**

    ``"categorical"`` (aliases ``multinomial``, ``nominal``)
        ``Y`` takes finitely many numeric class codes (after rounding to 6 decimals).
        Unconditional: Laplace-smoothed empirical class frequencies (hyperparameter
        ``categorical_laplace`` in ``estimator_kwargs``, default ``1.0``).
        With ``X``: multinomial logistic regression when ``scikit-learn`` is
        available (``torch`` backend is ignored for this family).  ``P(Y=y|X)``
        is the predicted class probability; ``E[Y|X]`` is
        ``sum_k level_k * P(Y=level_k|X)`` (a weighted code, not a count).

    **Non-parametric**

    ``"nonparametric"``
        Discrete Y → conditional frequency table with k-NN.
        Continuous Y → Gaussian KDE.
    """

    def __init__(
        self,
        family: str = "nonparametric",
        regularization: float = 1e4,
        backend: str = "numpy",
        torch_kwargs: Optional[Dict[str, Any]] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        family = family.lower().strip()
        self.family = _FAMILY_ALIASES.get(family, family)
        self.regularization = regularization
        self.backend = backend.lower().strip()
        if self.backend not in {"numpy", "torch", "numpyro"}:
            raise ValueError(
                f"Unsupported backend {backend!r}. Choose from 'numpy', 'torch', or 'numpyro'."
            )
        self.torch_kwargs = dict(torch_kwargs or {})
        self.estimator_kwargs = dict(estimator_kwargs or {})
        self._fitted = False
        if self.family in SUBUNIT_ONLY_FAMILIES:
            raise ValueError(
                f"Family {self.family!r} is only for subunit Q summaries "
                f"(``SubunitParamEstimator`` on 2-D matrices), not for ``ConditionalDensityEstimator``."
            )

    # ------------------------------------------------------------------ fit --

    def fit(self, Y: np.ndarray, X: Optional[np.ndarray] = None) -> "ConditionalDensityEstimator":
        """
        Fit the estimator.

        Parameters
        ----------
        Y : array of shape (n,)
            Observations of the outcome variable.
        X : array of shape (n, d) or (n,) or None
            Conditioning variables.  None → marginal estimator.
        """
        Y = np.asarray(Y, dtype=float).ravel()
        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != len(Y):
                X = X.T  # fix transposition
        self._Y = Y
        self._X = X
        self._n = len(Y)

        dispatch: Dict[str, Callable] = {
            "bernoulli":        self._fit_bernoulli,
            "poisson":          self._fit_poisson,
            "gaussian":         self._fit_gaussian,
            "gaussian_mixture": self._fit_gaussian_mixture,
            "laplace":          self._fit_laplace,
            "student_t":        self._fit_student_t,
            "exponential":      self._fit_exponential,
            "gamma":            self._fit_gamma,
            "lognormal":        self._fit_lognormal,
            "weibull":          self._fit_weibull,
            "inverse_gaussian": self._fit_inverse_gaussian,
            "beta":             self._fit_beta,
            "half_cauchy":      self._fit_half_cauchy,
            "categorical":      self._fit_categorical,
        }
        if self.family in dispatch:
            dispatch[self.family](Y, X)
        else:
            self._fit_nonparametric(Y, X)

        self._fitted = True
        return self

    # -------- Bernoulli -------------------------------------------------------

    def _fit_bernoulli(self, Y, X):
        self._lr_model = None
        self._p_marginal = float(np.clip(np.mean(Y), 1e-9, 1 - 1e-9))  # always set
        is_binary = set(np.unique(Y).tolist()).issubset({0.0, 1.0})
        if X is None or X.shape[1] == 0:
            pass  # _p_marginal already set
        elif self.backend == "torch":
            state = torch_fit_batched_bernoulli(
                x_batch=X[None, :, :],
                y_batch=Y[None, :],
                device=resolve_torch_device_from_mapping(self.torch_kwargs),
                devices=self.torch_kwargs.get("devices"),
                max_iter=int(self.torch_kwargs.get("max_iter", 200)),
                lr=float(self.torch_kwargs.get("lr", 5e-2)),
                weight_decay=float(self.torch_kwargs.get("weight_decay", 1e-4)),
            )
            self._torch_bernoulli_state = state
        elif SKLEARN_AVAILABLE and is_binary and len(np.unique(Y)) == 2:
            lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=self.regularization)
            lr.fit(X, Y.astype(int))
            self._lr_model = lr
        else:
            # Y is continuous proportions (Q-variable): use linear regression clipped to [0,1]
            if SKLEARN_AVAILABLE:
                lr = LinearRegression()
                lr.fit(X, Y)
                self._lr_model = lr  # reuse attribute; detect in eval by checking type
            else:
                self._p_marginal = float(np.clip(np.mean(Y), 1e-9, 1 - 1e-9))

    def _eval_bernoulli(self, x_query, y_query) -> float:
        p = self._expect_bernoulli(x_query)
        y_int = int(round(y_query))
        return p if y_int == 1 else (1.0 - p)

    def _expect_bernoulli(self, x_query) -> float:
        if hasattr(self, "_variational_state"):
            return float(variational_mean_prediction(self._variational_state, np.atleast_2d(np.asarray(x_query, dtype=float)))[0])
        if hasattr(self, "_torch_bernoulli_state"):
            x2d = np.atleast_2d(np.asarray(x_query, dtype=np.float32))[None, :, :]
            return float(torch_predict_batched_bernoulli(self._torch_bernoulli_state, x2d)[0, 0])
        if self._lr_model is None:
            return self._p_marginal
        x2d = np.atleast_2d(x_query)
        if hasattr(self._lr_model, "predict_proba"):  # LogisticRegression
            return float(self._lr_model.predict_proba(x2d)[0, 1])
        else:  # LinearRegression fallback for continuous Y
            return float(np.clip(self._lr_model.predict(x2d)[0], 0.0, 1.0))

    # -------- Gaussian --------------------------------------------------------

    def _fit_gaussian(self, Y, X):
        self._lr_gauss = None
        self._mu_marginal = float(np.mean(Y))      # always set
        self._sigma = float(np.std(Y) + 1e-9)      # always set
        if X is None or X.shape[1] == 0:
            pass  # marginal params already set
        elif self.backend == "torch":
            state = torch_fit_batched_gaussian(
                x_batch=X[None, :, :],
                y_batch=Y[None, :],
                device=resolve_torch_device_from_mapping(self.torch_kwargs),
                devices=self.torch_kwargs.get("devices"),
                ridge=float(self.torch_kwargs.get("ridge", 1e-4)),
            )
            self._torch_gaussian_state = state
            self._sigma = float(state.sigma[0])
        else:
            if SKLEARN_AVAILABLE:
                lr = LinearRegression()
                lr.fit(X, Y)
                self._lr_gauss = lr
                residuals = Y - lr.predict(X)
                self._sigma = float(np.std(residuals) + 1e-9)
            else:
                self._mu_marginal = float(np.mean(Y))
                self._sigma = float(np.std(Y) + 1e-9)

    def _eval_gaussian(self, x_query, y_query) -> float:
        mu = self._predict_mu_gaussian(x_query)
        if SCIPY_AVAILABLE:
            return float(_sp_stats.norm.pdf(y_query, loc=mu, scale=self._sigma))
        # Manual Gaussian pdf
        z = (y_query - mu) / self._sigma
        return float(np.exp(-0.5 * z * z) / (self._sigma * np.sqrt(2 * np.pi)))

    def _predict_mu_gaussian(self, x_query) -> float:
        if hasattr(self, "_variational_state"):
            return float(variational_mean_prediction(self._variational_state, np.atleast_2d(np.asarray(x_query, dtype=float)))[0])
        if hasattr(self, "_torch_gaussian_state"):
            x2d = np.atleast_2d(np.asarray(x_query, dtype=np.float32))[None, :, :]
            return float(torch_predict_batched_gaussian(self._torch_gaussian_state, x2d)[0, 0])
        if self._lr_gauss is not None:
            return float(self._lr_gauss.predict(np.atleast_2d(x_query))[0])
        return self._mu_marginal

    # -------- Gaussian mixture -----------------------------------------------

    def _fit_gaussian_mixture(self, Y, X):
        if self.backend == "numpyro":
            self._variational_state = fit_variational_conditional_estimator(
                y=Y,
                x=X,
                family="gaussian_mixture",
                n_components=int(self.estimator_kwargs.get("n_components", DEFAULT_GAUSSIAN_MIXTURE_COMPONENTS)),
                num_steps=int(self.estimator_kwargs.get("num_steps", 3500)),
                learning_rate=float(self.estimator_kwargs.get("learning_rate", 8e-3)),
                num_posterior_samples=int(self.estimator_kwargs.get("num_posterior_samples", 320)),
                seed=int(self.estimator_kwargs.get("seed", 0)),
                device=self.estimator_kwargs.get("device"),
            )
            self._gmm_x_dim = 0 if X is None or X.shape[1] == 0 else int(X.shape[1])
            self._gmm_is_conditional = self._gmm_x_dim > 0
            self._gmm_components = int(self.estimator_kwargs.get("n_components", DEFAULT_GAUSSIAN_MIXTURE_COMPONENTS))
            self._mu_marginal = float(variational_mean_prediction(self._variational_state, None)[0])
            return
        if not SKLEARN_MIXTURE_AVAILABLE:
            raise RuntimeError("GaussianMixture requires scikit-learn to be installed.")
        n_components = int(self.estimator_kwargs.get("n_components", DEFAULT_GAUSSIAN_MIXTURE_COMPONENTS))
        max_iter = int(self.estimator_kwargs.get("max_iter_gmm", 300))
        random_state = int(self.estimator_kwargs.get("random_state_gmm", 0))
        reg_covar = float(self.estimator_kwargs.get("reg_covar_gmm", 1e-6))
        self._gmm_x_dim = 0 if X is None or X.shape[1] == 0 else int(X.shape[1])
        self._gmm_is_conditional = self._gmm_x_dim > 0
        self._gmm_components = n_components

        if not self._gmm_is_conditional:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",
                    reg_covar=reg_covar,
                    max_iter=max_iter,
                    random_state=random_state,
                )
                gmm.fit(Y.reshape(-1, 1))
                self._gmm_model = gmm
                self._gmm_weights = gmm.weights_.copy()
                self._gmm_means = gmm.means_.reshape(-1)
                self._gmm_vars = gmm.covariances_.reshape(-1).clip(min=1e-9)
            except Exception:
                self._gmm_model = None
                self._gmm_weights = np.array([1.0], dtype=float)
                self._gmm_means = np.array([float(np.mean(Y))], dtype=float)
                self._gmm_vars = np.array([float(max(np.var(Y), 1e-9))], dtype=float)
            self._mu_marginal = float(np.sum(self._gmm_weights * self._gmm_means))
            return

        joint = np.column_stack([X, Y])
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=reg_covar,
                max_iter=max_iter,
                random_state=random_state,
            )
            gmm.fit(joint)
            self._gmm_model = gmm
            self._gmm_weights = gmm.weights_.copy()
            self._gmm_joint_means = gmm.means_.copy()
            self._gmm_joint_covs = gmm.covariances_.copy()
        except Exception:
            self._gmm_model = None
            self._gmm_weights = np.array([1.0], dtype=float)
            self._gmm_joint_means = np.column_stack([np.mean(X, axis=0, keepdims=True), np.array([[float(np.mean(Y))]])]).reshape(1, -1)
            cov = np.cov(joint.T) if joint.shape[0] > 1 else np.eye(joint.shape[1], dtype=float)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]], dtype=float)
            self._gmm_joint_covs = cov.reshape(1, cov.shape[0], cov.shape[1])
        self._mu_marginal = float(np.mean(Y))

    def _conditional_gmm_terms(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x_query, dtype=float).reshape(-1)
        if x_arr.size != self._gmm_x_dim:
            raise ValueError(
                f"Expected x_query with {self._gmm_x_dim} features, got {x_arr.size}."
            )
        log_weights: list[float] = []
        cond_means: list[float] = []
        cond_vars: list[float] = []
        for k in range(self._gmm_components):
            weight_k = float(self._gmm_weights[k])
            mean_joint = self._gmm_joint_means[k]
            cov_joint = self._gmm_joint_covs[k]
            mu_x = mean_joint[: self._gmm_x_dim]
            mu_y = float(mean_joint[self._gmm_x_dim])
            sigma_xx = cov_joint[: self._gmm_x_dim, : self._gmm_x_dim]
            sigma_xy = cov_joint[: self._gmm_x_dim, self._gmm_x_dim : self._gmm_x_dim + 1]
            sigma_yx = cov_joint[self._gmm_x_dim : self._gmm_x_dim + 1, : self._gmm_x_dim]
            sigma_yy = float(cov_joint[self._gmm_x_dim, self._gmm_x_dim])
            sigma_xx_inv = np.linalg.pinv(sigma_xx + 1e-8 * np.eye(self._gmm_x_dim))
            diff = x_arr - mu_x
            cond_mean = mu_y + float((sigma_yx @ sigma_xx_inv @ diff.reshape(-1, 1)).ravel()[0])
            cond_var = sigma_yy - float((sigma_yx @ sigma_xx_inv @ sigma_xy).ravel()[0])
            cond_var = float(max(cond_var, 1e-9))
            log_weight = np.log(weight_k + 1e-12) + _log_gaussian_density(x_arr, mu_x, sigma_xx)
            log_weights.append(log_weight)
            cond_means.append(cond_mean)
            cond_vars.append(cond_var)
        log_weights_np = np.asarray(log_weights, dtype=float)
        log_weights_np = log_weights_np - np.max(log_weights_np)
        weights = np.exp(log_weights_np)
        weights = weights / np.sum(weights)
        return weights, np.asarray(cond_means, dtype=float), np.asarray(cond_vars, dtype=float)

    def _eval_gaussian_mixture(self, x_query, y_query) -> float:
        if hasattr(self, "_variational_state"):
            x_arg = None if x_query is None else np.atleast_2d(np.asarray(x_query, dtype=float))
            draws = variational_density_samples(self._variational_state, float(y_query), x_arg)
            return float(np.mean(draws))
        y_f = float(y_query)
        if not getattr(self, "_gmm_is_conditional", False):
            densities = (
                np.exp(-0.5 * ((y_f - self._gmm_means) ** 2) / self._gmm_vars)
                / np.sqrt(2.0 * np.pi * self._gmm_vars)
            )
            return float(np.sum(self._gmm_weights * densities))
        weights, means, variances = self._conditional_gmm_terms(x_query)
        densities = (
            np.exp(-0.5 * ((y_f - means) ** 2) / variances)
            / np.sqrt(2.0 * np.pi * variances)
        )
        return float(np.sum(weights * densities))

    def _predict_mu_gaussian_mixture(self, x_query) -> float:
        if hasattr(self, "_variational_state"):
            x_arg = None if x_query is None else np.atleast_2d(np.asarray(x_query, dtype=float))
            return float(variational_mean_prediction(self._variational_state, x_arg)[0])
        if not getattr(self, "_gmm_is_conditional", False):
            return float(np.sum(self._gmm_weights * self._gmm_means))
        weights, means, _ = self._conditional_gmm_terms(x_query)
        return float(np.sum(weights * means))

    # -------- Poisson ---------------------------------------------------------

    def _fit_poisson(self, Y, X):
        self._lambda_marginal = float(np.maximum(np.mean(Y), 1e-9))
        self._lr_poisson = None
        if X is not None and X.shape[1] > 0 and self.backend == "torch":
            state = torch_fit_batched_poisson(
                x_batch=X[None, :, :],
                y_batch=Y[None, :],
                device=resolve_torch_device_from_mapping(self.torch_kwargs),
                devices=self.torch_kwargs.get("devices"),
                max_iter=int(self.torch_kwargs.get("max_iter", 200)),
                lr=float(self.torch_kwargs.get("lr", 5e-2)),
                weight_decay=float(self.torch_kwargs.get("weight_decay", 1e-4)),
            )
            self._torch_poisson_state = state
        elif X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            try:
                from sklearn.linear_model import PoissonRegressor
                glm = PoissonRegressor(max_iter=1000, alpha=0)
                glm.fit(X, np.maximum(Y, 0))
                self._lr_poisson = glm
            except Exception:
                # Fallback: log-linear via linear regression on log(Y+0.5)
                lr = LinearRegression()
                lr.fit(X, np.log(np.maximum(Y, 0.5)))
                self._lr_poisson = ("log_linear", lr)

    def _predict_lambda_poisson(self, x_query) -> float:
        if hasattr(self, "_torch_poisson_state"):
            x2d = np.atleast_2d(np.asarray(x_query, dtype=np.float32))[None, :, :]
            return float(torch_predict_batched_poisson(self._torch_poisson_state, x2d)[0, 0])
        if self._lr_poisson is None:
            return self._lambda_marginal
        if isinstance(self._lr_poisson, tuple):  # log-linear fallback
            lr = self._lr_poisson[1]
            return float(np.exp(np.clip(lr.predict(np.atleast_2d(x_query))[0], -10, 10)))
        return float(np.maximum(self._lr_poisson.predict(np.atleast_2d(x_query))[0], 1e-9))

    def _eval_poisson(self, x_query, y_query) -> float:
        lam = self._predict_lambda_poisson(x_query)
        if SCIPY_AVAILABLE:
            return float(_sp_stats.poisson.pmf(int(round(y_query)), lam))
        # Manual
        k = int(round(y_query))
        import math
        return float(np.exp(-lam) * (lam ** k) / math.factorial(k))

    # -------- Laplace ---------------------------------------------------------

    def _fit_laplace(self, Y, X):
        self._laplace_loc = float(np.median(Y))
        self._laplace_scale = float(np.mean(np.abs(Y - self._laplace_loc)) + 1e-9)
        self._lr_laplace = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            # LAD regression via quantile regression (sklearn ≥ 1.0)
            try:
                from sklearn.linear_model import QuantileRegressor
                qr = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
                qr.fit(X, Y)
                residuals = Y - qr.predict(X)
                self._laplace_scale = float(np.mean(np.abs(residuals)) + 1e-9)
                self._lr_laplace = qr
            except Exception:
                lr = LinearRegression()
                lr.fit(X, Y)
                residuals = Y - lr.predict(X)
                self._laplace_scale = float(np.mean(np.abs(residuals)) + 1e-9)
                self._lr_laplace = lr

    def _predict_loc_laplace(self, x_query) -> float:
        if self._lr_laplace is None:
            return self._laplace_loc
        return float(self._lr_laplace.predict(np.atleast_2d(x_query))[0])

    def _eval_laplace(self, x_query, y_query) -> float:
        loc = self._predict_loc_laplace(x_query)
        if SCIPY_AVAILABLE:
            return float(_sp_stats.laplace.pdf(y_query, loc=loc, scale=self._laplace_scale))
        return float(np.exp(-abs(y_query - loc) / self._laplace_scale) / (2 * self._laplace_scale))

    # -------- Student-t -------------------------------------------------------

    def _fit_student_t(self, Y, X):
        self._t_df, self._t_loc, self._t_scale = 10.0, float(np.mean(Y)), float(np.std(Y) + 1e-9)
        if SCIPY_AVAILABLE:
            try:
                self._t_df, self._t_loc, self._t_scale = _sp_stats.t.fit(Y)
            except Exception:
                pass
        self._lr_t = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y)
            residuals = Y - lr.predict(X)
            if SCIPY_AVAILABLE:
                try:
                    _, _, self._t_scale = _sp_stats.t.fit(residuals, f0=self._t_df)
                except Exception:
                    self._t_scale = float(np.std(residuals) + 1e-9)
            else:
                self._t_scale = float(np.std(residuals) + 1e-9)
            self._lr_t = lr

    def _predict_loc_t(self, x_query) -> float:
        if self._lr_t is None:
            return self._t_loc
        return float(self._lr_t.predict(np.atleast_2d(x_query))[0])

    def _eval_student_t(self, x_query, y_query) -> float:
        loc = self._predict_loc_t(x_query)
        if SCIPY_AVAILABLE:
            return float(_sp_stats.t.pdf(y_query, df=self._t_df, loc=loc, scale=self._t_scale))
        # Fallback to Gaussian
        z = (y_query - loc) / (self._t_scale + 1e-9)
        return float(np.exp(-0.5 * z * z) / ((self._t_scale + 1e-9) * np.sqrt(2 * np.pi)))

    # -------- Exponential -----------------------------------------------------

    def _fit_exponential(self, Y, X):
        Y_pos = np.maximum(Y, 1e-9)
        self._exp_lambda = float(1.0 / np.mean(Y_pos))
        self._lr_exp = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y_pos)
            self._lr_exp = lr

    def _predict_mean_exp(self, x_query) -> float:
        if self._lr_exp is None:
            return float(1.0 / (self._exp_lambda + 1e-9))
        return float(np.maximum(self._lr_exp.predict(np.atleast_2d(x_query))[0], 1e-9))

    def _eval_exponential(self, x_query, y_query) -> float:
        mu = self._predict_mean_exp(x_query)
        lam = 1.0 / mu
        if y_query < 0:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.expon.pdf(y_query, scale=mu))
        return float(lam * np.exp(-lam * y_query))

    # -------- Gamma -----------------------------------------------------------

    def _fit_gamma(self, Y, X):
        Y_pos = np.maximum(Y, 1e-9)
        mu = float(np.mean(Y_pos)); var = float(np.var(Y_pos) + 1e-9)
        # MOM initialisation (always safe)
        self._gamma_shape = float(np.maximum(mu * mu / var, 1e-3))
        self._gamma_scale = float(np.maximum(var / mu, 1e-9))
        if SCIPY_AVAILABLE:
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    sh, _, sc = _sp_stats.gamma.fit(Y_pos, floc=0)
                self._gamma_shape = float(np.maximum(sh, 1e-3))
                self._gamma_scale = float(np.maximum(sc, 1e-9))
            except Exception:
                pass  # keep MOM estimates
        self._gamma_mean = float(np.mean(Y_pos))
        self._lr_gamma = None
        if X is not None and X.shape[1] > 0 and self.backend == "torch":
            state = torch_fit_batched_gamma(
                x_batch=X[None, :, :],
                y_batch=Y_pos[None, :],
                device=resolve_torch_device_from_mapping(self.torch_kwargs),
                devices=self.torch_kwargs.get("devices"),
                max_iter=int(self.torch_kwargs.get("max_iter", 300)),
                lr=float(self.torch_kwargs.get("lr", 5e-2)),
                weight_decay=float(self.torch_kwargs.get("weight_decay", 1e-4)),
            )
            self._torch_gamma_state = state
            self._gamma_shape = float(state.shape[0])
        elif X is not None and X.shape[1] > 0 and self.backend == "numpyro":
            self._variational_state = fit_variational_conditional_estimator(
                y=Y_pos,
                x=X,
                family="gamma",
                num_steps=int(self.estimator_kwargs.get("num_steps", 2500)),
                learning_rate=float(self.estimator_kwargs.get("learning_rate", 1e-2)),
                num_posterior_samples=int(self.estimator_kwargs.get("num_posterior_samples", 256)),
                seed=int(self.estimator_kwargs.get("seed", 0)),
                device=self.estimator_kwargs.get("device"),
            )
        elif X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y_pos)
            self._lr_gamma = lr

    def _predict_mean_gamma(self, x_query) -> float:
        if hasattr(self, "_torch_gamma_state"):
            x2d = np.atleast_2d(np.asarray(x_query, dtype=np.float32))[None, :, :]
            return float(torch_predict_batched_gamma_mean(self._torch_gamma_state, x2d)[0, 0])
        if hasattr(self, "_variational_state"):
            return float(variational_mean_prediction(self._variational_state, np.atleast_2d(np.asarray(x_query, dtype=float)))[0])
        if self._lr_gamma is None:
            return self._gamma_mean
        return float(np.maximum(self._lr_gamma.predict(np.atleast_2d(x_query))[0], 1e-9))

    def _eval_gamma(self, x_query, y_query) -> float:
        if hasattr(self, "_variational_state"):
            draws = variational_density_samples(
                self._variational_state,
                float(y_query),
                np.atleast_2d(np.asarray(x_query, dtype=float)),
            )
            return float(np.mean(draws))
        mu = self._predict_mean_gamma(x_query)
        # Shape fixed from marginal; adjust scale so mean = shape * scale = mu
        scale = float(np.maximum(mu / (self._gamma_shape + 1e-9), 1e-9))
        if y_query <= 0:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.gamma.pdf(y_query, a=self._gamma_shape, scale=scale))
        # Manual: Gamma pdf ∝ y^(α-1) exp(-y/θ)
        import math
        a, th = self._gamma_shape, scale
        return float((y_query ** (a - 1)) * np.exp(-y_query / th) / (th ** a * math.gamma(a)))

    # -------- Log-normal ------------------------------------------------------

    def _fit_lognormal(self, Y, X):
        Y_pos = np.maximum(Y, 1e-9)
        logY = np.log(Y_pos)
        self._logn_mu = float(np.mean(logY))
        self._logn_sigma = float(np.std(logY) + 1e-9)
        self._lr_lognormal = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, logY)
            residuals = logY - lr.predict(X)
            self._logn_sigma = float(np.std(residuals) + 1e-9)
            self._lr_lognormal = lr

    def _predict_mu_lognormal(self, x_query) -> float:
        if self._lr_lognormal is None:
            return self._logn_mu
        return float(self._lr_lognormal.predict(np.atleast_2d(x_query))[0])

    def _eval_lognormal(self, x_query, y_query) -> float:
        mu_log = self._predict_mu_lognormal(x_query)
        if y_query <= 0:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.lognorm.pdf(y_query, s=self._logn_sigma, scale=np.exp(mu_log)))
        z = (np.log(y_query) - mu_log) / self._logn_sigma
        return float(np.exp(-0.5 * z * z) / (y_query * self._logn_sigma * np.sqrt(2 * np.pi)))

    # -------- Weibull ---------------------------------------------------------

    def _fit_weibull(self, Y, X):
        Y_pos = np.maximum(Y, 1e-9)
        # MOM fallback: shape ≈ 1, scale = mean
        self._weibull_c = 1.0
        self._weibull_scale = float(np.mean(Y_pos))
        if SCIPY_AVAILABLE:
            try:
                c, _, sc = _sp_stats.weibull_min.fit(Y_pos, floc=0)
                self._weibull_c = float(np.maximum(c, 0.1))
                self._weibull_scale = float(np.maximum(sc, 1e-9))
            except Exception:
                pass
        self._weibull_mean = float(np.mean(Y_pos))
        self._lr_weibull = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y_pos)
            self._lr_weibull = lr

    def _predict_scale_weibull(self, x_query) -> float:
        """Predict Weibull scale so E[Y|X] matches predicted mean."""
        import math
        mu = self._weibull_mean if self._lr_weibull is None else float(
            np.maximum(self._lr_weibull.predict(np.atleast_2d(x_query))[0], 1e-9))
        # E[Weibull(c, lam)] = lam * Gamma(1 + 1/c)
        try:
            gamma_factor = math.gamma(1.0 + 1.0 / self._weibull_c)
        except Exception:
            gamma_factor = 1.0
        return float(np.maximum(mu / (gamma_factor + 1e-9), 1e-9))

    def _eval_weibull(self, x_query, y_query) -> float:
        scale = self._predict_scale_weibull(x_query)
        if y_query <= 0:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.weibull_min.pdf(y_query, c=self._weibull_c, scale=scale))
        c, lam = self._weibull_c, scale
        return float((c / lam) * (y_query / lam) ** (c - 1) * np.exp(-(y_query / lam) ** c))

    # -------- Inverse-Gaussian (Wald) ----------------------------------------

    def _fit_inverse_gaussian(self, Y, X):
        Y_pos = np.maximum(Y, 1e-9)
        mu = float(np.mean(Y_pos))
        # MLE for lambda: n / sum(1/y - 1/mu)
        inv_sum = float(np.sum(1.0 / Y_pos - 1.0 / mu))
        self._ig_mu = mu
        self._ig_lambda = float(len(Y_pos) / (inv_sum + 1e-9)) if inv_sum > 0 else 1.0
        self._lr_ig = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y_pos)
            self._lr_ig = lr

    def _predict_mu_ig(self, x_query) -> float:
        if self._lr_ig is None:
            return self._ig_mu
        return float(np.maximum(self._lr_ig.predict(np.atleast_2d(x_query))[0], 1e-9))

    def _eval_inverse_gaussian(self, x_query, y_query) -> float:
        mu = self._predict_mu_ig(x_query)
        lam = self._ig_lambda
        if y_query <= 0:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.invgauss.pdf(y_query, mu=mu / lam, scale=lam))
        coeff = np.sqrt(lam / (2 * np.pi * y_query ** 3))
        exponent = -lam * (y_query - mu) ** 2 / (2 * mu ** 2 * y_query)
        return float(coeff * np.exp(exponent))

    # -------- Beta ------------------------------------------------------------

    def _fit_beta(self, Y, X):
        Y_clipped = np.clip(Y, 1e-6, 1 - 1e-6)
        mu = float(np.mean(Y_clipped))
        var = float(np.var(Y_clipped) + 1e-9)
        # MOM
        conc = mu * (1 - mu) / var - 1.0
        self._beta_alpha = float(np.maximum(mu * conc, 1e-3))
        self._beta_beta = float(np.maximum((1 - mu) * conc, 1e-3))
        self._beta_conc = float(np.maximum(conc, 1e-3))
        self._beta_mu = mu
        self._lr_beta = None
        if X is not None and X.shape[1] > 0 and self.backend == "torch":
            state = torch_fit_batched_beta(
                x_batch=X[None, :, :],
                y_batch=Y_clipped[None, :],
                device=resolve_torch_device_from_mapping(self.torch_kwargs),
                devices=self.torch_kwargs.get("devices"),
                max_iter=int(self.torch_kwargs.get("max_iter", 300)),
                lr=float(self.torch_kwargs.get("lr", 5e-2)),
                weight_decay=float(self.torch_kwargs.get("weight_decay", 1e-4)),
            )
            self._torch_beta_state = state
            self._beta_conc = float(state.concentration[0])
        elif X is not None and X.shape[1] > 0 and self.backend == "numpyro":
            self._variational_state = fit_variational_conditional_estimator(
                y=Y_clipped,
                x=X,
                family="beta",
                num_steps=int(self.estimator_kwargs.get("num_steps", 2500)),
                learning_rate=float(self.estimator_kwargs.get("learning_rate", 1e-2)),
                num_posterior_samples=int(self.estimator_kwargs.get("num_posterior_samples", 256)),
                seed=int(self.estimator_kwargs.get("seed", 0)),
                device=self.estimator_kwargs.get("device"),
            )
        elif X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            # Logit-linear regression for mean
            from sklearn.linear_model import LogisticRegression as _LR
            # Use linear regression on logit(Y) as proxy
            lr = LinearRegression()
            lr.fit(X, np.log(Y_clipped / (1 - Y_clipped)))  # logit
            self._lr_beta = lr

    def _predict_mu_beta(self, x_query) -> float:
        if hasattr(self, "_torch_beta_state"):
            x2d = np.atleast_2d(np.asarray(x_query, dtype=np.float32))[None, :, :]
            return float(torch_predict_batched_beta_mean(self._torch_beta_state, x2d)[0, 0])
        if hasattr(self, "_variational_state"):
            return float(variational_mean_prediction(self._variational_state, np.atleast_2d(np.asarray(x_query, dtype=float)))[0])
        if self._lr_beta is None:
            return self._beta_mu
        logit_pred = float(self._lr_beta.predict(np.atleast_2d(x_query))[0])
        return float(1.0 / (1.0 + np.exp(-logit_pred)))

    def _eval_beta(self, x_query, y_query) -> float:
        if hasattr(self, "_variational_state"):
            draws = variational_density_samples(
                self._variational_state,
                float(y_query),
                np.atleast_2d(np.asarray(x_query, dtype=float)),
            )
            return float(np.mean(draws))
        mu = self._predict_mu_beta(x_query)
        conc = self._beta_conc
        alpha = float(np.maximum(mu * conc, 1e-3))
        beta = float(np.maximum((1 - mu) * conc, 1e-3))
        if y_query <= 0 or y_query >= 1:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.beta.pdf(y_query, a=alpha, b=beta))
        import math
        return float((y_query ** (alpha - 1)) * ((1 - y_query) ** (beta - 1))
                     / (math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)))

    # -------- Half-Cauchy -----------------------------------------------------

    def _fit_half_cauchy(self, Y, X):
        """Half-Cauchy(0, γ): PDF = 2 / (π γ (1 + (y/γ)²)) for y > 0."""
        Y_pos = np.maximum(Y, 1e-9)
        # Median of Half-Cauchy equals γ * tan(π/4) = γ, so median is a good init
        self._hc_scale = float(np.median(Y_pos))
        if SCIPY_AVAILABLE:
            try:
                _, sc = _sp_stats.halfcauchy.fit(Y_pos, floc=0)
                self._hc_scale = float(np.maximum(sc, 1e-9))
            except Exception:
                pass
        self._hc_mean = float(np.mean(Y_pos))
        self._lr_hc = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y_pos)
            self._lr_hc = lr

    def _predict_scale_hc(self, x_query) -> float:
        if self._lr_hc is None:
            return self._hc_scale
        return float(np.maximum(self._lr_hc.predict(np.atleast_2d(x_query))[0], 1e-9))

    def _eval_half_cauchy(self, x_query, y_query) -> float:
        scale = self._predict_scale_hc(x_query)
        if y_query <= 0:
            return 0.0
        if SCIPY_AVAILABLE:
            return float(_sp_stats.halfcauchy.pdf(y_query, scale=scale))
        import math
        return float(2.0 / (math.pi * scale * (1.0 + (y_query / scale) ** 2)))

    # -------- Categorical (finite support, nominal / ordinal codes) -----------

    def _fit_categorical(self, Y, X) -> None:
        """Multinomial logistic regression when ``X`` is present; else Laplace-smoothed counts."""
        self._cat_lr = None
        y = np.asarray(Y, dtype=float).ravel()
        y = y[np.isfinite(y)]
        if len(y) == 0:
            self._cat_classes = np.array([0.0], dtype=float)
            self._cat_marginal_prob = np.array([1.0], dtype=float)
            return
        y_round = np.round(y, 6)
        self._cat_classes, y_idx = np.unique(y_round, return_inverse=True)
        k = int(len(self._cat_classes))
        laplace = float(self.estimator_kwargs.get("categorical_laplace", 1.0))
        counts = np.bincount(y_idx, minlength=k).astype(float) + laplace
        self._cat_marginal_prob = counts / float(np.sum(counts))
        if k == 1:
            return
        if X is None or X.shape[1] == 0 or not SKLEARN_AVAILABLE:
            return
        try:
            if k == 2:
                lr = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    C=float(self.regularization),
                )
            else:
                lr = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    C=float(self.regularization),
                    multi_class="multinomial",
                )
            lr.fit(X, y_idx)
            self._cat_lr = lr
        except Exception:
            self._cat_lr = None

    def _categorical_proba(self, x_query: Optional[np.ndarray]) -> np.ndarray:
        if getattr(self, "_cat_lr", None) is not None:
            x2d = np.atleast_2d(np.asarray(x_query, dtype=float))
            return np.asarray(self._cat_lr.predict_proba(x2d)[0], dtype=float).ravel()
        return np.asarray(self._cat_marginal_prob, dtype=float).ravel()

    def _eval_categorical(self, x_query: Optional[np.ndarray], y_query: float) -> float:
        proba = self._categorical_proba(x_query)
        j = int(np.argmin(np.abs(self._cat_classes - float(y_query))))
        return float(proba[j])

    def _expect_categorical(self, x_query: Optional[np.ndarray]) -> float:
        proba = self._categorical_proba(x_query)
        return float(np.sum(self._cat_classes.astype(float) * proba))

    # -------- Non-parametric --------------------------------------------------

    def _fit_nonparametric(self, Y, X):
        self._is_discrete = np.all(Y == Y.astype(int))
        self._np_X = X
        self._np_Y = Y
        if X is None:
            if self._is_discrete:
                vals, counts = np.unique(Y, return_counts=True)
                self._marginal_table: Dict[float, float] = dict(zip(vals.tolist(), (counts / counts.sum()).tolist()))
            else:
                if SCIPY_AVAILABLE:
                    self._kde_marginal = _sp_stats.gaussian_kde(Y)
        else:
            if not self._is_discrete and SCIPY_AVAILABLE:
                # Joint KDE on (X, Y)
                try:
                    joint = np.vstack([X.T, Y.reshape(1, -1)])
                    self._joint_kde = _sp_stats.gaussian_kde(joint)
                    self._X_kde = _sp_stats.gaussian_kde(X.T) if X.shape[1] > 0 else None
                except np.linalg.LinAlgError:
                    pass  # Singular matrix → fall through to k-NN

    def _eval_nonparametric(self, x_query, y_query) -> float:
        if self._np_X is None:
            if hasattr(self, "_marginal_table"):
                return self._marginal_table.get(float(y_query), 0.0)
            if hasattr(self, "_kde_marginal"):
                return float(self._kde_marginal(np.atleast_1d(y_query))[0])
            return 0.0
        # Conditional with X
        if not self._is_discrete and hasattr(self, "_joint_kde"):
            x_arr = np.atleast_1d(np.asarray(x_query, dtype=float))
            pt_joint = np.append(x_arr, float(y_query)).reshape(-1, 1)
            denom = float(np.atleast_1d(self._X_kde(x_arr.reshape(-1, 1)))[0]) if self._X_kde else 1.0
            return float(np.atleast_1d(self._joint_kde(pt_joint))[0]) / (denom + 1e-300)
        # Discrete or KDE unavailable → k-NN
        return self._knn_conditional(x_query, y_query)

    def _knn_conditional(self, x_query, y_query) -> float:
        if self._np_X is None:
            return float(np.mean(self._np_Y == y_query))
        x_arr = np.atleast_1d(np.asarray(x_query, dtype=float)).ravel()
        nf = int(self._np_X.shape[1])
        if x_arr.size != nf:
            return float(np.mean(self._np_Y == y_query))
        dists = np.linalg.norm(self._np_X - x_arr, axis=1)
        k = max(5, int(0.1 * len(self._np_Y)))
        idx = np.argsort(dists)[:k]
        neighbors_Y = self._np_Y[idx]
        if self._is_discrete:
            return float(np.mean(neighbors_Y == y_query))
        # Continuous: KDE on k-NN subset
        if SCIPY_AVAILABLE and len(np.unique(neighbors_Y)) > 1:
            try:
                return float(_sp_stats.gaussian_kde(neighbors_Y)(np.atleast_1d(y_query))[0])
            except (np.linalg.LinAlgError, TypeError):
                pass
        return float(np.mean(neighbors_Y))

    def _expect_nonparametric(self, x_query) -> float:
        if self._np_X is None:
            return float(np.mean(self._np_Y))
        x_arr = np.atleast_1d(np.asarray(x_query, dtype=float)).ravel()
        nf = int(self._np_X.shape[1])
        if x_arr.size != nf:
            return float(np.mean(self._np_Y))
        dists = np.linalg.norm(self._np_X - x_arr, axis=1)
        k = max(5, int(0.1 * len(self._np_Y)))
        idx = np.argsort(dists)[:k]
        return float(np.mean(self._np_Y[idx]))

    # ------------------------------------------------------------------ API --

    # ------------------------------------------------------------------ API --

    def prob(self, y_query: float, x_query: Optional[np.ndarray] = None) -> float:
        """
        Evaluate P(Y = y_query | X = x_query).

        For continuous Y, returns the probability *density* at y_query.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        x_arr = np.atleast_1d(np.asarray(x_query, dtype=float)) if x_query is not None else None
        _prob_dispatch = {
            "bernoulli":        self._eval_bernoulli,
            "poisson":          self._eval_poisson,
            "gaussian":         self._eval_gaussian,
            "gaussian_mixture": self._eval_gaussian_mixture,
            "laplace":          self._eval_laplace,
            "student_t":        self._eval_student_t,
            "exponential":      self._eval_exponential,
            "gamma":            self._eval_gamma,
            "lognormal":        self._eval_lognormal,
            "weibull":          self._eval_weibull,
            "inverse_gaussian": self._eval_inverse_gaussian,
            "beta":             self._eval_beta,
            "half_cauchy":      self._eval_half_cauchy,
            "categorical":      self._eval_categorical,
        }
        fn = _prob_dispatch.get(self.family)
        if fn is not None:
            return fn(x_arr, y_query)
        return self._eval_nonparametric(x_arr, y_query)

    def expectation(self, x_query: Optional[np.ndarray] = None, n_mc: int = 500) -> float:
        """
        Compute E[Y | X = x_query].

        Parametric families return the analytical conditional mean.
        Non-parametric falls back to k-NN mean.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        x_arr = np.atleast_1d(np.asarray(x_query, dtype=float)) if x_query is not None else None
        # Marginal expectation: return stored parameters directly to avoid
        # passing None into sklearn predict (np.atleast_2d(None) → ValueError).
        if x_arr is None:
            _marginal = {
                "bernoulli":        lambda: getattr(self, "_p_marginal", float("nan")),
                "poisson":          lambda: self._lambda_marginal,
                "gaussian":         lambda: self._mu_marginal,
                "gaussian_mixture": lambda: self._mu_marginal,
                "laplace":          lambda: self._laplace_loc,
                "student_t":        lambda: self._t_loc,
                "exponential":      lambda: 1.0 / (self._exp_lambda + 1e-9),
                "gamma":            lambda: self._gamma_mean,
                "lognormal":        lambda: float(np.exp(self._logn_mu + 0.5 * self._logn_sigma ** 2)),
                "weibull":          lambda: self._weibull_mean,
                "inverse_gaussian": lambda: self._ig_mu,
                "beta":             lambda: self._beta_mu,
                "half_cauchy":      lambda: self._hc_mean,
                "categorical":    lambda: float(
                    np.sum(getattr(self, "_cat_classes", np.zeros(1)) * getattr(self, "_cat_marginal_prob", np.ones(1)))
                ),
            }
            fn_marg = _marginal.get(self.family)
            if fn_marg is not None:
                return float(fn_marg())
            return self._expect_nonparametric(None)
        _expect_dispatch = {
            "bernoulli":        self._expect_bernoulli,
            "poisson":          self._predict_lambda_poisson,
            "gaussian":         self._predict_mu_gaussian,
            "gaussian_mixture": self._predict_mu_gaussian_mixture,
            "laplace":          self._predict_loc_laplace,
            "student_t":        self._predict_loc_t,
            "exponential":      self._predict_mean_exp,
            "gamma":            self._predict_mean_gamma,
            "lognormal":        lambda x: float(np.exp(self._predict_mu_lognormal(x) + 0.5 * self._logn_sigma ** 2)),
            "weibull":          lambda x: self._predict_scale_weibull(x) * __import__("math").gamma(1 + 1 / self._weibull_c),
            "inverse_gaussian": self._predict_mu_ig,
            "beta":             self._predict_mu_beta,
            "half_cauchy":      self._predict_scale_hc,
            "categorical":      self._expect_categorical,
        }
        fn = _expect_dispatch.get(self.family)
        if fn is not None:
            return float(fn(x_arr))
        return self._expect_nonparametric(x_arr)

    def params(self) -> Dict[str, float]:
        """Return the fitted marginal distribution parameters as a dict."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        p: Dict[str, float] = {}
        if self.family == "bernoulli":
            p["p"] = getattr(self, "_p_marginal", float("nan"))
        elif self.family == "poisson":
            p["lambda"] = self._lambda_marginal
        elif self.family == "gaussian":
            p["mu"] = getattr(self, "_mu_marginal", float("nan"))
            p["sigma"] = self._sigma
        elif self.family == "gaussian_mixture":
            p["n_components"] = float(self._gmm_components)
        elif self.family == "laplace":
            p["loc"] = self._laplace_loc; p["scale"] = self._laplace_scale
        elif self.family == "student_t":
            p["df"] = self._t_df; p["loc"] = self._t_loc; p["scale"] = self._t_scale
        elif self.family == "exponential":
            p["lambda"] = self._exp_lambda
        elif self.family == "gamma":
            p["shape"] = self._gamma_shape; p["scale"] = self._gamma_scale
        elif self.family == "lognormal":
            p["mu_log"] = self._logn_mu; p["sigma_log"] = self._logn_sigma
        elif self.family == "weibull":
            p["shape"] = self._weibull_c; p["scale"] = self._weibull_scale
        elif self.family == "inverse_gaussian":
            p["mu"] = self._ig_mu; p["lambda"] = self._ig_lambda
        elif self.family == "beta":
            p["alpha"] = self._beta_alpha; p["beta"] = self._beta_beta
        elif self.family == "half_cauchy":
            p["scale"] = self._hc_scale
        elif self.family == "categorical":
            p["n_classes"] = float(len(getattr(self, "_cat_classes", [])))
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Subunit distribution parameter estimator
# ─────────────────────────────────────────────────────────────────────────────

class SubunitParamEstimator:
    """
    Estimate the distribution parameters of a subunit variable per unit.

    Given a ``(n_units, n_subunits)`` matrix of observations, fits the
    specified parametric family to each unit's within-unit data
    ``{Y_ij}_{j=1}^{m}`` and returns per-unit parameter vectors.

    The result is a sample of the Q-variable ``Q^v``: each element ``q_i``
    is the parameter vector characterising the distribution of subunit
    variable *v* within unit *i*.

    Parameters
    ----------
    family : str
        Distribution family for the subunit variable.  Supported:

        ============= ================== ============================
        Family        Per-unit params    Shape
        ============= ================== ============================
        ``bernoulli`` p_i                (n_units,)
        ``poisson``   λ_i                (n_units,)
        ``exponential`` mean_i           (n_units,)
        ``gaussian``  (μ_i, σ²_i)        (n_units, 2)
        ``beta``      (α_i, β_i)         (n_units, 2)
        ``beta_unit_minmax`` (α_i, β_i) (n_units, 2); Beta on scores mapped to (0, 1) with that unit's min and max
        ``gamma``     (shape_i, scale_i) (n_units, 2)
        ``lognormal`` (μ_log_i, σ²_log_i) (n_units, 2)
        ``nonparametric`` (mean, std, skew, kurt) (n_units, 4)
        ============= ================== ============================
    """

    _SCALAR_FAMILIES: frozenset = frozenset({"bernoulli", "poisson", "exponential"})

    def __init__(self, family: str = "bernoulli", estimator_kwargs: Optional[Dict[str, Any]] = None) -> None:
        family = _FAMILY_ALIASES.get(family, family)
        allowed = SUPPORTED_FAMILIES | SUBUNIT_ONLY_FAMILIES
        if family not in allowed:
            raise ValueError(
                f"Unsupported family {family!r}.  "
                f"Choose from: {sorted(allowed)}."
            )
        self.family = family
        self.estimator_kwargs = dict(estimator_kwargs or {})

    @property
    def n_params(self) -> int:
        """Dimensionality of the per-unit parameter vector."""
        if self.family in self._SCALAR_FAMILIES:
            return 1
        if self.family == "gaussian_mixture":
            return 3 * DEFAULT_GAUSSIAN_MIXTURE_COMPONENTS
        if self.family in {"gaussian", "normal", "beta", "beta_unit_minmax", "gamma",
                           "lognormal", "log_normal",
                           "inverse_gaussian", "wald"}:
            return 2
        if self.family == "nonparametric":
            return 4
        return 1

    def fit_unit(self, y: np.ndarray) -> np.ndarray:
        """
        Fit the distribution to one unit's within-unit observations.

        Parameters
        ----------
        y : 1-D array of subunit observations for a single unit.

        Returns
        -------
        params : np.ndarray of shape ``(n_params,)``.
        """
        y = np.asarray(y, dtype=float).ravel()
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return np.zeros(self.n_params)

        if self.family == "bernoulli":
            return np.array([float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))])

        if self.family in ("gaussian", "normal"):
            return np.array([float(y.mean()), float(max(y.var(ddof=0), 1e-10))])

        if self.family == "gaussian_mixture":
            if not SKLEARN_MIXTURE_AVAILABLE:
                raise RuntimeError("GaussianMixture requires scikit-learn to be installed.")
            n_components = int(self.estimator_kwargs.get("n_components", DEFAULT_GAUSSIAN_MIXTURE_COMPONENTS))
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",
                    reg_covar=float(self.estimator_kwargs.get("reg_covar_gmm", 1e-6)),
                    max_iter=int(self.estimator_kwargs.get("max_iter_gmm", 300)),
                    random_state=int(self.estimator_kwargs.get("random_state_gmm", 0)),
                )
                gmm.fit(y.reshape(-1, 1))
                order = np.argsort(gmm.means_.reshape(-1))
                weights = gmm.weights_[order]
                means = gmm.means_.reshape(-1)[order]
                variances = gmm.covariances_.reshape(-1)[order].clip(min=1e-9)
            except Exception:
                weights = np.full(n_components, 1.0 / n_components, dtype=float)
                means = np.full(n_components, float(np.mean(y)), dtype=float)
                variances = np.full(n_components, float(max(np.var(y), 1e-9)), dtype=float)
            return np.concatenate([weights, means, variances])

        if self.family == "poisson":
            return np.array([float(max(y.mean(), 1e-10))])

        if self.family == "exponential":
            return np.array([float(max(y.mean(), 1e-10))])

        if self.family == "beta_unit_minmax":
            eps_r = float(self.estimator_kwargs.get("beta_unit_range_eps", 1e-9))
            lo = float(np.min(y))
            hi = float(np.max(y))
            denom = max(hi - lo, eps_r)
            z = (y - lo) / denom
            z = np.clip(z, 1e-6, 1.0 - 1e-6)
            m = float(np.clip(z.mean(), 1e-6, 1.0 - 1e-6))
            v = float(max(z.var(ddof=0), 1e-10))
            kappa = max(m * (1.0 - m) / v - 1.0, 0.01)
            return np.array([m * kappa, (1.0 - m) * kappa])

        if self.family == "beta":
            m = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
            v = float(max(y.var(ddof=0), 1e-10))
            kappa = max(m * (1.0 - m) / v - 1.0, 0.01)
            return np.array([m * kappa, (1.0 - m) * kappa])

        if self.family == "gamma":
            m = float(max(y.mean(), 1e-10))
            v = float(max(y.var(ddof=0), 1e-10))
            return np.array([m ** 2 / v, v / m])  # (shape α, scale β)

        if self.family in ("lognormal", "log_normal"):
            y_pos = y[y > 0]
            if len(y_pos) == 0:
                y_pos = np.array([1e-6])
            log_y = np.log(y_pos)
            return np.array([float(log_y.mean()),
                             float(max(log_y.var(ddof=0), 1e-10))])

        if self.family in ("inverse_gaussian", "wald"):
            m = float(max(y.mean(), 1e-10))
            v = float(max(y.var(ddof=0), 1e-10))
            return np.array([m, m ** 3 / v])  # (μ, λ)

        if self.family == "nonparametric":
            m = float(y.mean())
            s = float(max(y.std(ddof=0), 1e-10))
            sk = float(np.mean(((y - m) / s) ** 3))
            ku = float(np.mean(((y - m) / s) ** 4)) - 3.0
            return np.array([m, s, sk, ku])

        return np.array([float(y.mean())])

    def fit(self, Y: np.ndarray) -> np.ndarray:
        """
        Fit per-unit distributions from a subunit data matrix.

        Parameters
        ----------
        Y : ``(n_units, n_subunits)`` array.

        Returns
        -------
        params : ``(n_units,)`` for scalar families;
                 ``(n_units, n_params)`` for multi-parameter families.
            Each row ``q_i`` is the distribution parameter vector for unit *i*.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            p = self.fit_unit(Y)
            return p[0] if self.n_params == 1 else p

        rows = np.vstack([self.fit_unit(Y[i]) for i in range(Y.shape[0])])
        return rows.ravel() if self.n_params == 1 else rows


# ─────────────────────────────────────────────────────────────────────────────
# Q-variable density estimator (meta-distribution over parameter space)
# ─────────────────────────────────────────────────────────────────────────────

class QDensityEstimator:
    """
    Estimate the (conditional) density of a Q-variable.

    A Q-variable ``Q^v`` is *distribution-valued*: its realisation at unit *i*
    is the parameter vector ``q_i`` of the distribution of the subunit variable
    *v* within that unit (output of :class:`SubunitParamEstimator`).

    This class estimates ``P(Q^v = q)`` and ``P(Q^v = q | X = x)`` using
    kernel density estimation in the parameter space.  The conditional variant
    uses a residual-KDE approach: regress Q on X with linear regression, then
    fit a KDE on the regression residuals so that
    ``P(Q = q | X = x) = KDE(q − Ê[Q|X=x])``.

    Parameters
    ----------
    bandwidth : float or ``'scott'`` or ``'silverman'``, default ``'scott'``
        Bandwidth for the KDE.  ``'scott'`` and ``'silverman'`` use the
        standard rules of thumb.

    Examples
    --------
    Estimate the marginal density of Q^{y|a} (Gaussian family):

    >>> spe = SubunitParamEstimator(family="gaussian")
    >>> q_params = spe.fit(Y_subunit)          # (n_units, 2) array
    >>> qde = QDensityEstimator()
    >>> qde.fit(q_params)
    >>> p = qde.prob(np.array([0.5, 0.1]))     # P(μ=0.5, σ²=0.1)

    Conditional density given unit-level covariate Q^a:

    >>> qde_cond = QDensityEstimator()
    >>> qde_cond.fit(q_params, x_cond=Q_a)
    >>> p_cond = qde_cond.prob(np.array([0.5, 0.1]), x_val=np.array([0.7]))
    """

    def __init__(self, bandwidth: Union[float, str] = "scott") -> None:
        self.bandwidth = bandwidth
        self.family = "q_density"  # we tag meta-estimator; vector path uses isinstance(QDensityEstimator)
        self._q_samples: Optional[np.ndarray] = None    # (n, d)
        self._x_samples: Optional[np.ndarray] = None    # (n, p)
        self._regressor = None
        self._residuals: Optional[np.ndarray] = None    # (n, d)
        self._kde = None
        self._fallback_mean: Optional[np.ndarray] = None
        self._fallback_std: Optional[np.ndarray] = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        q_params: np.ndarray,
        x_cond: Optional[np.ndarray] = None,
    ) -> "QDensityEstimator":
        """
        Learn the distribution of Q-variable parameter vectors.

        Parameters
        ----------
        q_params : ``(n_units,)`` or ``(n_units, n_params)`` array
            Per-unit distribution parameters (from :class:`SubunitParamEstimator`).
        x_cond : ``(n_units,)`` or ``(n_units, n_features)`` array, optional
            Unit-level conditioning variables.  Enables ``P(Q | X = x)``
            queries via a residual-KDE approach.

        Returns
        -------
        self
        """
        q = np.asarray(q_params, dtype=float)
        if q.ndim == 1:
            q = q.reshape(-1, 1)
        self._q_samples = q
        residuals = q.copy()

        if x_cond is not None:
            x = np.asarray(x_cond, dtype=float)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            self._x_samples = x
            if SKLEARN_AVAILABLE:
                from sklearn.linear_model import LinearRegression
                self._regressor = LinearRegression().fit(x, q)
                residuals = q - self._regressor.predict(x)
        self._residuals = residuals
        self._fit_kde(residuals)
        return self

    def _fit_kde(self, samples: np.ndarray) -> None:
        if not SCIPY_AVAILABLE:
            self._fallback_mean = samples.mean(axis=0)
            self._fallback_std = np.maximum(samples.std(axis=0, ddof=1), 1e-8)
            return
        if samples.ndim != 2 or samples.shape[0] <= samples.shape[1]:
            # KDE in d dimensions needs comfortably more than d unit samples;
            # otherwise scipy rightfully fails with a singular covariance matrix.
            self._fallback_mean = samples.mean(axis=0)
            self._fallback_std = np.maximum(samples.std(axis=0, ddof=1), 1e-8)
            return
        try:
            self._kde = _sp_stats.gaussian_kde(
                samples.T, bw_method=self.bandwidth
            )
        except (np.linalg.LinAlgError, ValueError):
            # Singular covariance (e.g., constant column) → Gaussian fallback
            self._fallback_mean = samples.mean(axis=0)
            self._fallback_std = np.maximum(samples.std(axis=0, ddof=1), 1e-8)

    # ── Density evaluation ────────────────────────────────────────────────────

    def log_prob(
        self,
        q_val: np.ndarray,
        x_val: Optional[np.ndarray] = None,
    ) -> float:
        """
        Log-density at ``Q = q_val``, optionally conditional on ``X = x_val``.

        Parameters
        ----------
        q_val : array-like of shape ``(n_params,)``
            The parameter vector at which to evaluate the density.
        x_val : array-like, optional
            Conditioning variable values (unit-level).

        Returns
        -------
        float
            ``log P(Q = q_val | X = x_val)``.
        """
        q = np.asarray(q_val, dtype=float).ravel()
        residual = q.copy()

        if x_val is not None and self._regressor is not None:
            x = np.asarray(x_val, dtype=float).ravel().reshape(1, -1)
            residual = q - self._regressor.predict(x).ravel()

        if self._kde is not None:
            val = float(self._kde.evaluate(residual.reshape(-1, 1))[0])
            return float(np.log(max(val, 1e-300)))

        # Gaussian fallback
        z = (residual - self._fallback_mean) / self._fallback_std
        return float(
            -0.5 * np.sum(z ** 2)
            - np.sum(np.log(self._fallback_std))
            - 0.5 * len(z) * np.log(2.0 * np.pi)
        )

    def prob(
        self,
        q_val: np.ndarray,
        x_val: Optional[np.ndarray] = None,
    ) -> float:
        """Density ``P(Q = q_val | X = x_val)``."""
        return float(np.exp(self.log_prob(q_val, x_val)))

    # ── Moments ───────────────────────────────────────────────────────────────

    def mean(self, x_val: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ``E[Q | X = x_val]`` as a parameter vector.

        Returns
        -------
        np.ndarray of shape ``(n_params,)``
        """
        if x_val is not None and self._regressor is not None:
            x = np.asarray(x_val, dtype=float).ravel().reshape(1, -1)
            return self._regressor.predict(x).ravel()
        return self._q_samples.mean(axis=0)

    def scalar_mean(self, x_val: Optional[np.ndarray] = None) -> float:
        """
        First component of ``E[Q | X = x_val]``.

        For scalar families (Bernoulli, Poisson) this equals ``E[Q | X]``.
        For multi-parameter families (Gaussian) it returns ``E[μ | X]``.
        """
        return float(self.mean(x_val)[0])

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        x_val: Optional[np.ndarray] = None,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw *n* samples from ``P(Q | X = x_val)``.

        Returns
        -------
        np.ndarray of shape ``(n, n_params)``
        """
        if rng is None:
            rng = np.random.default_rng()
        mean = self.mean(x_val)
        if self._residuals is not None and len(self._residuals) > 0:
            idx = rng.integers(0, len(self._residuals), size=n)
            return mean + self._residuals[idx]
        std = (self._fallback_std if self._fallback_std is not None
               else np.ones_like(mean) * 0.1)
        return mean + rng.normal(0.0, 1.0, (n, len(mean))) * std

    # ── Convenience: expectation of a function of Q ───────────────────────────

    def expectation(
        self,
        x_val: Optional[np.ndarray] = None,
        n_mc: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Estimate ``E[Q_first_param | X = x_val]`` (primary distribution parameter).

        This is used by the formula evaluator as a scalar proxy for the
        Q-variable (e.g., the Bernoulli probability or Gaussian mean).
        """
        return self.scalar_mean(x_val)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Q-variable computation & name resolution
# ─────────────────────────────────────────────────────────────────────────────

def compute_q_from_subunit_data(
    data: Dict[str, np.ndarray],
    families: Optional[Dict[str, str]] = None,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Enrich *data* with Q-variable arrays from raw subunit observations.

    For each key *v* whose value is a 2-D array ``(n_units, n_subunits)``,
    adds the corresponding Q-variable entry via :class:`SubunitParamEstimator`:

    * **Scalar families** (``bernoulli``, ``poisson``, ``exponential``):
      ``Q^v`` is the per-unit parameter as a 1-D array ``(n_units,)``.

    * **Multi-parameter families** (``gaussian`` → (μ_i, σ²_i); ``beta`` →
      (α_i, β_i); ``beta_unit_minmax`` → (α_i, β_i) on within-unit rescaled scores; …):
      ``Q^v`` is a 2-D array ``(n_units, n_params)``.
      The formula evaluator will use :class:`QDensityEstimator` for terms
      involving such variables.

    Pre-existing entries in *data* are never overwritten.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Raw data.  2-D values are subunit matrices; 1-D values are
        unit-level arrays.
    families : dict[str, str], optional
        Maps variable names to distribution families.
        Defaults to ``'bernoulli'`` for 2-D variables without an explicit entry.
    """
    families = families or {}
    result = dict(data)
    tasks: list[tuple[str, np.ndarray, str]] = []
    torch_kwargs = dict(torch_kwargs or {})
    estimator_backend = estimator_backend.lower().strip()
    local_n_jobs = 1 if estimator_backend == "numpyro" else n_jobs

    for key, arr in data.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim != 2:
            continue

        # Derive canonical Q-variable name: Q^{base}
        base = key.lower()
        q_key = f"Q^{{{base}}}" if len(base) > 1 else f"Q^{base}"
        # Fall back to the older simple convention for single-char keys
        if q_key not in result:
            q_key_simple = f"Q^{base}"
            if q_key_simple in result:
                continue
            # Use the simpler name for single-char keys (A → Q^a, Y → Q^y)
            q_key = q_key_simple

        if q_key in result:
            continue  # already provided by user

        family = families.get(key, families.get(q_key, "bernoulli"))
        if estimator_backend == "torch" and family in {
            "bernoulli", "poisson", "gaussian", "normal", "beta", "beta_unit_minmax", "gamma",
        }:
            if q_key not in result:
                result[q_key] = torch_compute_subunit_params(
                    arr_np,
                    family=family,
                    device=resolve_torch_device_from_mapping(torch_kwargs),
                    devices=torch_kwargs.get("devices"),
                )
            continue
        local_estimator_kwargs = _resolve_estimator_kwargs(
            estimator_kwargs,
            variable_name=key,
            family=family,
        )
        tasks.append((q_key, arr_np, family, local_estimator_kwargs))

    for q_key, q_values in parallel_map(
        tasks,
        _fit_q_from_subunit_task,
        n_jobs=local_n_jobs,
        backend=parallel_backend,
    ):
        if q_key not in result:
            result[q_key] = q_values

    return result


def _sanitize(name: str) -> str:
    """Sanitize paper notation to pyAgrum-compatible identifier."""
    return name.replace("^", "").replace("{", "").replace("}", "").replace("|", "_")


def _build_name_maps(keys: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return (sanitized→paper, paper→sanitized) bidirectional maps.
    """
    san_to_paper: Dict[str, str] = {}
    paper_to_san: Dict[str, str] = {}
    for k in keys:
        s = _sanitize(k)
        san_to_paper[s] = k
        paper_to_san[k] = s
    return san_to_paper, paper_to_san


# ─────────────────────────────────────────────────────────────────────────────
# Formula AST data structures (internal representation for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class _ASTFormula:
    """Base class for our internal formula representation."""
    def collect_conditionals(self) -> List[Dict]:
        return []


class _ASTLeaf(_ASTFormula):
    def __init__(self, value: float = 1.0):
        self.value = value


class _ASTConditional(_ASTFormula):
    """Represents P(outcome_vars | cond_vars)."""
    def __init__(self, outcome_vars: List[str], cond_vars: List[str]):
        self.outcome_vars = outcome_vars
        self.cond_vars = cond_vars

    def key(self) -> Tuple:
        return ("P", tuple(sorted(self.outcome_vars)), tuple(sorted(self.cond_vars)))

    def collect_conditionals(self) -> List[Dict]:
        return [{"outcome_vars": self.outcome_vars, "cond_vars": self.cond_vars, "key": self.key()}]


class _ASTProduct(_ASTFormula):
    def __init__(self, children: List[_ASTFormula]):
        self.children = children

    def collect_conditionals(self) -> List[Dict]:
        out = []
        for c in self.children:
            out.extend(c.collect_conditionals())
        return out


class _ASTSum(_ASTFormula):
    """Represents Σ_{sum_vars} formula."""
    def __init__(self, sum_vars: List[str], formula: _ASTFormula):
        self.sum_vars = sum_vars
        self.formula = formula

    def collect_conditionals(self) -> List[Dict]:
        return self.formula.collect_conditionals()


# ─────────────────────────────────────────────────────────────────────────────
# AST extractor: pyAgrum introspection + LaTeX fallback
# ─────────────────────────────────────────────────────────────────────────────

def _get(obj, *attr_names, default=None):
    """Try multiple attribute names; return first found."""
    for name in attr_names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _walk_pyagrum_node(node) -> _ASTFormula:
    """
    Recursively walk a pyAgrum AST node using duck-typing / introspection.

    pyAgrum's causal AST classes (internal to pyAgrum):
      - Summation  : type name contains "sum"
      - Product    : type name contains "prod"
      - Conditional: type name contains "posterior" or "conditional"
      - Joint prob : type name contains "joint" or "proba"
    """
    node_type = type(node).__name__.lower()

    # ---- Summation -----------------------------------------------------------
    if "sum" in node_type:
        raw_vars = _get(node, "_varnames", "vars", "variables", default=[])
        if isinstance(raw_vars, str):
            raw_vars = [raw_vars]
        elif raw_vars is None:
            raw_vars = []
        elif hasattr(raw_vars, "tolist") and not isinstance(raw_vars, (list, tuple)):
            raw_vars = raw_vars.tolist()
        sum_vars = [str(v) for v in raw_vars]
        term_node = _get(node, "_term", "term", "child", "expr", "formula")
        inner = _walk_pyagrum_node(term_node) if term_node is not None else _ASTLeaf(1.0)
        return _ASTSum(sum_vars, inner)

    # ---- Product -------------------------------------------------------------
    if "prod" in node_type:
        children_attr = _get(node, "_terms", "terms", "children", "factors", default=[])
        return _ASTProduct([_walk_pyagrum_node(c) for c in children_attr])

    # ---- Conditional P(Y|X) --------------------------------------------------
    if "posterior" in node_type or "conditional" in node_type:
        out_vars = _get(node, "_vars", "_nodes", "nodes", "outcome", default=[])
        cond_vars = _get(node, "_cond", "cond", "given", "conditioning", default=[])
        if isinstance(out_vars, str):
            out_vars = [out_vars]
        if isinstance(cond_vars, str):
            cond_vars = [cond_vars]
        return _ASTConditional(list(out_vars), list(cond_vars))

    # ---- Joint P(X, Y) -------------------------------------------------------
    if "joint" in node_type or "proba" in node_type:
        out_vars = _get(node, "_vars", "_nodes", "nodes", "variables", default=[])
        cond_vars = _get(node, "_cond", "cond", "given", default=[])
        if isinstance(out_vars, str):
            out_vars = [out_vars]
        if isinstance(cond_vars, str):
            cond_vars = [cond_vars]
        return _ASTConditional(list(out_vars), list(cond_vars))

    # ---- Binary operation (+, *, /) ------------------------------------------
    if "binary" in node_type or "op" in node_type:
        op = _get(node, "_op", "op", default="*")
        left = _get(node, "_left", "left")
        right = _get(node, "_right", "right")
        l_f = _walk_pyagrum_node(left) if left is not None else _ASTLeaf(1.0)
        r_f = _walk_pyagrum_node(right) if right is not None else _ASTLeaf(1.0)
        if op == "+":
            # Treat as sum with no summation variable
            return _ASTSum([], _ASTProduct([l_f, r_f]))
        return _ASTProduct([l_f, r_f])

    # ---- Unknown: try repr ---------------------------------------------------
    node_repr = repr(node)
    cond_m = re.search(r"P\(([^|)]+)\|([^)]+)\)", node_repr)
    joint_m = re.search(r"P\(([^)]+)\)", node_repr)
    if cond_m:
        outs = [v.strip() for v in cond_m.group(1).split(",")]
        conds = [v.strip() for v in cond_m.group(2).split(",")]
        return _ASTConditional(outs, conds)
    if joint_m:
        outs = [v.strip() for v in joint_m.group(1).split(",")]
        return _ASTConditional(outs, [])

    return _ASTLeaf(1.0)


def _parse_latex_formula(latex: str) -> _ASTFormula:
    """
    Parse a LaTeX identification formula string into our internal AST.

    Handles patterns like::

        \\sum_{U} P(Qy|Qa,U) \\cdot P(U)
    """
    # Normalise
    s = latex
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\mid", "|")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\,", " ")

    # Extract all P(Y|X) and P(Y) terms
    prob_terms: List[_ASTFormula] = []
    for m in re.finditer(r"P\(([^|)]+)(?:\|([^)]+))?\)", s):
        outs = [v.strip() for v in m.group(1).split(",") if v.strip()]
        conds = [v.strip() for v in m.group(2).split(",") if v.strip()] if m.group(2) else []
        prob_terms.append(_ASTConditional(outs, conds))

    # Extract summation variables
    sum_vars: List[str] = []
    for m in re.finditer(r"\\sum_\{([^}]+)\}", s):
        sum_vars.extend(v.strip() for v in m.group(1).split(",") if v.strip())
    for m in re.finditer(r"\\sum_([A-Za-z_]+)", s):
        sum_vars.append(m.group(1).strip())

    if not prob_terms:
        return _ASTLeaf(1.0)

    body: _ASTFormula = prob_terms[0] if len(prob_terms) == 1 else _ASTProduct(prob_terms)
    if sum_vars:
        body = _ASTSum(sum_vars, body)
    return body


def _extract_formula(ast) -> _ASTFormula:
    """
    Extract our internal formula representation from a pyAgrum AST object.

    Tries three strategies in order:
      1. Direct introspection of ``._root`` / ``.root`` attribute.
      2. Calling ``._formula`` or ``.formula`` method/attribute.
      3. Parsing the ``.toLatex()`` string.
    """
    # Strategy 1: root node introspection
    try:
        root = _get(ast, "_root", "root", "formula", "_formula")
        if root is not None and not callable(root):
            return _walk_pyagrum_node(root)
    except Exception:
        pass

    # Strategy 2: iterate children of ast directly
    try:
        children = list(ast)
        if children:
            return _ASTProduct([_walk_pyagrum_node(c) for c in children])
    except Exception:
        pass

    # Strategy 3: parse LaTeX
    try:
        latex = ast.toLatex() if hasattr(ast, "toLatex") else str(ast)
        return _parse_latex_formula(latex)
    except Exception:
        return _ASTLeaf(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Formula evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _data_first_lookup(data: Dict[str, Any], k1: Optional[str], k2: Optional[str]) -> Any:
    """we fetch data[k1] or data[k2] without ``a or b`` (ndarray truthiness is undefined)."""
    if k1 is not None and k1 in data:
        return data[k1]
    if k2 is not None and k2 in data:
        return data[k2]
    return None


def _formula_scalar_y(y_val: Any) -> Optional[float]:
    """we map context outcome value to a scalar point mass, or None to use E[Y|X]."""
    if y_val is None:
        return None
    a = np.asarray(y_val, dtype=float).ravel()
    if a.size != 1:
        return None
    return float(a[0])


def _batch_arr_length_n(
    arr: Any,
    n: int,
    est: "ConditionalDensityEstimator",
    X_batch: np.ndarray,
) -> np.ndarray:
    """we coerce batch outputs to shape (n,); on mismatch we fall back to per-row expectation."""
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.shape == (n,):
        return a
    return np.array([float(est.expectation(X_batch[i])) for i in range(n)])


def _eval_formula(
    node: _ASTFormula,
    context: Dict[str, float],
    fitted: Dict[tuple, ConditionalDensityEstimator],
    data: Dict[str, np.ndarray],
    resolve: Callable[[str], Optional[str]],
    n_mc: int,
    rng: np.random.Generator,
    unit_n: Optional[int] = None,
) -> float:
    """
    Recursively evaluate a formula node given a ``context`` of variable values.

    For ``_ASTConditional`` nodes: if the outcome variable is *not* in *context*,
    return ``E[Y | X=x]`` (expectation); otherwise return ``P(Y=y | X=x)``.
    For ``_ASTSum`` nodes: enumerate discrete supports or draw Monte Carlo samples.
    """
    if isinstance(node, _ASTLeaf):
        return node.value

    if isinstance(node, _ASTConditional):
        key = node.key()
        est = fitted.get(key)
        if est is None:
            return 1.0

        # Collect conditioning values from context (must match ast_to_estimator columns).
        x_vals: List[float] = []
        cond_key_pairs = getattr(est, "_hcm_cond_parent_keys", None)
        if cond_key_pairs:
            for cv_paper, raw_cv in cond_key_pairs:
                val = context.get(cv_paper, context.get(raw_cv))
                if val is None:
                    d = _data_first_lookup(data, cv_paper, raw_cv)
                    if d is not None:
                        d_np = np.asarray(d, dtype=float)
                        val = float(d_np.mean() if d_np.ndim == 1 else d_np[:, -1].mean())
                    else:
                        val = 0.0
                x_vals.append(float(val))
        else:
            for cv in node.cond_vars:
                cv_paper = resolve(cv) or cv
                val = context.get(cv_paper, context.get(cv))
                if val is None:
                    d = _data_first_lookup(data, cv_paper, cv)
                    if d is not None:
                        d_np = np.asarray(d, dtype=float)
                        val = float(d_np.mean() if d_np.ndim == 1 else d_np[:, -1].mean())
                    else:
                        val = 0.0
                x_vals.append(float(val))
        x_q = np.array(x_vals, dtype=float) if x_vals else None

        # QDensityEstimator: return scalar_mean(X) as the expected Q value
        if isinstance(est, QDensityEstimator):
            return est.scalar_mean(x_q)

        # Standard ConditionalDensityEstimator path
        out_paper = resolve(node.outcome_vars[0]) if node.outcome_vars else None
        y_val = context.get(out_paper) if out_paper else None
        if y_val is None and node.outcome_vars:
            y_val = context.get(node.outcome_vars[0])

        y_pt = _formula_scalar_y(y_val)
        if y_pt is not None:
            return est.prob(y_pt, x_q)
        return est.expectation(x_q, n_mc=n_mc)

    if isinstance(node, _ASTProduct):
        result = 1.0
        for child in node.children:
            result *= _eval_formula(child, context, fitted, data, resolve, n_mc, rng, unit_n)
        return result

    if isinstance(node, _ASTSum):
        total = 0.0
        # If no explicit summation variables, just evaluate body
        if len(node.sum_vars) == 0:
            return _eval_formula(node.formula, context, fitted, data, resolve, n_mc, rng, unit_n)
        # Evaluate by marginalising over each summation variable
        for sv in node.sum_vars:
            sv_paper = resolve(sv) or sv
            sv_data_arr = _data_first_lookup(data, sv_paper, sv)
            if sv_data_arr is None:
                # No data → evaluate without marginalisation
                total += _eval_formula(node.formula, context, fitted, data, resolve, n_mc, rng, unit_n)
                continue
            sv_np = np.asarray(sv_data_arr, dtype=float)
            if sv_np.ndim == 2:
                # 2D: each row is one unit's full conditional profile
                # (e.g. Q^{a|z} evaluated at each unique z value).
                # Sample rows to preserve the joint profile per unit.
                mc_idx = rng.integers(0, sv_np.shape[0], size=n_mc)
                mc_samples_nd = sv_np[mc_idx]  # shape (n_mc, n_cols)
                batch_result = _eval_formula_vec(node.formula, context, sv_paper,
                                                 mc_samples_nd, fitted, data, resolve,
                                                 cancel_marginal=sv_paper, unit_n=unit_n)
                total += float(np.mean(batch_result))
            else:
                sv_arr = sv_np.ravel()
                unique_vals = np.unique(sv_arr)
                is_discrete = np.all(sv_arr == sv_arr.astype(int)) and len(unique_vals) <= 30

                if is_discrete:
                    for val in unique_vals:
                        new_ctx = {**context, sv_paper: float(val), sv: float(val)}
                        total += _eval_formula(node.formula, new_ctx, fitted, data, resolve, n_mc, rng, unit_n)
                else:
                    # Monte Carlo: sample u_k ~ empirical p(sv), compute mean of body(u_k).
                    # This correctly estimates ∫ body(u) p(u) du  =  E_{u~p}[body(u)].
                    # We must NOT multiply by p(u_k) again inside the body; pass sv_paper as the
                    # "marginal being cancelled" so _eval_formula_vec skips the P(sv) factor.
                    mc_idx = rng.integers(0, len(sv_arr), size=n_mc)
                    mc_samples = sv_arr[mc_idx]
                    batch_result = _eval_formula_vec(node.formula, context, sv_paper, mc_samples,
                                                     fitted, data, resolve, cancel_marginal=sv_paper,
                                                     unit_n=unit_n)
                    total += float(np.mean(batch_result))

        return total

    return 1.0


def _batch_expectation(est: ConditionalDensityEstimator, X_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised conditional expectation E[Y | X=x] for each row of X_batch.

    Uses batch sklearn predict where possible; falls back to per-sample loop.
    """
    n = len(X_batch)

    def _linreg_predict(lr_attr: str) -> Optional[np.ndarray]:
        lr = getattr(est, lr_attr, None)
        if lr is not None and hasattr(lr, "predict"):
            return lr.predict(X_batch)
        return None

    fam = est.family

    if fam == "gaussian":
        if hasattr(est, "_torch_gaussian_state"):
            x3 = np.asarray(X_batch, dtype=np.float32)[None, :, :]
            return torch_predict_batched_gaussian(est._torch_gaussian_state, x3)[0]
        pred = _linreg_predict("_lr_gauss")
        arr = pred if pred is not None else np.full(n, est._mu_marginal)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "gaussian_mixture":
        return np.array([float(est._predict_mu_gaussian_mixture(X_batch[i])) for i in range(n)])

    if fam == "bernoulli":
        if hasattr(est, "_torch_bernoulli_state"):
            x3 = np.asarray(X_batch, dtype=np.float32)[None, :, :]
            return torch_predict_batched_bernoulli(est._torch_bernoulli_state, x3)[0]
        if est._lr_model is None:
            arr = np.full(n, est._p_marginal)
        elif hasattr(est._lr_model, "predict_proba"):
            arr = logistic_positive_proba_batch(est._lr_model, X_batch)
        else:
            arr = np.clip(est._lr_model.predict(X_batch), 0.0, 1.0)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "poisson":
        if hasattr(est, "_torch_poisson_state"):
            x3 = np.asarray(X_batch, dtype=np.float32)[None, :, :]
            return torch_predict_batched_poisson(est._torch_poisson_state, x3)[0]
        lr = getattr(est, "_lr_poisson", None)
        if lr is None:
            arr = np.full(n, est._lambda_marginal)
        elif isinstance(lr, tuple):  # log-linear fallback
            arr = np.exp(np.clip(lr[1].predict(X_batch), -10, 10))
        else:
            arr = np.maximum(lr.predict(X_batch), 1e-9)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "laplace":
        pred = _linreg_predict("_lr_laplace")
        arr = pred if pred is not None else np.full(n, est._laplace_loc)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "student_t":
        pred = _linreg_predict("_lr_t")
        arr = pred if pred is not None else np.full(n, est._t_loc)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "exponential":
        pred = _linreg_predict("_lr_exp")
        arr = np.maximum(pred, 1e-9) if pred is not None else np.full(n, 1.0 / (est._exp_lambda + 1e-9))
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "gamma":
        if hasattr(est, "_torch_gamma_state"):
            x3 = np.asarray(X_batch, dtype=np.float32)[None, :, :]
            return torch_predict_batched_gamma_mean(est._torch_gamma_state, x3)[0]
        pred = _linreg_predict("_lr_gamma")
        arr = np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._gamma_mean)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "lognormal":
        pred = _linreg_predict("_lr_lognormal")
        log_mu = pred if pred is not None else np.full(n, est._logn_mu)
        arr = np.exp(log_mu + 0.5 * est._logn_sigma ** 2)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "weibull":
        pred = _linreg_predict("_lr_weibull")
        arr = np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._weibull_mean)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "inverse_gaussian":
        pred = _linreg_predict("_lr_ig")
        arr = np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._ig_mu)
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "beta":
        if hasattr(est, "_torch_beta_state"):
            x3 = np.asarray(X_batch, dtype=np.float32)[None, :, :]
            return torch_predict_batched_beta_mean(est._torch_beta_state, x3)[0]
        lr = getattr(est, "_lr_beta", None)
        if lr is None:
            arr = np.full(n, est._beta_mu)
        else:
            logit_pred = linear_predict_batch(lr, X_batch)
            arr = expit_array(np.asarray(logit_pred, dtype=np.float64))
        return _batch_arr_length_n(arr, n, est, X_batch)

    if fam == "half_cauchy":
        pred = _linreg_predict("_lr_hc")
        arr = np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._hc_scale)
        return _batch_arr_length_n(arr, n, est, X_batch)

    # Non-parametric: per-sample loop
    return np.array([float(est.expectation(X_batch[i])) for i in range(n)])


def _eval_formula_vec(
    node: _ASTFormula,
    context: Dict[str, float],
    sv_name: str,
    sv_values: np.ndarray,
    fitted: Dict[tuple, ConditionalDensityEstimator],
    data: Dict[str, np.ndarray],
    resolve: Callable[[str], Optional[str]],
    cancel_marginal: Optional[str] = None,
    unit_n: Optional[int] = None,
) -> np.ndarray:
    """
    Vectorised formula evaluator for a batch of summation variable values.

    Evaluates ``node`` at each value in ``sv_values``, using ``context`` for all
    other variable assignments.  Returns an array of shape ``(len(sv_values),)``.

    Parameters
    ----------
    cancel_marginal : str or None
        When set to the name of the variable being marginalised via Monte Carlo,
        any ``_ASTConditional`` node whose *sole outcome* is that variable and has
        *no conditioning parents* (i.e. P(sv)) is replaced by 1.0.  This corrects
        the double-counting that arises when MC samples u_k ~ p(sv) already encode
        the marginal weight, so multiplying by p(u_k) again would be incorrect.
    """
    n = len(sv_values)

    if isinstance(node, _ASTLeaf):
        return np.full(n, node.value)

    if isinstance(node, _ASTConditional):
        # Cancel marginal P(sv) when sampling is already done from P(sv)
        if cancel_marginal is not None and len(node.cond_vars) == 0 and len(node.outcome_vars) == 1:
            out_paper = resolve(node.outcome_vars[0]) or node.outcome_vars[0]
            if out_paper == cancel_marginal or node.outcome_vars[0] == cancel_marginal:
                return np.ones(n)  # this term is handled by the MC sampling weights

        key = node.key()
        est = fitted.get(key)
        if est is None:
            return np.ones(n)

        # Build X matrix for the full batch (same parents as at fit time when metadata exists).
        x_cols: List[np.ndarray] = []
        cond_key_pairs_vec = getattr(est, "_hcm_cond_parent_keys", None)
        if cond_key_pairs_vec:
            for cv_paper, raw_cv in cond_key_pairs_vec:
                if cv_paper == sv_name or raw_cv == sv_name:
                    if hasattr(sv_values, "ndim") and sv_values.ndim == 2:
                        x_cols.append(sv_values[:, -1])
                    else:
                        x_cols.append(sv_values)
                else:
                    val = context.get(cv_paper, context.get(raw_cv))
                    if val is None:
                        d = _data_first_lookup(data, cv_paper, raw_cv)
                        if d is not None:
                            d_np = np.asarray(d, dtype=float)
                            if unit_n is not None and d_np.shape[0] != unit_n:
                                val = 0.0
                            else:
                                val = float(d_np.mean() if d_np.ndim == 1 else d_np[:, -1].mean())
                        else:
                            val = 0.0
                    x_cols.append(np.full(n, float(val)))
        else:
            for cv in node.cond_vars:
                cv_paper = resolve(cv) or cv
                if cv_paper == sv_name or cv == sv_name:
                    if hasattr(sv_values, "ndim") and sv_values.ndim == 2:
                        x_cols.append(sv_values[:, -1])
                    else:
                        x_cols.append(sv_values)
                else:
                    val = context.get(cv_paper, context.get(cv))
                    if val is None:
                        d = _data_first_lookup(data, cv_paper, cv)
                        if d is not None:
                            d_np = np.asarray(d, dtype=float)
                            if unit_n is not None and d_np.shape[0] != unit_n:
                                continue
                            val = float(np.mean(d_np))
                        else:
                            val = 0.0
                    x_cols.append(np.full(n, float(val)))
        X_batch = np.column_stack(x_cols) if x_cols else None

        # Outcome: check if a point-probability or expectation is requested
        out_paper = resolve(node.outcome_vars[0]) if node.outcome_vars else None
        y_val = context.get(out_paper) if out_paper else None
        if y_val is None and node.outcome_vars:
            y_val = context.get(node.outcome_vars[0])
        y_f = _formula_scalar_y(y_val)

        # we mirror _eval_formula: Q-variable KDE terms contribute E[Q|X] only (no P(Q=q) in vec path)
        if isinstance(est, QDensityEstimator):
            if X_batch is not None:
                return np.array([float(est.scalar_mean(X_batch[i])) for i in range(n)])
            return np.full(n, float(est.scalar_mean(None)))

        if y_f is not None:
            if est.family == "bernoulli":
                p = _batch_expectation(est, X_batch) if X_batch is not None else np.full(n, est._p_marginal)
                return p if int(round(y_f)) == 1 else (1.0 - p)
            if est.family == "gaussian":
                mu = _batch_expectation(est, X_batch) if X_batch is not None else np.full(n, est._mu_marginal)
                if SCIPY_AVAILABLE:
                    return _sp_stats.norm.pdf(y_f, loc=mu, scale=est._sigma)
                z = (y_f - mu) / est._sigma
                return np.exp(-0.5 * z * z) / (est._sigma * np.sqrt(2 * np.pi))
            return np.array([est.prob(y_f, X_batch[i] if X_batch is not None else None) for i in range(n)])
        if X_batch is not None:
            return _batch_expectation(est, X_batch)
        return np.full(n, est.expectation(None))

    if isinstance(node, _ASTProduct):
        result = np.ones(n)
        for child in node.children:
            result = result * _eval_formula_vec(child, context, sv_name, sv_values,
                                                fitted, data, resolve, cancel_marginal, unit_n)
        return result

    if isinstance(node, _ASTSum):
        # Nested sum: scalar fallback per sv_value (2D batch → one scalar per row, last column)
        if getattr(sv_values, "ndim", 0) == 2:
            return np.array([
                _eval_formula(
                    node,
                    {**context, sv_name: float(np.asarray(row, dtype=float).ravel()[-1])},
                    fitted,
                    data,
                    resolve,
                    50,
                    np.random.default_rng(
                        int(abs(float(np.asarray(row, dtype=float).ravel()[-1])) * 1e6) % (2**31)
                    ),
                    unit_n,
                )
                for row in sv_values
            ])
        return np.array([
            _eval_formula(node, {**context, sv_name: float(v)}, fitted, data,
                          resolve, 50, np.random.default_rng(int(abs(v) * 1e6) % (2**31)), unit_n)
            for v in sv_values
        ])

    return np.ones(n)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Q-variable precomputation
# ─────────────────────────────────────────────────────────────────────────────

def _collect_all_formula_vars(formula: _ASTFormula) -> List[str]:
    """Collect all variable names referenced anywhere in the formula."""
    if isinstance(formula, _ASTConditional):
        return list(formula.outcome_vars) + list(formula.cond_vars)
    if isinstance(formula, _ASTSum):
        result = list(formula.sum_vars)
        result.extend(_collect_all_formula_vars(formula.formula))
        return result
    if isinstance(formula, _ASTProduct):
        result = []
        for child in formula.children:
            result.extend(_collect_all_formula_vars(child))
        return result
    return []


def _collect_sum_vars(formula: _ASTFormula) -> set:
    """Collect only variables that appear as explicit summation variables in the formula."""
    if isinstance(formula, _ASTSum):
        result = set(formula.sum_vars)
        result.update(_collect_sum_vars(formula.formula))
        return result
    if isinstance(formula, _ASTProduct):
        result = set()
        for child in formula.children:
            result.update(_collect_sum_vars(child))
        return result
    return set()


def _precompute_conditional_q_vars(
    data: Dict[str, np.ndarray],
    formula: _ASTFormula,
    families: Dict[str, str],
    iv_val: float,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Precompute conditional Q-variables (e.g. Q^{y|a}) found in the formula.

    Traverses the formula to find all variable names (sum vars, outcome vars,
    and conditioning vars in _ASTConditional nodes).
    For each variable matching pattern Q[outcome]_[cond] (e.g. Qy_a) **or**
    ``Q[outcome]_[p1]_[p2]_…`` (e.g. ``Qy_g_l_m`` → :math:`Q^{y|g,l,m}`):
      - Single parent: same as before — per-unit fit on one conditioner, evaluate
        at ``iv_val`` (treatment level for :math:`Q^{y|a}`-style symbols).
      - **Multiple parents**: per-unit fit on stacked subunit covariates; store
        the **mean over subunits** of conditional expectations E[Y | X_ij] (no
        ``iv_val`` injection on non-A parents). This fills ``enriched`` so
        identification factors are not silently replaced by ``1``; it is a
        pragmatic summary, not a full interventional profile for every parent.
        If such a symbol is also a **summation** variable in the formula, the
        same scalar summary is used (full multi-column Σ-profiles are not
        implemented).

    Returns a dict of new entries to add to enriched data.
    """
    all_vars = _collect_all_formula_vars(formula)
    # Variables that are explicit summation targets get a full 2D conditional profile
    # (evaluated at all unique conditioning values).  Variables that only appear as
    # outcomes or conditioning terms get a scalar (1D) estimate evaluated at iv_val.
    formula_sum_vars = _collect_sum_vars(formula)

    # Use a set to avoid redundant work
    seen: set = set()
    deduped_vars = [v for v in all_vars if v not in seen and not seen.add(v)]  # type: ignore
    new_entries: Dict[str, np.ndarray] = {}
    torch_kwargs = dict(torch_kwargs or {})
    estimator_backend = estimator_backend.lower().strip()
    local_parallel_backend: ParallelBackend = parallel_backend
    local_n_jobs = n_jobs
    if estimator_backend == "numpyro" and parallel_backend == "threads":
        local_parallel_backend = "processes"
    if estimator_backend == "numpyro":
        # Variational fits already use JAX/XLA internally; spawning multiple Python
        # workers around them is both memory-hungry and unstable on a single GPU.
        local_n_jobs = 1

    disable_multiparent_fix = str(os.environ.get("HCM_DISABLE_MULTIPARENT_Q_PRECOMPUTE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    for sv in deduped_vars:
        m = re.match(r"^Q([a-zA-Z]+)_(.+)$", sv)
        if not m:
            continue
        outcome_letters = m.group(1).lower()
        rest = m.group(2).lower()
        parent_tokens = [p for p in rest.split("_") if p]
        if not parent_tokens:
            continue

        sanitized_key = sv
        if len(parent_tokens) == 1:
            paper_key = f"Q^{{{outcome_letters}|{parent_tokens[0]}}}"
        else:
            paper_key = f"Q^{{{outcome_letters}|{','.join(parent_tokens)}}}"

        if paper_key in data or sanitized_key in data:
            continue
        if paper_key in new_entries or sanitized_key in new_entries:
            continue

        out_pair = _find_subunit_matrix_for_letter(data, outcome_letters)
        if out_pair is None:
            continue
        outcome_key, outcome_arr = out_pair

        cond_mats: list[np.ndarray] = []
        bad = False
        for pt in parent_tokens:
            pr = _find_subunit_matrix_for_letter(data, pt)
            if pr is None:
                bad = True
                break
            cond_mats.append(pr[1])
        if bad or not cond_mats:
            continue
        shape0 = outcome_arr.shape
        if any(np.asarray(c).shape != shape0 for c in cond_mats):
            continue

        n_units = outcome_arr.shape[0]
        family_outcome = families.get(outcome_key, "bernoulli")
        local_estimator_kwargs = _resolve_estimator_kwargs(
            estimator_kwargs,
            variable_name=outcome_key,
            family=family_outcome,
        )

        is_sum_var = sv in formula_sum_vars or paper_key in formula_sum_vars

        if len(parent_tokens) > 1 and disable_multiparent_fix:
            continue

        if len(parent_tokens) > 1 and is_sum_var:
            # A full Σ-profile (one column per distinct parent tuple) is not implemented for
            # multi-parent Q.  Fall back to the scalar multi-parent summary so ``enriched``
            # contains the symbol and factors are not replaced by 1 (approximate vs. theory).
            is_sum_var = False

        if len(parent_tokens) == 1:
            cond_arr = cond_mats[0]
            if is_sum_var:
                eval_vals = np.unique(cond_arr.ravel())
            else:
                eval_vals = np.array([iv_val])

            if estimator_backend == "torch" and family_outcome in {
                "bernoulli",
                "poisson",
                "gaussian",
                "normal",
                "beta",
                "gamma",
            }:
                per_unit_arr = torch_conditional_expectations_per_unit(
                    y=outcome_arr,
                    x=cond_arr,
                    eval_values=eval_vals,
                    family=family_outcome,
                    device=resolve_torch_device_from_mapping(torch_kwargs),
                    devices=torch_kwargs.get("devices"),
                    ridge=float(torch_kwargs.get("ridge", 1e-4)),
                    max_iter=int(torch_kwargs.get("max_iter", 200)),
                    lr=float(torch_kwargs.get("lr", 5e-2)),
                    weight_decay=float(torch_kwargs.get("weight_decay", 1e-4)),
                )
            else:
                unit_tasks = [
                    (
                        outcome_arr[i],
                        cond_arr[i],
                        eval_vals,
                        family_outcome,
                        estimator_backend,
                        torch_kwargs,
                        local_estimator_kwargs,
                    )
                    for i in range(n_units)
                ]
                per_unit_rows = parallel_map(
                    unit_tasks,
                    _fit_conditional_q_row_task,
                    n_jobs=local_n_jobs,
                    backend=local_parallel_backend,
                )
                per_unit_arr = (
                    np.vstack(per_unit_rows) if per_unit_rows else np.zeros((0, len(eval_vals)))
                )

            result_arr = per_unit_arr[:, 0] if len(eval_vals) == 1 else per_unit_arr
        else:
            X_stack = np.stack(cond_mats, axis=-1)
            unit_tasks_mp = [
                (
                    outcome_arr[i],
                    X_stack[i],
                    family_outcome,
                    estimator_backend,
                    torch_kwargs,
                    local_estimator_kwargs,
                )
                for i in range(n_units)
            ]
            per_unit_rows_mp = parallel_map(
                unit_tasks_mp,
                _fit_conditional_q_multiparent_row_task,
                n_jobs=local_n_jobs,
                backend=local_parallel_backend,
            )
            per_unit_arr_mp = (
                np.vstack(per_unit_rows_mp) if per_unit_rows_mp else np.zeros((0, 1))
            )
            result_arr = per_unit_arr_mp[:, 0]

        new_entries[paper_key] = result_arr
        new_entries[sanitized_key] = result_arr

    return new_entries


def _build_unit_context(
    enriched: Dict[str, np.ndarray],
    all_outcome_vars: set[str],
    context: Dict[str, float],
    unit_index: int,
) -> Dict[str, float]:
    """Build one unit-specific evaluation context."""
    unit_ctx: Dict[str, float] = {}
    for key_d, arr_d in enriched.items():
        if key_d in all_outcome_vars or key_d in context:
            continue
        arr_np = np.asarray(arr_d, dtype=float)
        if arr_np.ndim == 1 and unit_index < len(arr_np):
            unit_ctx[key_d] = float(arr_np[unit_index])
        elif arr_np.ndim == 2 and unit_index < arr_np.shape[0]:
            for j, pval in enumerate(arr_np[unit_index]):
                unit_ctx[f"{key_d}__{j}"] = float(pval)
            # Align with ast_to_estimator conditioning design: 2D Q-profiles use the last column.
            unit_ctx[key_d] = float(arr_np[unit_index, -1])
    unit_ctx.update(context)
    return unit_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────


@overload
def ast_to_estimator(
    ast: Any,
    data: Dict[str, np.ndarray],
    intervention_value: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
    n_mc_samples: int = 1000,
    random_seed: Optional[int] = 0,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    *,
    return_artifacts: Literal[False] = False,
) -> float: ...


@overload
def ast_to_estimator(
    ast: Any,
    data: Dict[str, np.ndarray],
    intervention_value: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
    n_mc_samples: int = 1000,
    random_seed: Optional[int] = 0,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    *,
    return_artifacts: Literal[True],
) -> Tuple[float, Dict[str, Any]]: ...


def ast_to_estimator(
    ast: Any,
    data: Dict[str, np.ndarray],
    intervention_value: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
    n_mc_samples: int = 1000,
    random_seed: Optional[int] = 0,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    *,
    return_artifacts: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Numerically evaluate a causal estimand from a pyAgrum identification formula.

    Parameters
    ----------
    ast : pyAgrum causal ASTtree
        Symbolic identification formula returned by
        ``csl.identifyingIntervention()`` or ``DoCalculusResult.ast``.
    data : dict[str, np.ndarray]
        Observed data.  Keys are **paper-notation** variable names
        (e.g. ``"Q^{y|a}"``, ``"Q^a"``, ``"U"``, ``"Y"``).
        Values are ``np.ndarray``:

        * **Unit-level** variables (confounders, unit outcomes): shape ``(n_units,)``.
        * **Subunit-level raw observations**: shape ``(n_units, n_subunits)``.

        For 2-D arrays, per-unit means (Q-variables) are added automatically.
        You may also pre-compute Q-variables and include them directly.
    intervention_value : float or dict[str, float]
        The value ``x*`` for ``do(X = x*)``.  Pass a scalar ``float`` for a
        single intervention variable; pass ``{var_name: value}`` for multiple.
    distribution_families : dict[str, str], optional
        Maps **paper-notation variable names** to distribution families.
        Supported values:

        ``"gaussian"``
            Normal distribution.  Estimates (μ, σ²) via MLE.
            Conditional version uses linear regression for μ.

        ``"bernoulli"``
            Bernoulli / binary.  Unconditional: sample proportion.
            Conditional: logistic regression.

        ``"beta"``
            Beta distribution on (0,1). Parameters (α,β) via MOM.

        ``"half_cauchy"``
            Half-Cauchy distribution for positive scale/variance variables.
            Used in HCM paper simulations as prior on variance τ.

        ``"poisson"``, ``"laplace"``, ``"student_t"``, ``"exponential"``,
        ``"gamma"``, ``"lognormal"``, ``"weibull"``, ``"inverse_gaussian"``
            Other parametric families; see ``ConditionalDensityEstimator``
            docstring for details.

        ``"nonparametric"`` *(default)*
            Discrete Y → conditional frequency table with k-NN.
            Continuous Y → Gaussian KDE.

    n_mc_samples : int, default 1000
        Monte Carlo samples for continuous marginalisation in the formula
        evaluator.  For large values, ``estimator_backend="torch"`` with
        ``torch_kwargs["device"]="cuda"`` runs the batched MC tensor path on GPU.
    random_seed : int or None, default 0
        Seed for the Monte Carlo sampler (``None`` → non-deterministic).
    n_jobs : int, default 1
        Number of workers for repeated independent per-unit fits/evaluations.
    parallel_backend : {"threads", "processes"}, default "threads"
        Backend used when ``n_jobs`` requests parallel work.
    estimator_backend : {"numpy", "torch", "numpyro"}, default "numpy"
        Backend for supported estimators inside the HCM estimation path.
    torch_kwargs : dict[str, Any], optional
        Torch backend options such as ``device``, ``devices``, ``ridge``,
        ``max_iter``, ``lr``, and ``weight_decay``.
    estimator_kwargs : dict[str, Any], optional
        Estimator-specific hyperparameters. Supports either a flat dict or a
        nested mapping with keys like ``"__default__"``, family names
        (for example ``"gaussian_mixture"``), or variable names
        (for example ``"Q^{y|a}"``).

    Returns
    -------
    float
        Estimated value of the causal estimand E[Y | do(X = x*)].

    Examples
    --------
    >>> # Confounder model: binary A, Y, unobserved U
    >>> # data["A"] shape (30, 10): 30 units, 10 subunits each
    >>> # data["Y"] shape (30, 10)
    >>> from estimation import ast_to_estimator
    >>> ate = ast_to_estimator(
    ...     ast=result.ast,
    ...     data={"A": A_matrix, "Y": Y_matrix},
    ...     intervention_value=1.0,
    ...     distribution_families={"Q^{y|a}": "bernoulli"},
    ... )
    >>> print(f"ATE (do A=1): {ate:.4f}")

    Notes
    -----
    **HCM-specific Q-variable handling.**  In the paper, Q-variables are
    unit-level random distributions over subunit variables.  For binary
    subunit variable *v*, ``Q^v_i`` is the within-unit proportion
    (Bernoulli parameter).  This function estimates such Q-variables as
    within-unit means when raw 2-D subunit data is provided.

    **Formula evaluation.**  The function walks the pyAgrum AST (or parses
    ``.toLatex()`` as fallback) to discover which conditional densities appear,
    fits estimators from the observed unit-level Q-variable arrays, and
    evaluates the formula term by term using Monte Carlo marginalisation for
    continuous summation variables.  Conditional Q-variables (e.g. Q^{y|a})
    needed by the formula are precomputed from raw subunit data automatically.
    """
    families = distribution_families or {}
    rng = np.random.default_rng(random_seed)
    local_n_jobs = n_jobs
    if estimator_backend.lower().strip() == "numpyro" and n_jobs not in (None, 1) and parallel_backend == "threads":
        parallel_backend = "processes"
    if estimator_backend.lower().strip() == "numpyro":
        local_n_jobs = 1

    # ── 1. Enrich data: add Q-variables computed from subunit data ────────────
    # SubunitParamEstimator is used per variable according to the families dict.
    enriched = compute_q_from_subunit_data(
        data,
        families=families,
        n_jobs=local_n_jobs,
        parallel_backend=parallel_backend,
        estimator_backend=estimator_backend,
        torch_kwargs=torch_kwargs,
        estimator_kwargs=estimator_kwargs,
    )

    # ── 2. Resolve intervention map ───────────────────────────────────────────
    if isinstance(intervention_value, dict):
        iv_map: Dict[str, float] = dict(intervention_value)
    else:
        iv_map = {"__single__": float(intervention_value)}

    # Determine a scalar intervention value for Q-variable precomputation
    _iv_scalar = float(iv_map.get("__single__") or next(iter(iv_map.values()), 0.0))

    # ── 3. Parse formula and precompute conditional Q-variables ───────────────
    formula = _extract_formula(ast)
    q_cond_entries = _precompute_conditional_q_vars(
        data,
        formula,
        families,
        _iv_scalar,
        n_jobs=local_n_jobs,
        parallel_backend=parallel_backend,
        estimator_backend=estimator_backend,
        torch_kwargs=torch_kwargs,
        estimator_kwargs=estimator_kwargs,
    )
    for k, v in q_cond_entries.items():
        if k not in enriched:
            enriched[k] = v

    # ── 4. Build name resolution maps ─────────────────────────────────────────
    san_to_paper, paper_to_san = _build_name_maps(list(enriched.keys()))

    def resolve(name: str) -> Optional[str]:
        if name in enriched:
            return name
        candidate = san_to_paper.get(name)
        if candidate is not None and candidate in enriched:
            return candidate
        return None

    # ── 5. Collect unique conditional terms ───────────────────────────────────
    # formula was already parsed in step 3 above.
    cond_terms = formula.collect_conditionals()
    seen: set = set()
    unique_terms: List[Dict] = []
    for t in cond_terms:
        if t["key"] not in seen:
            seen.add(t["key"])
            unique_terms.append(t)

    # ── 7. Fit density estimators ─────────────────────────────────────────────
    # For each conditional term P(Y | X₁, …, Xₖ) in the formula:
    #   • If Y is a Q-variable with multi-parameter representation (Gaussian,
    #     Beta, …), the per-unit params form a 2-D array → use QDensityEstimator
    #     (KDE in the parameter space).
    #   • Otherwise, treat Y as a unit-level scalar → ConditionalDensityEstimator.
    #
    # The distinction maps directly onto the paper:
    #   - P(Q^v | X) for subunit distribution variables → QDensityEstimator
    #   - P(Y_unit | X) for unit-level outcome/covariate → ConditionalDensityEstimator
    fitted: Dict[tuple, Any] = {}

    for term in unique_terms:
        out_vars = term["outcome_vars"]
        cond_vars = term["cond_vars"]
        key = term["key"]

        # Determine family for the outcome variable
        family = "nonparametric"
        outcome_variable_name = None
        for candidate in out_vars:
            paper_c = resolve(candidate) or candidate
            outcome_variable_name = paper_c
            f = families.get(paper_c)
            if f is None:
                f = families.get(candidate)
            if isinstance(f, str) and len(f) > 0:
                family = f
                break

        # Outcome data
        out_paper = resolve(out_vars[0]) if out_vars else None
        if out_paper is None:
            warnings.warn(f"No data for outcome {out_vars}; skipping term.")
            continue
        Y_data = np.asarray(enriched[out_paper], dtype=float)

        # Determine reference length for conditioning data alignment
        n_ref = Y_data.shape[0]

        # Conditioning data: flatten multi-dim Q-params to columns
        X_cols: List[np.ndarray] = []
        cond_keys_for_eval: List[tuple[str, str]] = []
        for cv in cond_vars:
            cv_paper = resolve(cv)
            if cv_paper is not None and cv_paper in enriched:
                col_data = np.asarray(enriched[cv_paper], dtype=float)
                if col_data.shape[0] != n_ref:
                    continue
                if col_data.ndim == 1:
                    X_cols.append(col_data)
                else:
                    # 2D conditional profile (e.g. Q^{a|z} at multiple z values).
                    # Use only the last column (highest conditioning value = z_max),
                    # which is the strongest IV proxy for the unobserved confounder U.
                    # Using all columns simultaneously causes collinearity: for binary Z,
                    # Q^{a|z=0} and Q^{a|z=1} differ by a near-constant gap, so the
                    # regression cannot distinguish them.
                    X_cols.append(col_data[:, -1])
                cond_keys_for_eval.append((cv_paper, str(cv)))
        X_arr = np.column_stack(X_cols) if X_cols else None

        if Y_data.ndim == 2:
            # Multi-parameter Q-variable (e.g. Gaussian (μ, σ²) per unit).
            # Estimate P(Q = q | X = x) via KDE in the parameter space.
            est = QDensityEstimator()
            est.fit(Y_data, x_cond=X_arr)
        else:
            # Unit-level scalar variable or scalar Q-variable.
            Y_arr = Y_data.ravel()
            if X_arr is not None and X_arr.shape[0] != len(Y_arr):
                X_arr = None
            est = ConditionalDensityEstimator(
                family=family,
                backend=estimator_backend,
                torch_kwargs=torch_kwargs,
                estimator_kwargs=_resolve_estimator_kwargs(
                    estimator_kwargs,
                    variable_name=outcome_variable_name,
                    family=family,
                ),
            )
            est.fit(Y_arr, X_arr)

        if cond_keys_for_eval:
            est._hcm_cond_parent_keys = tuple(cond_keys_for_eval)  # type: ignore[attr-defined]

        fitted[key] = est

    # ── 8. Build evaluation context from intervention ─────────────────────────
    context: Dict[str, float] = {}
    for k, v in iv_map.items():
        if k != "__single__":
            paper_k = resolve(k) or k
            context[paper_k] = v
            context[k] = v

    # ── 9. Identify outcome variables (should NOT be pre-set in context) ──────
    # Conditioning and summed variables can be set from data; outcome vars must
    # remain absent from context so the estimator returns E[Y|X] not P(Y=y|X).
    all_outcome_vars: set = set()
    for t in unique_terms:
        for v in t["outcome_vars"]:
            pv = resolve(v) or v
            all_outcome_vars.add(pv)
            all_outcome_vars.add(v)

    # ── 10. Evaluate: per-unit average if unit-level data present ─────────────
    n_units = _infer_n_units(enriched)
    if n_units > 1:
        if n_jobs is None or n_jobs == 1:
            unit_vals = []
            for i in range(n_units):
                unit_ctx = _build_unit_context(
                    enriched,
                    all_outcome_vars,
                    context,
                    i,
                )
                unit_vals.append(
                    _eval_formula(
                        formula,
                        unit_ctx,
                        fitted,
                        enriched,
                        resolve,
                        n_mc_samples,
                        rng,
                        n_units,
                    )
                )
        else:
            if random_seed is None:
                child_sequences = [np.random.SeedSequence() for _ in range(n_units)]
            else:
                child_sequences = np.random.SeedSequence(random_seed).spawn(n_units)

            def _eval_one_unit(i: int) -> float:
                unit_ctx = _build_unit_context(
                    enriched,
                    all_outcome_vars,
                    context,
                    i,
                )
                unit_rng = np.random.default_rng(child_sequences[i])
                return float(
                    _eval_formula(
                        formula,
                        unit_ctx,
                        fitted,
                        enriched,
                        resolve,
                        n_mc_samples,
                        unit_rng,
                        n_units,
                    )
                )

            unit_vals = parallel_map(
                range(n_units),
                _eval_one_unit,
                n_jobs=local_n_jobs,
                backend=parallel_backend,
            )
        val = float(np.mean(unit_vals))
    else:
        val = float(
            _eval_formula(
                formula,
                context,
                fitted,
                enriched,
                resolve,
                n_mc_samples,
                rng,
                n_units,
            )
        )

    if return_artifacts:
        artifacts: Dict[str, Any] = {
            "enriched": enriched,
            "fitted": fitted,
            "formula": formula,
            "distribution_families": dict(families),
            "intervention_map": iv_map,
            "intervention_iv_scalar": float(_iv_scalar),
            "n_mc_samples": int(n_mc_samples),
            "random_seed": random_seed,
            "estimator_backend": estimator_backend,
            "n_jobs": local_n_jobs,
            "parallel_backend": parallel_backend,
            "torch_kwargs": dict(torch_kwargs or {}),
            "estimator_kwargs": dict(estimator_kwargs or {}),
            "unique_terms": unique_terms,
        }
        return val, artifacts
    return val


def _infer_n_units(data: Dict[str, np.ndarray]) -> int:
    """we infer the unit count from the largest leading axis among 1D/2D arrays."""
    nmax = 1
    for arr in data.values():
        a = np.asarray(arr)
        if a.ndim == 1:
            nmax = max(nmax, len(a))
        elif a.ndim == 2:
            nmax = max(nmax, a.shape[0])
    return nmax if nmax > 1 else 1


# ─────────────────────────────────────────────────────────────────────────────
# Primary public API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_causal_effect(
    result: Any,
    data: Dict[str, np.ndarray],
    intervention: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
    n_mc_samples: int = 1000,
    random_seed: Optional[int] = 0,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    *,
    return_artifacts: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Estimate ``E[Y | do(X = x*)]`` from an identified causal formula and data.

    This is the **primary entry point** for numerical evaluation of HCM causal
    estimands.  The caller is expected to have already produced a
    ``DoCalculusResult`` via the pipeline in ``do_calculus.py``::

        hscm        = HSCMParametric(...)
        cgm         = collapse(hscm)
        q_hat, pars = suggest_augment_for_outcome(hscm, "Y")
        aug_cgm     = augment_collapsed_model(cgm, q_hat, pars)
        mar_cgm     = marginalize_augmented_model(aug_cgm, ...)
        result      = identify_effect(mar_cgm, Y={"Q^y"}, X={"Q^a"})
        ate         = estimate_causal_effect(result, data, {"Q^a": 1.0})

    Estimation strategy
    -------------------
    1. **Enrich** raw data with Q-variable summaries via
       :class:`SubunitParamEstimator`:

       - Scalar families (Bernoulli, Poisson): per-unit parameter as
         ``(n_units,)``.
       - Multi-parameter families (Gaussian → (μ, σ²); Beta → (α, β); …):
         per-unit parameter vectors as ``(n_units, n_params)``.

    2. **Formula-driven evaluation** (all HCM model types): the identified
       ASTree formula is parsed; conditional Q-variables (e.g. Q^{y|a})
       required by the formula are precomputed from raw subunit data; each
       conditional term ``P(Y | X₁, …, Xₖ)`` is estimated:

       - Unit-level scalars or scalar Q-variables → :class:`ConditionalDensityEstimator`.
       - Distribution-valued Q-variables (multi-param) → :class:`QDensityEstimator`
         (KDE in the parameter space; residual-KDE for conditional terms).

    Parameters
    ----------
    result : ``DoCalculusResult`` or pyAgrum ASTtree
        Identification result from ``identify_effect()``.  Must have
        ``identifiable=True``; raises ``ValueError`` otherwise.
    data : dict[str, np.ndarray]
        Observed data.  Keys are paper-notation variable names (``"Q^a"``,
        ``"Q^{y|a}"``, ``"Y"``, ``"A"``, …).

        - 2-D arrays ``(n_units, n_subunits)``: raw subunit observations.
          Q-variable summaries are added automatically using
          :class:`SubunitParamEstimator` with the specified families.
        - 1-D arrays ``(n_units,)``: unit-level observations (pre-computed
          Q-parameters, unit outcomes, covariates, …).

    intervention : float or dict[str, float]
        The do-intervention.  Scalar float for a single unnamed variable;
        ``{var_name: value}`` to name the intervention variable explicitly
        (e.g., ``{"Q^a": 1.0}``).
    distribution_families : dict[str, str], optional
        Maps paper-notation variable names to distribution families.

        Two roles:

        1. Passed to :class:`SubunitParamEstimator` to determine how 2-D
           subunit arrays are summarised into Q-variable parameter vectors.
           **Parametric** (``"gaussian"``, ``"beta"``, …): MLE/MoM params.
           **Nonparametric** (``"nonparametric"``): 4 moment statistics.

        2. Passed to :class:`ConditionalDensityEstimator` for unit-level
           ``P(Y | X)`` terms.

        Supported families: ``"bernoulli"``, ``"gaussian"``, ``"beta"``,
        ``"gamma"``, ``"poisson"``, ``"exponential"``, ``"lognormal"``,
        ``"weibull"``, ``"laplace"``, ``"student_t"``, ``"inverse_gaussian"``,
        ``"half_cauchy"``, ``"categorical"`` (aliases ``multinomial``, ``nominal``),
        ``"nonparametric"`` (default).

    n_mc_samples : int, default 1000
        Monte Carlo samples for continuous marginalisation in the formula
        evaluator.  For large values, ``estimator_backend="torch"`` together
        with ``torch_kwargs["device"]="cuda"`` keeps the batched conditional
        expectations on the GPU instead of many small NumPy calls on CPU.
    random_seed : int or None, default 0
    n_jobs : int, default 1
        Number of workers for repeated independent per-unit fits/evaluations.
    parallel_backend : {"threads", "processes"}, default "threads"
        Backend used when ``n_jobs`` requests parallel work.
    estimator_backend : {"numpy", "torch", "numpyro"}, default "numpy"
        Backend for supported estimators inside the HCM estimation path.
    torch_kwargs : dict[str, Any], optional
        Torch backend options such as ``device``, ``devices``, ``ridge``,
        ``max_iter``, ``lr``, and ``weight_decay``.
    estimator_kwargs : dict[str, Any], optional
        Estimator-specific hyperparameters, including ``n_components`` for
        ``gaussian_mixture`` and variational options for the ``numpyro``
        backend such as ``num_steps``, ``learning_rate``,
        ``num_posterior_samples``, ``seed``, and JAX ``device``.

    Returns
    -------
    float or tuple
        Estimated causal effect ``E[Y | do(X = x*)]``.  If ``return_artifacts=True``,
        returns ``(float, dict)`` with fitted conditional / Q-density estimators and
        the enriched data dict (suitable for ``pickle``); conditional Q-precomputation
        depends on the intervention value, so call separately per ``do`` level.

    Raises
    ------
    ValueError
        If ``result`` has ``identifiable=False`` or carries no formula.

    See Also
    --------
    SubunitParamEstimator : Per-unit distribution parameter estimation.
    QDensityEstimator     : Density over Q-variable parameter space.
    """
    # ── Unpack DoCalculusResult ────────────────────────────────────────────────
    if hasattr(result, "identifiable"):
        if not result.identifiable:
            raise ValueError(
                "Effect is not identifiable: {}".format(
                    getattr(result, "explanation", None)
                    or getattr(result, "error", "unknown reason")
                )
            )
        ast = getattr(result, "ast", None)
        formula_latex = getattr(result, "formula_latex", None)
        if formula_latex:
            # Always prefer latex parsing: it reliably reconstructs the full
            # formula including sum wrappers.  Direct pyAgrum AST introspection
            # can inadvertently strip the outer ASTSum (returning only its body),
            # which causes the MC marginalisation loop to be skipped entirely.
            class _LatexAST:
                def toLatex(self): return formula_latex
            ast = _LatexAST()
        elif ast is None:
            raise ValueError(
                "DoCalculusResult carries no AST and no formula_latex.  "
                "Re-run identify_effect() with pyagrum installed."
            )
    else:
        ast = result  # raw pyAgrum ASTtree

    # ── Formula evaluation (single path for all HCM model types) ──────────────
    return ast_to_estimator(
        ast=ast,
        data=data,
        intervention_value=intervention,
        distribution_families=distribution_families,
        n_mc_samples=n_mc_samples,
        random_seed=random_seed,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        estimator_backend=estimator_backend,
        torch_kwargs=torch_kwargs,
        estimator_kwargs=estimator_kwargs,
        return_artifacts=return_artifacts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────────────


