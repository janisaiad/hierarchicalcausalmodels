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

import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

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
    # Non-parametric fallback
    "nonparametric",
}

_FAMILY_ALIASES: Dict[str, str] = {
    "normal": "gaussian",
    "t": "student_t",
    "log_normal": "lognormal",
    "wald": "inverse_gaussian",
}


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

    **Non-parametric**

    ``"nonparametric"``
        Discrete Y → conditional frequency table with k-NN.
        Continuous Y → Gaussian KDE.
    """

    def __init__(self, family: str = "nonparametric"):
        family = family.lower().strip()
        self.family = _FAMILY_ALIASES.get(family, family)
        self._fitted = False

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
            "laplace":          self._fit_laplace,
            "student_t":        self._fit_student_t,
            "exponential":      self._fit_exponential,
            "gamma":            self._fit_gamma,
            "lognormal":        self._fit_lognormal,
            "weibull":          self._fit_weibull,
            "inverse_gaussian": self._fit_inverse_gaussian,
            "beta":             self._fit_beta,
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
        elif SKLEARN_AVAILABLE and is_binary and len(np.unique(Y)) == 2:
            lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e4)
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
        if self._lr_gauss is not None:
            return float(self._lr_gauss.predict(np.atleast_2d(x_query))[0])
        return self._mu_marginal

    # -------- Poisson ---------------------------------------------------------

    def _fit_poisson(self, Y, X):
        self._lambda_marginal = float(np.maximum(np.mean(Y), 1e-9))
        self._lr_poisson = None
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
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
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, Y_pos)
            self._lr_gamma = lr

    def _predict_mean_gamma(self, x_query) -> float:
        if self._lr_gamma is None:
            return self._gamma_mean
        return float(np.maximum(self._lr_gamma.predict(np.atleast_2d(x_query))[0], 1e-9))

    def _eval_gamma(self, x_query, y_query) -> float:
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
        if X is not None and X.shape[1] > 0 and SKLEARN_AVAILABLE:
            # Logit-linear regression for mean
            from sklearn.linear_model import LogisticRegression as _LR
            # Use linear regression on logit(Y) as proxy
            lr = LinearRegression()
            lr.fit(X, np.log(Y_clipped / (1 - Y_clipped)))  # logit
            self._lr_beta = lr

    def _predict_mu_beta(self, x_query) -> float:
        if self._lr_beta is None:
            return self._beta_mu
        logit_pred = float(self._lr_beta.predict(np.atleast_2d(x_query))[0])
        return float(1.0 / (1.0 + np.exp(-logit_pred)))

    def _eval_beta(self, x_query, y_query) -> float:
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
        x_arr = np.atleast_1d(np.asarray(x_query, dtype=float))
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
        x_arr = np.atleast_1d(np.asarray(x_query, dtype=float))
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
            "laplace":          self._eval_laplace,
            "student_t":        self._eval_student_t,
            "exponential":      self._eval_exponential,
            "gamma":            self._eval_gamma,
            "lognormal":        self._eval_lognormal,
            "weibull":          self._eval_weibull,
            "inverse_gaussian": self._eval_inverse_gaussian,
            "beta":             self._eval_beta,
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
                "laplace":          lambda: self._laplace_loc,
                "student_t":        lambda: self._t_loc,
                "exponential":      lambda: 1.0 / (self._exp_lambda + 1e-9),
                "gamma":            lambda: self._gamma_mean,
                "lognormal":        lambda: float(np.exp(self._logn_mu + 0.5 * self._logn_sigma ** 2)),
                "weibull":          lambda: self._weibull_mean,
                "inverse_gaussian": lambda: self._ig_mu,
                "beta":             lambda: self._beta_mu,
            }
            fn_marg = _marginal.get(self.family)
            if fn_marg is not None:
                return float(fn_marg())
            return self._expect_nonparametric(None)
        _expect_dispatch = {
            "bernoulli":        self._expect_bernoulli,
            "poisson":          self._predict_lambda_poisson,
            "gaussian":         self._predict_mu_gaussian,
            "laplace":          self._predict_loc_laplace,
            "student_t":        self._predict_loc_t,
            "exponential":      self._predict_mean_exp,
            "gamma":            self._predict_mean_gamma,
            "lognormal":        lambda x: float(np.exp(self._predict_mu_lognormal(x) + 0.5 * self._logn_sigma ** 2)),
            "weibull":          lambda x: self._predict_scale_weibull(x) * __import__("math").gamma(1 + 1 / self._weibull_c),
            "inverse_gaussian": self._predict_mu_ig,
            "beta":             self._predict_mu_beta,
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
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit Q-variable estimator (paper Appendix D)
# ─────────────────────────────────────────────────────────────────────────────

class PerUnitQEstimator:
    """
    Estimate a Q-variable μ^{y|a}_i for each unit i from within-unit
    subunit observations (paper Appendix D.1/D.2).

    For each unit i with n_i subunit observations {(A_ij, Y_ij)}_j:
      - ``"bernoulli"``  : μ^{y|a}_i(a*) = (#subunits j with Y_ij=1, A_ij=a*) / (#subunits j with A_ij=a*)
      - ``"gaussian"``   : per-unit OLS regression; μ^{y|a}_i(a*) = predicted mean at a*
      - ``"nonparametric"``: per-unit k-NN regressor

    The ATE is then (1/n) Σ_i μ^{y|a}_i(a*).
    """

    def __init__(self, family: str = "bernoulli"):
        self.family = family.lower()
        self._unit_estimators: List[ConditionalDensityEstimator] = []
        self._fitted = False

    def fit(self, Y_subunit: np.ndarray, A_subunit: np.ndarray) -> "PerUnitQEstimator":
        """
        Fit per-unit estimators.

        Parameters
        ----------
        Y_subunit : array of shape (n_units, n_subunits)
        A_subunit : array of shape (n_units, n_subunits)
        """
        Y_subunit = np.asarray(Y_subunit, dtype=float)
        A_subunit = np.asarray(A_subunit, dtype=float)
        assert Y_subunit.shape == A_subunit.shape, "Y and A must have the same shape."
        n_units = Y_subunit.shape[0]
        self._unit_estimators = []
        for i in range(n_units):
            Y_i = Y_subunit[i]
            A_i = A_subunit[i]
            est = ConditionalDensityEstimator(family=self.family)
            est.fit(Y_i, A_i.reshape(-1, 1))
            self._unit_estimators.append(est)
        self._fitted = True
        return self

    def ate(self, intervention_value: float) -> float:
        """
        ATE = (1/n) Σ_i E[Y_ij | A_ij = intervention_value, unit i].
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        x_query = np.array([[intervention_value]])
        unit_means = [est.expectation(x_query) for est in self._unit_estimators]
        return float(np.mean(unit_means))

    def per_unit_means(self, intervention_value: float) -> np.ndarray:
        """Return per-unit E[Y | A = a*] as array of length n_units."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        x_query = np.array([[intervention_value]])
        return np.array([est.expectation(x_query) for est in self._unit_estimators])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Q-variable computation & name resolution
# ─────────────────────────────────────────────────────────────────────────────

def compute_q_from_subunit_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Augment *data* with Q-variable arrays computed from raw subunit observations.

    For each key ``v`` whose value is a 2-D array of shape ``(n_units, n_subunits)``,
    adds ``"Q^v"`` (if not already present) as the within-unit mean over subunits.

    This gives a simple scalar proxy for the unit-level distribution — appropriate
    for binary/continuous subunit variables when only a mean summary is needed.
    """
    result = dict(data)
    for key, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim == 2:
            q_key = f"Q^{key.lower()}" if not key.startswith("Q") else key
            if q_key not in result:
                result[q_key] = arr.mean(axis=1)
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
        sum_vars = list(raw_vars)
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

def _eval_formula(
    node: _ASTFormula,
    context: Dict[str, float],
    fitted: Dict[tuple, ConditionalDensityEstimator],
    data: Dict[str, np.ndarray],
    resolve: Callable[[str], Optional[str]],
    n_mc: int,
    rng: np.random.Generator,
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

        # Collect conditioning values
        x_vals = []
        for cv in node.cond_vars:
            cv_paper = resolve(cv) or cv
            val = context.get(cv_paper, context.get(cv))
            if val is None:
                # Use marginal mean as fallback
                d = data.get(cv_paper) or data.get(cv)
                val = float(np.mean(d)) if d is not None else 0.0
            x_vals.append(float(val))
        x_q = np.array(x_vals) if x_vals else None

        # Check whether to return P(Y=y|X) or E[Y|X]
        out_paper = resolve(node.outcome_vars[0]) if node.outcome_vars else None
        y_val = context.get(out_paper) if out_paper else None
        if y_val is None and node.outcome_vars:
            y_val = context.get(node.outcome_vars[0])

        if y_val is not None:
            return est.prob(float(y_val), x_q)
        else:
            return est.expectation(x_q, n_mc=n_mc)

    if isinstance(node, _ASTProduct):
        result = 1.0
        for child in node.children:
            result *= _eval_formula(child, context, fitted, data, resolve, n_mc, rng)
        return result

    if isinstance(node, _ASTSum):
        total = 0.0
        # If no explicit summation variables, just evaluate body
        if not node.sum_vars:
            return _eval_formula(node.formula, context, fitted, data, resolve, n_mc, rng)
        # Evaluate by marginalising over each summation variable
        for sv in node.sum_vars:
            sv_paper = resolve(sv) or sv
            sv_data_arr = data.get(sv_paper) if data.get(sv_paper) is not None else data.get(sv)
            if sv_data_arr is None:
                # No data → evaluate without marginalisation
                total += _eval_formula(node.formula, context, fitted, data, resolve, n_mc, rng)
                continue
            sv_arr = np.asarray(sv_data_arr, dtype=float).ravel()
            unique_vals = np.unique(sv_arr)
            is_discrete = np.all(sv_arr == sv_arr.astype(int)) and len(unique_vals) <= 30

            if is_discrete:
                _, counts = np.unique(sv_arr, return_counts=True)
                probs = counts / counts.sum()
                for val, p in zip(unique_vals, probs):
                    new_ctx = {**context, sv_paper: float(val), sv: float(val)}
                    total += p * _eval_formula(node.formula, new_ctx, fitted, data, resolve, n_mc, rng)
            else:
                # Monte Carlo: sample u_k ~ empirical p(sv), compute mean of body(u_k).
                # This correctly estimates ∫ body(u) p(u) du  =  E_{u~p}[body(u)].
                # We must NOT multiply by p(u_k) again inside the body; pass sv_paper as the
                # "marginal being cancelled" so _eval_formula_vec skips the P(sv) factor.
                mc_idx = rng.integers(0, len(sv_arr), size=n_mc)
                mc_samples = sv_arr[mc_idx]
                batch_result = _eval_formula_vec(node.formula, context, sv_paper, mc_samples,
                                                 fitted, data, resolve, cancel_marginal=sv_paper)
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
        pred = _linreg_predict("_lr_gauss")
        return pred if pred is not None else np.full(n, est._mu_marginal)

    if fam == "bernoulli":
        if est._lr_model is None:
            return np.full(n, est._p_marginal)
        if hasattr(est._lr_model, "predict_proba"):
            return est._lr_model.predict_proba(X_batch)[:, 1]
        return np.clip(est._lr_model.predict(X_batch), 0.0, 1.0)

    if fam == "poisson":
        lr = getattr(est, "_lr_poisson", None)
        if lr is None:
            return np.full(n, est._lambda_marginal)
        if isinstance(lr, tuple):  # log-linear fallback
            return np.exp(np.clip(lr[1].predict(X_batch), -10, 10))
        return np.maximum(lr.predict(X_batch), 1e-9)

    if fam == "laplace":
        pred = _linreg_predict("_lr_laplace")
        return pred if pred is not None else np.full(n, est._laplace_loc)

    if fam == "student_t":
        pred = _linreg_predict("_lr_t")
        return pred if pred is not None else np.full(n, est._t_loc)

    if fam == "exponential":
        pred = _linreg_predict("_lr_exp")
        return np.maximum(pred, 1e-9) if pred is not None else np.full(n, 1.0 / (est._exp_lambda + 1e-9))

    if fam == "gamma":
        pred = _linreg_predict("_lr_gamma")
        return np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._gamma_mean)

    if fam == "lognormal":
        pred = _linreg_predict("_lr_lognormal")
        log_mu = pred if pred is not None else np.full(n, est._logn_mu)
        return np.exp(log_mu + 0.5 * est._logn_sigma ** 2)

    if fam == "weibull":
        import math
        pred = _linreg_predict("_lr_weibull")
        mu = np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._weibull_mean)
        try:
            gf = math.gamma(1.0 + 1.0 / est._weibull_c)
        except Exception:
            gf = 1.0
        return mu  # scale * gamma_factor = mu by construction

    if fam == "inverse_gaussian":
        pred = _linreg_predict("_lr_ig")
        return np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._ig_mu)

    if fam == "beta":
        lr = getattr(est, "_lr_beta", None)
        if lr is None:
            return np.full(n, est._beta_mu)
        logit_pred = lr.predict(X_batch)
        return 1.0 / (1.0 + np.exp(-logit_pred))

    # Non-parametric: per-sample loop
    return np.array([est.expectation(X_batch[i]) for i in range(n)])


def _eval_formula_vec(
    node: _ASTFormula,
    context: Dict[str, float],
    sv_name: str,
    sv_values: np.ndarray,
    fitted: Dict[tuple, ConditionalDensityEstimator],
    data: Dict[str, np.ndarray],
    resolve: Callable[[str], Optional[str]],
    cancel_marginal: Optional[str] = None,
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

        # Build X matrix for the full batch
        x_cols = []
        for cv in node.cond_vars:
            cv_paper = resolve(cv) or cv
            if cv_paper == sv_name or cv == sv_name:
                x_cols.append(sv_values)  # the batch variable
            else:
                val = context.get(cv_paper, context.get(cv))
                if val is None:
                    d = data.get(cv_paper) if data.get(cv_paper) is not None else data.get(cv)
                    val = float(np.mean(d)) if d is not None else 0.0
                x_cols.append(np.full(n, float(val)))
        X_batch = np.column_stack(x_cols) if x_cols else None

        # Outcome: check if a point-probability or expectation is requested
        out_paper = resolve(node.outcome_vars[0]) if node.outcome_vars else None
        y_val = context.get(out_paper) if out_paper else None
        if y_val is None and node.outcome_vars:
            y_val = context.get(node.outcome_vars[0])

        if y_val is not None:
            y_f = float(y_val)
            if est.family == "bernoulli":
                p = _batch_expectation(est, X_batch) if X_batch is not None else np.full(n, est._p_marginal)
                return p if int(round(y_f)) == 1 else (1.0 - p)
            elif est.family == "gaussian":
                mu = _batch_expectation(est, X_batch) if X_batch is not None else np.full(n, est._mu_marginal)
                if SCIPY_AVAILABLE:
                    return _sp_stats.norm.pdf(y_f, loc=mu, scale=est._sigma)
                z = (y_f - mu) / est._sigma
                return np.exp(-0.5 * z * z) / (est._sigma * np.sqrt(2 * np.pi))
            else:
                return np.array([est.prob(y_f, X_batch[i] if X_batch is not None else None) for i in range(n)])
        else:
            if X_batch is not None:
                return _batch_expectation(est, X_batch)
            return np.full(n, est.expectation(None))

    if isinstance(node, _ASTProduct):
        result = np.ones(n)
        for child in node.children:
            result = result * _eval_formula_vec(child, context, sv_name, sv_values,
                                                fitted, data, resolve, cancel_marginal)
        return result

    if isinstance(node, _ASTSum):
        # Nested sum: scalar fallback per sv_value
        return np.array([
            _eval_formula(node, {**context, sv_name: float(v)}, fitted, data,
                          resolve, 50, np.random.default_rng(int(abs(v) * 1e6) % (2**31)))
            for v in sv_values
        ])

    return np.ones(n)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def ast_to_estimator(
    ast: Any,
    data: Dict[str, np.ndarray],
    intervention_value: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
    n_mc_samples: int = 1000,
    per_unit: bool = True,
    random_seed: Optional[int] = 0,
) -> float:
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

        ``"nonparametric"`` *(default)*
            Discrete Y → conditional frequency table with k-NN.
            Continuous Y → Gaussian KDE.

    n_mc_samples : int, default 1000
        Number of Monte Carlo samples for continuous marginalisation.
    per_unit : bool, default True
        When ``True`` and **subunit-level** data is present for both the
        outcome and treatment variables, the HCM-specific per-unit
        regression approach (paper Appendix D.1) is used to compute the ATE
        directly, bypassing formula evaluation.  This is the recommended
        path for the confounder model.
    random_seed : int or None, default 0
        Seed for the Monte Carlo sampler (``None`` → non-deterministic).

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

    **Per-unit regression (Appendix D.1).**  When ``per_unit=True`` and the
    data contains 2-D arrays for both the outcome and treatment, the ATE is
    computed as the average over units of the per-unit conditional expectation
    estimated by fitting a separate ``ConditionalDensityEstimator`` on each
    unit's subunit observations.  This bypasses formula evaluation and
    directly implements the paper's estimator.

    **Formula evaluation fallback.**  When per-unit data is not available,
    the function walks the pyAgrum AST (or parses ``.toLatex()`` as fallback)
    to discover which conditional densities appear, fits estimators from the
    observed unit-level Q-variable arrays, and evaluates the formula term by
    term using Monte Carlo marginalisation for continuous summation variables.
    """
    families = distribution_families or {}
    rng = np.random.default_rng(random_seed)

    # ── 1. Enrich data: add Q-variables computed from subunit data ────────────
    enriched = compute_q_from_subunit_data(data)

    # ── 2. Resolve intervention map ───────────────────────────────────────────
    if isinstance(intervention_value, dict):
        iv_map: Dict[str, float] = dict(intervention_value)
    else:
        iv_map = {"__single__": float(intervention_value)}

    # ── 3. Per-unit shortcut (confounder / HCM Appendix D.1) ─────────────────
    if per_unit:
        _result = _try_per_unit_ate(enriched, iv_map, families)
        if _result is not None:
            return _result

    # ── 4. Build name resolution maps ─────────────────────────────────────────
    san_to_paper, paper_to_san = _build_name_maps(list(enriched.keys()))

    def resolve(name: str) -> Optional[str]:
        if name in enriched:
            return name
        candidate = san_to_paper.get(name)
        if candidate and candidate in enriched:
            return candidate
        return None

    # ── 5. Parse the formula AST ──────────────────────────────────────────────
    formula = _extract_formula(ast)

    # ── 6. Collect unique conditional terms ───────────────────────────────────
    cond_terms = formula.collect_conditionals()
    seen: set = set()
    unique_terms: List[Dict] = []
    for t in cond_terms:
        if t["key"] not in seen:
            seen.add(t["key"])
            unique_terms.append(t)

    # ── 7. Fit density estimators ─────────────────────────────────────────────
    fitted: Dict[tuple, ConditionalDensityEstimator] = {}

    for term in unique_terms:
        out_vars = term["outcome_vars"]
        cond_vars = term["cond_vars"]
        key = term["key"]

        # Determine family
        family = "nonparametric"
        for candidate in out_vars:
            paper_c = resolve(candidate) or candidate
            f = families.get(paper_c) or families.get(candidate)
            if f:
                family = f
                break

        # Outcome data
        out_paper = resolve(out_vars[0]) if out_vars else None
        if out_paper is None:
            warnings.warn(f"No data for outcome {out_vars}; skipping term.")
            continue
        Y_arr = np.asarray(enriched[out_paper], dtype=float).ravel()

        # Conditioning data
        X_cols = []
        for cv in cond_vars:
            cv_paper = resolve(cv)
            if cv_paper and cv_paper in enriched:
                col = np.asarray(enriched[cv_paper], dtype=float).ravel()
                if len(col) == len(Y_arr):
                    X_cols.append(col)
        X_arr = np.column_stack(X_cols) if X_cols else None

        est = ConditionalDensityEstimator(family=family)
        est.fit(Y_arr, X_arr)
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
        unit_vals = []
        for i in range(n_units):
            unit_ctx = {}
            for key_d, arr_d in enriched.items():
                # Skip outcome variables (should be estimated, not queried at a point).
                # Skip variables already fixed by intervention (they must not be overwritten).
                if key_d in all_outcome_vars or key_d in context:
                    continue
                arr_np = np.asarray(arr_d)
                if arr_np.ndim == 1 and i < len(arr_np):
                    unit_ctx[key_d] = float(arr_np[i])
            # Intervention values take precedence over observed data
            unit_ctx.update(context)
            unit_vals.append(
                _eval_formula(formula, unit_ctx, fitted, enriched, resolve, n_mc_samples, rng)
            )
        return float(np.mean(unit_vals))
    else:
        return _eval_formula(formula, context, fitted, enriched, resolve, n_mc_samples, rng)


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit shortcut
# ─────────────────────────────────────────────────────────────────────────────

def _try_per_unit_ate(
    enriched: Dict[str, np.ndarray],
    iv_map: Dict[str, float],
    families: Dict[str, str],
) -> Optional[float]:
    """
    Attempt per-unit ATE estimation (paper Appendix D.1).

    Looks for pairs of 2-D subunit arrays (treatment, outcome).  If found,
    fits a ``PerUnitQEstimator`` and returns the ATE.  Returns ``None`` if
    the required data is not present.
    """
    # Find intervention variable and value
    iv_val: Optional[float] = None
    iv_paper: Optional[str] = None
    single_val = iv_map.get("__single__")

    # Collect 2-D arrays
    subunit_keys = {k for k, v in enriched.items() if np.asarray(v).ndim == 2}
    if len(subunit_keys) < 2:
        return None

    # Try to identify treatment (intervention) and outcome from iv_map
    for k, v in iv_map.items():
        if k == "__single__":
            iv_val = v
            continue
        arr = enriched.get(k)
        if arr is not None and np.asarray(arr).ndim == 2:
            iv_paper = k
            iv_val = v
            break

    if iv_val is None:
        return None

    # If we only have a scalar intervention value (no named variable found),
    # try to guess treatment variable from subunit keys
    if iv_paper is None:
        # Heuristic: treat variable whose name suggests treatment (A, T, X, D)
        for k in sorted(subunit_keys):
            if any(hint in k.lower() for hint in ["a", "treat", "t", "x", "d"]):
                iv_paper = k
                break
    if iv_paper is None and subunit_keys:
        iv_paper = sorted(subunit_keys)[0]
    if iv_paper is None:
        return None

    # Identify outcome variable (different from treatment)
    outcome_candidates = [k for k in subunit_keys if k != iv_paper]
    if not outcome_candidates:
        return None
    # Prefer variables with "y" or "outcome" in name
    outcome_paper = None
    for k in outcome_candidates:
        if any(hint in k.lower() for hint in ["y", "outcome", "out"]):
            outcome_paper = k
            break
    if outcome_paper is None:
        outcome_paper = outcome_candidates[0]

    A_mat = np.asarray(enriched[iv_paper], dtype=float)
    Y_mat = np.asarray(enriched[outcome_paper], dtype=float)
    if A_mat.shape != Y_mat.shape:
        return None

    family = families.get(outcome_paper, "nonparametric")
    estimator = PerUnitQEstimator(family=family)
    estimator.fit(Y_mat, A_mat)
    return estimator.ate(float(iv_val))


def _infer_n_units(data: Dict[str, np.ndarray]) -> int:
    for arr in data.values():
        a = np.asarray(arr)
        if a.ndim == 1 and len(a) > 1:
            return len(a)
    return 1
