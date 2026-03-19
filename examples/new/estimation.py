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
    # Positive heavy-tailed (variance/scale priors in HCM simulations)
    "half_cauchy", "halfcauchy",
    # Non-parametric fallback
    "nonparametric",
}

_FAMILY_ALIASES: Dict[str, str] = {
    "normal": "gaussian",
    "t": "student_t",
    "log_normal": "lognormal",
    "wald": "inverse_gaussian",
    "halfcauchy": "half_cauchy",
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

    **Positive heavy-tailed (variance/scale priors)**

    ``"half_cauchy"`` / ``"halfcauchy"``
        Y > 0.  Parameter γ (scale).  Half-Cauchy(0, γ).
        Unconditional: γ estimated via scipy MLE or median heuristic.
        Conditional: linear regression for scale.
        Used in HCM paper simulations as prior on variance parameters τ.

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
            "half_cauchy":      self._fit_half_cauchy,
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
            "half_cauchy":      self._eval_half_cauchy,
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
                "half_cauchy":      lambda: self._hc_mean,
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
            "half_cauchy":      self._predict_scale_hc,
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
        elif self.family == "half_cauchy":
            p["scale"] = self._hc_scale
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
        ``gamma``     (shape_i, scale_i) (n_units, 2)
        ``lognormal`` (μ_log_i, σ²_log_i) (n_units, 2)
        ``nonparametric`` (mean, std, skew, kurt) (n_units, 4)
        ============= ================== ============================
    """

    _SCALAR_FAMILIES: frozenset = frozenset({"bernoulli", "poisson", "exponential"})

    def __init__(self, family: str = "bernoulli") -> None:
        family = _FAMILY_ALIASES.get(family, family)
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported family {family!r}.  "
                f"Choose from: {sorted(SUPPORTED_FAMILIES)}."
            )
        self.family = family

    @property
    def n_params(self) -> int:
        """Dimensionality of the per-unit parameter vector."""
        if self.family in self._SCALAR_FAMILIES:
            return 1
        if self.family in {"gaussian", "normal", "beta", "gamma",
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

        if self.family == "poisson":
            return np.array([float(max(y.mean(), 1e-10))])

        if self.family == "exponential":
            return np.array([float(max(y.mean(), 1e-10))])

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
        try:
            self._kde = _sp_stats.gaussian_kde(
                samples.T, bw_method=self.bandwidth
            )
        except np.linalg.LinAlgError:
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
) -> Dict[str, np.ndarray]:
    """
    Enrich *data* with Q-variable arrays from raw subunit observations.

    For each key *v* whose value is a 2-D array ``(n_units, n_subunits)``,
    adds the corresponding Q-variable entry via :class:`SubunitParamEstimator`:

    * **Scalar families** (``bernoulli``, ``poisson``, ``exponential``):
      ``Q^v`` is the per-unit parameter as a 1-D array ``(n_units,)``.

    * **Multi-parameter families** (``gaussian`` → (μ_i, σ²_i); ``beta`` →
      (α_i, β_i); …): ``Q^v`` is a 2-D array ``(n_units, n_params)``.
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
        est = SubunitParamEstimator(family=family)
        result[q_key] = est.fit(arr_np)

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

        # Collect conditioning values from context
        x_vals = []
        for cv in node.cond_vars:
            cv_paper = resolve(cv) or cv
            val = context.get(cv_paper, context.get(cv))
            if val is None:
                d = data.get(cv_paper) or data.get(cv)
                if d is not None:
                    d_np = np.asarray(d, dtype=float)
                    # For multi-dim Q: use first param (primary) as scalar
                    val = float(d_np.mean() if d_np.ndim == 1
                                else d_np[:, 0].mean())
                else:
                    val = 0.0
            x_vals.append(float(val))
        x_q = np.array(x_vals) if x_vals else None

        # QDensityEstimator: return scalar_mean(X) as the expected Q value
        if isinstance(est, QDensityEstimator):
            return est.scalar_mean(x_q)

        # Standard ConditionalDensityEstimator path
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

    if fam == "half_cauchy":
        pred = _linreg_predict("_lr_hc")
        return np.maximum(pred, 1e-9) if pred is not None else np.full(n, est._hc_scale)

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
    # SubunitParamEstimator is used per variable according to the families dict.
    enriched = compute_q_from_subunit_data(data, families=families)

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
        Y_data = np.asarray(enriched[out_paper], dtype=float)

        # Determine reference length for conditioning data alignment
        n_ref = Y_data.shape[0]

        # Conditioning data: flatten multi-dim Q-params to columns
        X_cols = []
        for cv in cond_vars:
            cv_paper = resolve(cv)
            if cv_paper and cv_paper in enriched:
                col_data = np.asarray(enriched[cv_paper], dtype=float)
                if col_data.shape[0] != n_ref:
                    continue
                if col_data.ndim == 1:
                    X_cols.append(col_data)
                else:
                    # Multi-dim Q-param: add each parameter as its own column
                    for j in range(col_data.shape[1]):
                        X_cols.append(col_data[:, j])
        X_arr = np.column_stack(X_cols) if X_cols else None

        if Y_data.ndim == 2:
            # Multi-parameter Q-variable (e.g. Gaussian (μ, σ²) per unit).
            # Estimate P(Q = q | X = x) via KDE in the parameter space.
            est: Any = QDensityEstimator()
            est.fit(Y_data, x_cond=X_arr)
        else:
            # Unit-level scalar variable or scalar Q-variable.
            Y_arr = Y_data.ravel()
            if X_arr is not None and X_arr.shape[0] != len(Y_arr):
                X_arr = None
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
            unit_ctx: Dict[str, float] = {}
            for key_d, arr_d in enriched.items():
                if key_d in all_outcome_vars or key_d in context:
                    continue
                arr_np = np.asarray(arr_d, dtype=float)
                if arr_np.ndim == 1 and i < len(arr_np):
                    unit_ctx[key_d] = float(arr_np[i])
                elif arr_np.ndim == 2 and i < arr_np.shape[0]:
                    # Multi-param Q-variable: expose each param as a separate
                    # key suffix __0, __1, … and the primary param under the
                    # original key so conditioning works in _eval_formula.
                    for j, pval in enumerate(arr_np[i]):
                        unit_ctx[f"{key_d}__{j}"] = float(pval)
                    unit_ctx[key_d] = float(arr_np[i, 0])  # primary param
            unit_ctx.update(context)  # intervention overwrites observation
            unit_vals.append(
                _eval_formula(formula, unit_ctx, fitted, enriched, resolve,
                              n_mc_samples, rng)
            )
        return float(np.mean(unit_vals))
    else:
        return _eval_formula(formula, context, fitted, enriched, resolve,
                             n_mc_samples, rng)


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


# ─────────────────────────────────────────────────────────────────────────────
# Formula-driven dispatch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pyagrum_to_paper(name: str) -> str:
    """Convert a pyAgrum variable name to paper notation.

    Examples: ``Qa_z`` → ``Q^{a|z}``, ``Qy_a`` → ``Q^{y|a}``, ``Qa`` → ``Q^a``.
    """
    m = re.match(r'^Q([a-zA-Z]+)_([a-zA-Z]+)$', name)
    if m:
        return f'Q^{{{m.group(1)}|{m.group(2)}}}'
    m = re.match(r'^Q([a-zA-Z]+)$', name)
    if m:
        return f'Q^{m.group(1)}'
    return name


def _find_in_data(paper_name: str, data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Look up *paper_name* in *data* using progressively looser matching.

    1. Exact key match.
    2. Case-insensitive match.
    3. Normalised match (strip all non-alphanumeric characters, lowercase).
    """
    if paper_name in data:
        return np.asarray(data[paper_name], dtype=float)
    for k, v in data.items():
        if k.lower() == paper_name.lower():
            return np.asarray(v, dtype=float)

    def _norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', s.lower())

    norm_target = _norm(paper_name)
    for k, v in data.items():
        if _norm(k) == norm_target:
            return np.asarray(v, dtype=float)
    return None


def _analyze_formula(formula: _ASTFormula) -> Dict[str, Any]:
    """Classify an internal formula into one of three canonical HCM patterns.

    Returns a dict with key ``'pattern'`` set to one of:
    ``'confounder'``, ``'instrument'``, ``'ci'``, or ``'unknown'``.

    Detection rules
    ---------------
    * **C&I**: outermost node is ``_ASTSum`` whose body contains a nested
      ``_ASTSum`` (front-door formula).
    * **INSTRUMENT**: outermost ``_ASTSum`` whose sum variable matches the
      pattern ``a[_|]z`` (i.e. ``Qa_z`` / ``Q^{a|z}``).
    * **CONFOUNDER**: outermost ``_ASTSum`` whose sum variable matches the
      pattern ``[yw][_|]a`` (i.e. ``Qy_a`` / ``Q^{y|a}``).
    """
    if not isinstance(formula, _ASTSum):
        return {'pattern': 'unknown'}

    sum_vars: List[str] = formula.sum_vars
    body: _ASTFormula = formula.formula

    def _has_nested_sum(node: _ASTFormula) -> bool:
        if isinstance(node, _ASTSum):
            return True
        if isinstance(node, _ASTProduct):
            return any(_has_nested_sum(c) for c in node.children)
        return False

    if _has_nested_sum(body):
        return {'pattern': 'ci', 'sum_vars': sum_vars}

    for sv in sum_vars:
        if re.search(r'a[_|]z', sv, re.IGNORECASE) or 'a|z' in sv:
            return {'pattern': 'instrument', 'sum_var': sv, 'sum_vars': sum_vars}

    for sv in sum_vars:
        if re.search(r'[yw][_|]a', sv, re.IGNORECASE) or re.search(r'[yw]\|a', sv):
            return {'pattern': 'confounder', 'sum_var': sv, 'sum_vars': sum_vars}

    return {'pattern': 'unknown', 'sum_vars': sum_vars}


def _formula_aware_estimate(
    formula: _ASTFormula,
    data: Dict[str, np.ndarray],
    intervention_value: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """Dispatch to the optimal paper estimator based on *formula* structure.

    Analyses the internal ``_ASTFormula`` (produced by :func:`_extract_formula`)
    to decide which of the three canonical HCM estimators to invoke:

    * **CONFOUNDER** → :func:`estimate_confounder_ate`
    * **INSTRUMENT** → :func:`estimate_instrument_ate`
    * **C&I**        → :func:`estimate_confounder_interference_ate`

    Returns ``None`` for unrecognised patterns so that the caller can fall back
    to the generic formula evaluator.
    """
    info = _analyze_formula(formula)
    pattern = info.get('pattern', 'unknown')
    families = distribution_families or {}

    if isinstance(intervention_value, dict):
        vals = list(intervention_value.values())
        if not vals:
            return None
        iv_val = float(vals[0])
    else:
        iv_val = float(intervention_value)

    arrays_2d = {k: np.asarray(v, dtype=float) for k, v in data.items()
                 if np.asarray(v).ndim == 2}
    arrays_1d = {k: np.asarray(v, dtype=float) for k, v in data.items()
                 if np.asarray(v).ndim == 1}

    if pattern == 'instrument':
        # Resolve Q^{a|z} from the sum variable in the formula
        sum_var = info.get('sum_var', '')
        qaz = _find_in_data(_pyagrum_to_paper(sum_var), data)
        if qaz is None:
            for k, v in arrays_1d.items():
                kl = k.lower()
                if 'a|z' in k or '{a|z}' in k or (kl.startswith('q') and 'a' in kl and 'z' in kl):
                    qaz = v
                    break
        if qaz is None:
            return None
        qa = next(
            (v for k, v in arrays_1d.items()
             if k.lower().startswith('q') and 'a' in k.lower() and 'z' not in k.lower()),
            None,
        )
        y_key = next(
            (k for k in arrays_1d
             if k.lower().startswith('y') and 'q' not in k.lower()),
            None,
        )
        if qa is None or y_key is None:
            return None
        family = families.get(y_key, families.get('Y', 'bernoulli'))
        return estimate_instrument_ate(arrays_1d[y_key], qa, qaz, iv_val, family_outcome=family)

    elif pattern == 'confounder':
        if len(arrays_2d) < 2:
            return None
        a_key = next((k for k in arrays_2d if 'a' in k.lower()), None)
        y_key = next((k for k in arrays_2d if 'y' in k.lower()), None)
        if a_key is None or y_key is None or a_key == y_key:
            sorted_keys = sorted(arrays_2d.keys())
            a_key, y_key = sorted_keys[0], sorted_keys[1]
        family = families.get(y_key, families.get('Y', 'bernoulli'))
        return estimate_confounder_ate(arrays_2d[y_key], arrays_2d[a_key], iv_val, family=family)

    elif pattern == 'ci':
        if len(arrays_2d) < 2:
            return None
        a_key = next((k for k in arrays_2d if 'a' in k.lower()), None)
        y_key = next((k for k in arrays_2d if 'y' in k.lower()), None)
        if a_key is None or y_key is None or a_key == y_key:
            sorted_keys = sorted(arrays_2d.keys())
            a_key, y_key = sorted_keys[0], sorted_keys[1]
        z_key = next((k for k in arrays_1d if k.upper() == 'Z'), None)
        if z_key is None:
            return None
        family_out = families.get(y_key, families.get('Y', 'bernoulli'))
        family_med = families.get(z_key, families.get('Z', 'bernoulli'))
        return estimate_confounder_interference_ate(
            arrays_2d[y_key], arrays_2d[a_key], arrays_1d[z_key], iv_val,
            family_outcome=family_out, family_mediator=family_med,
        )

    return None


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
) -> float:
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

    2. **Model-specific optimal estimators** (Appendix D, Weinstein & Blei,
       2023) are used when the data structure matches a known pattern:

       - 2-D treatment + outcome (no mediator): per-unit regression
         (:class:`PerUnitQEstimator`, Appendix D.1, CONFOUNDER model).
       - 2-D treatment + outcome + unit-level ``Z``: front-door adjustment
         (:func:`estimate_confounder_interference_ate`, Appendix D.2,
         CONFOUNDER & INTERFERENCE model).
       - 1-D ``Y`` + ``Q^a`` + ``Q^{a|z}``: backdoor regression
         (:func:`estimate_instrument_ate`, Appendix D.3, INSTRUMENT model).

    3. **Generic formula evaluation** for any other identified model: each
       conditional term ``P(Y | X₁, …, Xₖ)`` in the pyAgrum ASTree is
       estimated from data:

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
        ``"half_cauchy"``, ``"nonparametric"`` (default).

    n_mc_samples : int, default 1000
        Monte Carlo samples for continuous marginalisation.
    random_seed : int or None, default 0

    Returns
    -------
    float
        Estimated causal effect ``E[Y | do(X = x*)]``.

    Raises
    ------
    ValueError
        If ``result`` has ``identifiable=False`` or carries no formula.

    See Also
    --------
    SubunitParamEstimator : Per-unit distribution parameter estimation.
    QDensityEstimator     : Density over Q-variable parameter space.
    PerUnitQEstimator     : Per-unit regression (Appendix D.1).
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
        if ast is None and formula_latex:
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

    # ── Model-specific optimal estimators (Appendix D) ────────────────────────
    # Analyse the identified formula to determine which canonical HCM estimator
    # to use.  The formula structure (not the data key names) is the ground
    # truth: it reflects what identify_effect() learned from the ASTree.
    formula = _extract_formula(ast)
    fast = _formula_aware_estimate(formula, data, intervention, distribution_families)
    if fast is not None:
        return fast

    # ── Generic formula evaluation (any identified model) ─────────────────────
    # Evaluate the ASTree literally: fit density estimators for each
    # conditional term and marginalise over summation variables.
    return ast_to_estimator(
        ast=ast,
        data=data,
        intervention_value=intervention,
        distribution_families=distribution_families,
        n_mc_samples=n_mc_samples,
        per_unit=False,   # disable per-unit shortcut; formula is evaluated directly
        random_seed=random_seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────────────

def estimate_from_do_calculus(
    result: Any,
    data: Dict[str, np.ndarray],
    intervention_value: Union[float, Dict[str, float]],
    distribution_families: Optional[Dict[str, str]] = None,
    n_mc_samples: int = 1000,
    per_unit: bool = True,
    random_seed: Optional[int] = 0,
) -> float:
    """
    Backward-compatible alias for :func:`estimate_causal_effect`.

    Prefer ``estimate_causal_effect`` for new code.
    """
    return estimate_causal_effect(
        result=result,
        data=data,
        intervention=intervention_value,
        distribution_families=distribution_families,
        n_mc_samples=n_mc_samples,
        random_seed=random_seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HCM model-specific standalone estimators (paper Appendix D)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_confounder_ate(
    Y_subunit: np.ndarray,
    A_subunit: np.ndarray,
    intervention_value: float,
    family: str = "bernoulli",
) -> float:
    """
    Estimate E[Y | do(A = a*)] for the CONFOUNDER HCM (Appendix D.1).

    Implements per-unit regression: for each unit i, fit E[Y_ij | A_ij, unit i]
    from within-unit subunit observations, then average over units at A = a*.

    Parameters
    ----------
    Y_subunit : (n_units, n_subunits) array
        Subunit outcome observations.
    A_subunit : (n_units, n_subunits) array
        Subunit treatment observations.
    intervention_value : float
        The do(A = a*) value.
    family : str, default "bernoulli"
        Distribution family for Y | A within each unit.  Use ``"bernoulli"``
        for binary outcomes, ``"gaussian"`` for continuous.

    Returns
    -------
    float
        Estimated ATE = (1/n) Σ_i E[Y_ij | A_ij = a*, unit i].
    """
    est = PerUnitQEstimator(family=family)
    est.fit(np.asarray(Y_subunit, dtype=float), np.asarray(A_subunit, dtype=float))
    return est.ate(float(intervention_value))


def estimate_confounder_interference_ate(
    Y_subunit: np.ndarray,
    A_subunit: np.ndarray,
    Z_unit: np.ndarray,
    intervention_value: float,
    family_outcome: str = "bernoulli",
    family_mediator: str = "bernoulli",
    n_mc: int = 500,
    random_seed: Optional[int] = 0,
) -> float:
    """
    Estimate E[Y | do(A = a*)] for the CONFOUNDER & INTERFERENCE HCM (Appendix D.2).

    Uses the front-door formula on the collapsed model (Eqs. 29-30).  The
    unit-level observable Z_i mediates between Q^a_i and Q^{y|a}_i, breaking
    the unobserved confounding path through U_i::

        E[Y|do(a)] ≈ (1/n) Σ_i Σ_z P(Z_i=z | Q^a_i) · E[Y_ij | A_ij=a, Z_i=z]

    where E[Y | A, Z] is estimated from pooled (flattened) subunit data with Z_i
    replicated across all subunits j of unit i.

    Parameters
    ----------
    Y_subunit : (n_units, n_subunits) array
    A_subunit : (n_units, n_subunits) array
    Z_unit : (n_units,) array
        Unit-level observable that is a descendant of Q^a and an ancestor of Y
        (the "mediator" / "interference proxy" in the C&I graph).
    intervention_value : float
        The do(A = a*) value.
    family_outcome : str, default "bernoulli"
        Family for Y | (A, Z) estimated from pooled subunit data.
    family_mediator : str, default "bernoulli"
        Family for Z | Q^a estimated from unit-level data.
    n_mc : int, default 500
        Monte Carlo samples used when Z is continuous.
    random_seed : int or None, default 0

    Returns
    -------
    float
        Estimated causal effect E[Y | do(A = a*)].
    """
    rng = np.random.default_rng(random_seed)
    Y_sub = np.asarray(Y_subunit, dtype=float)
    A_sub = np.asarray(A_subunit, dtype=float)
    Z_u = np.asarray(Z_unit, dtype=float).ravel()
    n_units, n_sub = Y_sub.shape

    # Q^a_i = within-unit mean of A (sufficient summary of U_i for the treatment)
    Q_a = A_sub.mean(axis=1)

    # Fit P(Z_i | Q^a_i) from unit-level data
    z_est = ConditionalDensityEstimator(family=family_mediator)
    z_est.fit(Z_u, Q_a.reshape(-1, 1))

    # Fit E[Y_ij | A_ij, Z_i] from pooled (flattened) subunit data
    Y_flat = Y_sub.ravel()
    A_flat = A_sub.ravel()
    Z_rep = np.repeat(Z_u, n_sub)
    X_yz = np.column_stack([A_flat, Z_rep])
    y_est = ConditionalDensityEstimator(family=family_outcome)
    y_est.fit(Y_flat, X_yz)

    is_discrete_z = np.all(Z_u == Z_u.astype(int)) and len(np.unique(Z_u)) <= 20

    unit_vals: List[float] = []
    for i in range(n_units):
        qa_i = float(Q_a[i])
        if is_discrete_z:
            z_vals = np.unique(Z_u)
            sum_val = 0.0
            for z_v in z_vals:
                p_z = z_est.prob(float(z_v), np.array([qa_i]))
                ey = y_est.expectation(np.array([intervention_value, float(z_v)]))
                sum_val += p_z * ey
        else:
            mc_z = rng.choice(Z_u, size=n_mc)
            x_queries = np.column_stack([
                np.full(n_mc, intervention_value),
                mc_z,
            ])
            ey_vals = _batch_expectation(y_est, x_queries)
            sum_val = float(np.mean(ey_vals))
        unit_vals.append(sum_val)

    return float(np.mean(unit_vals))


def estimate_instrument_ate(
    Y_unit: np.ndarray,
    Q_a_unit: np.ndarray,
    Q_az_unit: np.ndarray,
    intervention_value: float,
    family_outcome: str = "bernoulli",
) -> float:
    """
    Estimate E[Y | do(A = a*)] for the INSTRUMENT HCM (Appendix D.3).

    Uses the backdoor adjustment formula (Eq. 34) in the collapsed model.
    The instrument Z_i breaks the U_i → A_ij confounding path.  In the
    collapsed model the adjustment set is Q^a_i and Q^{a|z}_i::

        E[Y|do(a)] = (1/n) Σ_i E[Y_i | Q^a_i = a, Q^{a|z}_i]

    where Q^{a|z}_i = E[A_ij | Z_i] is the compliance/first-stage rate for
    unit i, and we set Q^a = a (the intervention) while marginalizing over the
    observed distribution of Q^{a|z}_i (backdoor adjustment formula).

    Parameters
    ----------
    Y_unit : (n_units,) or (n_units, n_subunits) array
        Unit-level outcome (or subunit matrix, averaged automatically).
    Q_a_unit : (n_units,) array
        Unit-level marginal mean of A: Q^a_i = (1/m) Σ_j A_ij.
    Q_az_unit : (n_units,) array
        Unit-level compliance rate Q^{a|z}_i = E[A_ij | Z_i].
        Typically estimated as the within-unit mean of A among units sharing
        the same Z_i value, or from a first-stage regression A ~ Z within each unit.
    intervention_value : float
        The do(A = a*) value; replaces Q^{a|z}_i in the prediction.
    family_outcome : str, default "bernoulli"
        Distribution family for Y | (Q^a, Q^{a|z}).

    Returns
    -------
    float
        Estimated causal effect E[Y | do(A = a*)].
    """
    Y_u = np.asarray(Y_unit, dtype=float)
    if Y_u.ndim == 2:
        Y_u = Y_u.mean(axis=1)
    Y_u = Y_u.ravel()
    Q_a = np.asarray(Q_a_unit, dtype=float).ravel()
    Q_az = np.asarray(Q_az_unit, dtype=float).ravel()
    n_units = len(Y_u)

    # ── Backdoor adjustment via Q^{a|z} ──────────────────────────────────────
    # Identification formula (from do-calculus on collapsed INSTRUMENT model):
    #
    #   E[Y|do(Q^a=a*)] = Σ_q P(Y|Q^a=a*, Q^{a|z}=q) · P(Q^{a|z}=q)
    #
    # Q^{a|z}_i is a valid adjustment set: it blocks the only backdoor path
    # Q^a ← Q^{a|z} ← U → Y in the marginalized collapsed graph.
    #
    # Implementation: fit E[Y | Q^a, Q^{a|z}] by logistic regression, then set
    # Q^a = intervention_value and marginalize over the observed Q^{a|z} sample.
    #
    # ⚠ Estimation quality depends on instrument strength.  When the Z→A
    #   coefficient is small relative to U→A, Q^a and Q^{a|z} both track U
    #   with nearly the same slope, making the regression ill-conditioned and
    #   producing biased estimates (|error| ≈ 0.10–0.20 at N=200 units).
    #   Increasing N does not fully remedy this because the estimator converges
    #   to a bias fixed point caused by structural collinearity.  A stronger
    #   Z coefficient or more subunits per unit is required for reliable
    #   identification.

    X = np.column_stack([Q_a, Q_az])
    y_est = ConditionalDensityEstimator(family=family_outcome)
    y_est.fit(Y_u, X)

    # Backdoor: set Q^a = intervention_value, marginalize over observed Q^{a|z}
    # P(Y|do(Q^a=a)) = (1/n) Σ_i E[Y | Q^a=a, Q^{a|z}=Q^{a|z}_i]
    X_pred = np.column_stack([np.full(n_units, intervention_value), Q_az])
    unit_vals = _batch_expectation(y_est, X_pred)
    return float(np.mean(unit_vals))
