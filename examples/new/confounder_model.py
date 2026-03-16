"""
Confounder HCM DGP: binary U, A, Y; Beta-Bernoulli q^a and q^{y|a}.
Paper appendix; used by ate_recovery_demo and parametric_ate_pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist, bernoulli


def default_confounder_params(omega: float = 0.2):
    """Paper appendix: alpha^a(0)=0.5, alpha^a(1)=4, beta^a=1; alpha^{y|a}(a,u), beta^{y|a}=2."""
    return {
        "omega": omega,
        "alpha_a": (0.5, 4.0),
        "beta_a": 1.0,
        "alpha_ya": {(0, 0): 0.5, (1, 0): 2.0, (0, 1): 1.0, (1, 1): 4.0},
        "beta_ya": 2.0,
    }


class ConfounderModel:
    """Binary confounder HCM: U -> A, U -> Y, A -> Y. q^a and q^{y|a} are Beta-Bernoulli."""

    def __init__(self, omega: float, alpha_a: tuple, beta_a: float, alpha_ya: dict, beta_ya: float):
        self.omega = omega
        self.alpha_a = alpha_a
        self.beta_a = beta_a
        self.alpha_ya = alpha_ya
        self.beta_ya = beta_ya

    def sample(self, n: int, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (A, Y) shaped (n, m)."""
        U = bernoulli.rvs(self.omega, size=n, random_state=rng)
        mu_a = np.array([beta_dist.rvs(self.alpha_a[int(u)], self.beta_a, random_state=rng) for u in U])
        mu_ya_0 = np.array([beta_dist.rvs(self.alpha_ya[(0, int(u))], self.beta_ya, random_state=rng) for u in U])
        mu_ya_1 = np.array([beta_dist.rvs(self.alpha_ya[(1, int(u))], self.beta_ya, random_state=rng) for u in U])
        A = bernoulli.rvs(mu_a[:, None] * np.ones((n, m)), random_state=rng).astype(np.float64)
        Y = np.where(A == 0, bernoulli.rvs(mu_ya_0[:, None] * np.ones((n, m)), random_state=rng), bernoulli.rvs(mu_ya_1[:, None] * np.ones((n, m)), random_state=rng)).astype(np.float64)
        return A, Y

    def true_ate(self) -> float:
        """Closed-form ATE from paper."""
        o = self.omega
        a00 = self.alpha_ya[(0, 0)]
        a10 = self.alpha_ya[(1, 0)]
        a01 = self.alpha_ya[(0, 1)]
        a11 = self.alpha_ya[(1, 1)]
        b = self.beta_ya
        e1 = (1 - o) * (a10 / (a10 + b)) + o * (a11 / (a11 + b))
        e0 = (1 - o) * (a00 / (a00 + b)) + o * (a01 / (a01 + b))
        return float(e1 - e0)

    def estimate_ate(self, A: np.ndarray, Y: np.ndarray) -> float:
        """Per-unit E[Y|A=a] with pseudocounts, then average; paper estimator."""
        n, m = A.shape
        ate = 0.0
        for i in range(n):
            a_i, y_i = A[i], Y[i]
            n0 = max(int((a_i == 0).sum()), 1)
            n1 = max(int((a_i == 1).sum()), 1)
            mu_0 = (float(y_i[a_i == 0].sum()) + 1.0) / (n0 + 2.0)
            mu_1 = (float(y_i[a_i == 1].sum()) + 1.0) / (n1 + 2.0)
            ate += mu_1 - mu_0
        return ate / n
