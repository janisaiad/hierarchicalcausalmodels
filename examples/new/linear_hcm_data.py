"""
Hierarchical linear causal DGP: confounder HCM with Gaussian unit/subunit noise.
We generate (U, A, Y) and provide true ATE for comparison with estimates.
"""
from __future__ import annotations

import numpy as np


def default_linear_confounder_params(
    beta_a: float = 1.0,
    beta_u: float = 0.8,
    gamma_u: float = 0.6,
    sigma_u: float = 1.0,
    sigma_a: float = 1.0,
    sigma_y: float = 1.0,
) -> dict:
    """Default params for linear confounder HCM: Y = beta_a*A + beta_u*U + noise, A = gamma_u*U + noise."""
    return {
        "beta_a": beta_a,
        "beta_u": beta_u,
        "gamma_u": gamma_u,
        "sigma_u": sigma_u,
        "sigma_a": sigma_a,
        "sigma_y": sigma_y,
    }


class LinearConfounderHCM:
    """
    Hierarchical linear confounder: U_i (unit) -> A_ij, U_i -> Y_ij, A_ij -> Y_ij.
    A_ij = gamma_u * U_i + eps^a_ij,  Y_ij = beta_a * A_ij + beta_u * U_i + eps^y_ij.
    """

    def __init__(
        self,
        beta_a: float,
        beta_u: float,
        gamma_u: float,
        sigma_u: float = 1.0,
        sigma_a: float = 1.0,
        sigma_y: float = 1.0,
    ):
        self.beta_a = beta_a
        self.beta_u = beta_u
        self.gamma_u = gamma_u
        self.sigma_u = sigma_u
        self.sigma_a = sigma_a
        self.sigma_y = sigma_y

    def sample(
        self, n: int, m: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (U, A, Y). U shape (n,), A and Y shape (n, m)."""
        U = rng.normal(0, self.sigma_u, size=n)
        eps_a = rng.normal(0, self.sigma_a, size=(n, m))
        eps_y = rng.normal(0, self.sigma_y, size=(n, m))
        A = self.gamma_u * U[:, None] + eps_a
        Y = self.beta_a * A + self.beta_u * U[:, None] + eps_y
        return U, A, Y

    def true_ate(self) -> float:
        """ATE = E[Y|do(A=a1)] - E[Y|do(A=a0)]. In this linear SCM, do(A=a) gives E[Y] = beta_a*a + beta_u*E[U] = beta_a*a (E[U]=0). So ATE = beta_a * (a1 - a0). We use a1=1, a0=0."""
        return float(self.beta_a)

    def true_ate_general(self, a1: float, a0: float) -> float:
        """ATE for do(A=a1) vs do(A=a0)."""
        return float(self.beta_a * (a1 - a0))


def estimate_ate_linear_per_unit(A: np.ndarray, Y: np.ndarray) -> float:
    """Per-unit linear regression E[Y|A] then E[Y|A=1]-E[Y|A=0], average over units. A, Y shape (n, m)."""
    n = A.shape[0]
    ate_list: list[float] = []
    for i in range(n):
        a_i = A[i].reshape(-1, 1)
        y_i = Y[i]
        if a_i.size < 2 or np.ptp(a_i) < 1e-10:
            ate_list.append(0.0)
            continue
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(a_i, y_i)
        pred_0 = float(reg.predict([[0.0]])[0])
        pred_1 = float(reg.predict([[1.0]])[0])
        ate_list.append(pred_1 - pred_0)
    return float(np.mean(ate_list)) if ate_list else 0.0
