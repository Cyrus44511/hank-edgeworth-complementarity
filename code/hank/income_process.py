"""
income_process.py
-----------------
Rouwenhorst discretization of an AR(1) income process in logs:

    log y_t = rho * log y_{t-1} + sigma_e * epsilon_t,    epsilon ~ N(0, 1)

The Rouwenhorst method is preferred over Tauchen for high persistence
(rho close to 1), which is empirically relevant for household income.
Produces a grid of N productivity states and an NxN Markov transition
matrix that exactly matches the unconditional mean, variance, and
autocorrelation of the underlying AR(1).

Reference: Kopecky & Suen (2010, Review of Economic Dynamics).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def rouwenhorst(n: int, rho: float, sigma_e: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize an AR(1): log y_t = rho * log y_{t-1} + sigma_e * eps_t

    Parameters
    ----------
    n : number of states (>= 2)
    rho : persistence
    sigma_e : standard deviation of the innovation

    Returns
    -------
    y : 1-D array of n exponentiated grid points (levels), normalized to mean 1
    P : n x n Markov transition matrix where P[i, j] = Pr(y_{t+1}=y_j | y_t=y_i)
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    p = (1 + rho) / 2.0
    q = p

    # Unconditional std of log y:
    sigma_y = sigma_e / np.sqrt(1.0 - rho ** 2)
    # Grid spans +/- sqrt(n - 1) * sigma_y
    psi = sigma_y * np.sqrt(n - 1)
    log_grid = np.linspace(-psi, psi, n)

    # Build transition matrix recursively
    P = np.array([[p, 1 - p], [1 - q, q]])
    for k in range(3, n + 1):
        P_new = np.zeros((k, k))
        P_new[:-1, :-1] += p * P
        P_new[:-1, 1:]  += (1 - p) * P
        P_new[1:, :-1]  += (1 - q) * P
        P_new[1:, 1:]   += q * P
        P_new[1:-1, :]  /= 2.0
        P = P_new

    y = np.exp(log_grid)
    y /= y.mean()          # normalize so E[y] = 1 in the stationary dist
    return y, P


def stationary_distribution(P: np.ndarray, tol: float = 1e-12,
                            maxit: int = 10_000) -> np.ndarray:
    """Compute the stationary distribution of a Markov chain."""
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(maxit):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    return pi / pi.sum()


if __name__ == "__main__":
    y, P = rouwenhorst(n=7, rho=0.96, sigma_e=np.sqrt(0.015))
    pi = stationary_distribution(P)
    print(f"Grid: {y}")
    print(f"Stationary dist: {pi}")
    print(f"Mean: {pi @ y:.4f}")
    print(f"Var(log y): {pi @ np.log(y) ** 2 - (pi @ np.log(y)) ** 2:.4f}")
