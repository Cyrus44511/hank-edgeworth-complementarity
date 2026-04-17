"""
household.py
------------
Endogenous-grid-method (EGM) solver for the household problem with
Edgeworth complementarity:

    max  E sum_t beta^t [ ((c_t + theta(a)*g_t)^(1-sigma) - 1) / (1-sigma)
                          - chi * (n_t)^(1+phi) / (1+phi) ]
    s.t. c_t + a_{t+1} = (1 + r) * a_t + w_t * y_t * n_t + transfers
         a_{t+1} >= a_min

where theta depends on the household's WEALTH POSITION (not its current
state), so we treat theta as a function of position on the asset grid.
This is the natural HANK extension of the THANK two-type model: bottom of
the wealth distribution gets theta_H, top gets theta_S, with smooth
interpolation in between.

Inelastic labor (n = y) is assumed for simplicity in this implementation.
The labor-supply elasticity is captured at the aggregate level through the
NKPC. Adding endogenous labor at the household level is straightforward
but adds another dimension to the EGM iteration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class HouseholdParams:
    sigma:    float = 2.0           # IES inverse
    beta:     float = 0.99          # discount factor
    a_min:    float = -0.25         # borrowing limit (in steady-state-y units)
    a_max:    float = 50.0          # upper bound of asset grid
    n_a:      int   = 200           # number of asset grid points
    grid_curv: float = 3.0          # curvature: dense grid near a_min
    tol:      float = 1e-7
    maxit:    int   = 1500


def make_asset_grid(params: HouseholdParams) -> np.ndarray:
    """Construct an exponential grid concentrated near the borrowing limit."""
    u = np.linspace(0.0, 1.0, params.n_a)
    # exponential transformation: dense near 0, sparse near 1
    g = (np.exp(params.grid_curv * u) - 1.0) / (np.exp(params.grid_curv) - 1.0)
    return params.a_min + (params.a_max - params.a_min) * g


def theta_by_wealth(a_grid: np.ndarray, theta_S: float, theta_H: float
                    ) -> np.ndarray:
    """
    Map theta to each point on the asset grid.

    Bottom 1/3 of grid -> theta_H (constrained-type complementarity)
    Top 1/3            -> theta_S (saver substitutability)
    Middle 1/3         -> linear interpolation

    Mapping uses asset RANK not value, so it adapts to the chosen grid.
    """
    n = len(a_grid)
    weights = np.ones(n) * 0.5
    n_third = n // 3
    weights[:n_third] = 1.0
    weights[-n_third:] = 0.0
    # Linear interp in middle
    weights[n_third:-n_third] = np.linspace(1.0, 0.0, n - 2 * n_third)
    return weights * theta_H + (1 - weights) * theta_S


# -----------------------------------------------------------------------------
# Steady-state household problem (EGM)
# -----------------------------------------------------------------------------
def util_marginal(c_eff: np.ndarray, sigma: float) -> np.ndarray:
    """Marginal utility of effective consumption: u'(c_eff) = c_eff^{-sigma}"""
    return np.maximum(c_eff, 1e-12) ** (-sigma)


def util_marginal_inv(uc: np.ndarray, sigma: float) -> np.ndarray:
    """Inverse of marginal utility."""
    return np.maximum(uc, 1e-12) ** (-1.0 / sigma)


def egm_step(c_next: np.ndarray, a_grid: np.ndarray,
             y_grid: np.ndarray, P: np.ndarray,
             theta_grid: np.ndarray, g: float,
             r: float, w: float, params: HouseholdParams
             ) -> np.ndarray:
    """
    One EGM iteration. Given consumption next period c_next on the (a,y) grid,
    return the new consumption policy this period.

    c_next has shape (n_a, n_y).
    """
    n_a, n_y = c_next.shape
    # Effective consumption next period
    c_eff_next = c_next + theta_grid[:, None] * g
    # Marginal utility next period
    uc_next = util_marginal(c_eff_next, params.sigma)
    # Expected MU under transition matrix P (rows = today's y, cols = tomorrow's y)
    Euc_next = uc_next @ P.T            # shape (n_a, n_y)
    # Today's MU from Euler:  u'(c_eff_today) = beta * (1+r) * E[u'(c_eff_next)]
    uc_today = params.beta * (1.0 + r) * Euc_next
    # Today's effective consumption
    c_eff_today = util_marginal_inv(uc_today, params.sigma)
    # Today's actual consumption
    c_today = c_eff_today - theta_grid[:, None] * g
    c_today = np.maximum(c_today, 1e-9)
    # Today's implied assets via budget: a_today = (c + a_next - w*y) / (1+r)
    a_today = (c_today + a_grid[:, None] - w * y_grid[None, :]) / (1.0 + r)
    # Now interpolate back onto the regular asset grid
    c_policy = np.zeros_like(c_next)
    for j in range(n_y):
        # For asset values below a_today[0,j], household is constrained
        c_policy[:, j] = np.interp(a_grid, a_today[:, j], c_today[:, j])
        # Constrained households: a_next = a_min, so c = (1+r)*a + w*y - a_min
        constrained = a_grid < a_today[0, j]
        c_policy[constrained, j] = (
            (1.0 + r) * a_grid[constrained] + w * y_grid[j] - params.a_min
        )
        c_policy[constrained, j] = np.maximum(c_policy[constrained, j], 1e-9)
    return c_policy


def solve_household(a_grid: np.ndarray, y_grid: np.ndarray, P: np.ndarray,
                    theta_grid: np.ndarray, g: float, r: float, w: float,
                    params: HouseholdParams,
                    c_init: np.ndarray | None = None) -> np.ndarray:
    """Iterate EGM to convergence and return the optimal consumption policy."""
    n_a, n_y = len(a_grid), len(y_grid)
    if c_init is None:
        # Initial guess: spend all current resources
        c_init = (1.0 + r) * a_grid[:, None] + w * y_grid[None, :] - a_grid[0]
        c_init = np.maximum(c_init, 1e-3)
    c = c_init.copy()
    for it in range(params.maxit):
        c_new = egm_step(c, a_grid, y_grid, P, theta_grid, g, r, w, params)
        diff = np.max(np.abs(c_new - c))
        c = c_new
        if diff < params.tol:
            break
    return c


# -----------------------------------------------------------------------------
# Stationary distribution over (a, y)
# -----------------------------------------------------------------------------
def policy_a_next(c: np.ndarray, a_grid: np.ndarray, y_grid: np.ndarray,
                  r: float, w: float) -> np.ndarray:
    """Implied next-period asset policy from consumption policy."""
    return (1.0 + r) * a_grid[:, None] + w * y_grid[None, :] - c


def stationary_distribution_az(a_next: np.ndarray, a_grid: np.ndarray,
                               P: np.ndarray, tol: float = 1e-10,
                               maxit: int = 5000) -> np.ndarray:
    """
    Compute the stationary distribution over (a, y) via iteration.
    Uses lottery (linear interpolation) for off-grid asset realizations.
    """
    n_a, n_y = a_next.shape
    # Pre-compute interpolation weights
    idx = np.zeros((n_a, n_y), dtype=int)
    weight = np.zeros((n_a, n_y))
    for j in range(n_y):
        for i in range(n_a):
            ap = a_next[i, j]
            ap = max(min(ap, a_grid[-1]), a_grid[0])
            k = np.searchsorted(a_grid, ap) - 1
            k = max(0, min(k, n_a - 2))
            idx[i, j] = k
            weight[i, j] = (a_grid[k + 1] - ap) / (a_grid[k + 1] - a_grid[k])

    # Initial uniform distribution
    D = np.ones((n_a, n_y)) / (n_a * n_y)

    for it in range(maxit):
        D_new = np.zeros_like(D)
        for j in range(n_y):
            for jp in range(n_y):
                p = P[j, jp]
                if p < 1e-12:
                    continue
                # Mass at (i, j) goes to (idx, jp) and (idx+1, jp)
                m = D[:, j] * p
                np.add.at(D_new[:, jp], idx[:, j], m * weight[:, j])
                np.add.at(D_new[:, jp], idx[:, j] + 1, m * (1 - weight[:, j]))
        diff = np.max(np.abs(D_new - D))
        D = D_new
        if diff < tol:
            break
    return D / D.sum()


if __name__ == "__main__":
    from income_process import rouwenhorst
    params = HouseholdParams(n_a=100)
    a_grid = make_asset_grid(params)
    y_grid, P = rouwenhorst(n=5, rho=0.96, sigma_e=np.sqrt(0.015))
    theta_grid = theta_by_wealth(a_grid, theta_S=0.29, theta_H=-0.76)
    c = solve_household(a_grid, y_grid, P, theta_grid, g=0.20,
                        r=0.01, w=1.0, params=params)
    a_next = policy_a_next(c, a_grid, y_grid, 0.01, 1.0)
    D = stationary_distribution_az(a_next, a_grid, P)
    print(f"Aggregate consumption: {(D * c).sum():.4f}")
    print(f"Aggregate assets:      {(D.sum(axis=1) @ a_grid):.4f}")
