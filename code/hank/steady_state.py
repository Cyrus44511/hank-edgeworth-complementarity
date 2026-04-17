"""
steady_state.py
---------------
Solve for the steady-state of the HANK model with Edgeworth complementarity,
calibrating beta to match a target asset/wealth ratio.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from income_process import rouwenhorst, stationary_distribution
from household import (
    HouseholdParams, make_asset_grid, theta_by_wealth,
    solve_household, policy_a_next, stationary_distribution_az,
)


@dataclass
class SteadyState:
    # prices
    r: float
    w: float
    beta: float
    # aggregates
    C: float
    A: float
    Y: float
    G: float
    # objects needed for Jacobian
    c_policy: np.ndarray
    a_next: np.ndarray
    D: np.ndarray
    a_grid: np.ndarray
    y_grid: np.ndarray
    P: np.ndarray
    theta_grid: np.ndarray


def _solve_ss_given_beta(beta: float, theta_S: float, theta_H: float,
                         a_max: float, g_over_y: float,
                         rho_e: float, sigma_e2: float,
                         n_y: int, n_a: int,
                         r_target: float) -> Tuple[float, dict]:
    """Inner solve: return (A, full_dict) at a given beta."""
    params = HouseholdParams(beta=beta, n_a=n_a, a_max=a_max)
    a_grid = make_asset_grid(params)
    y_grid, P = rouwenhorst(n_y, rho_e, np.sqrt(sigma_e2))
    theta_grid = theta_by_wealth(a_grid, theta_S, theta_H)

    r = r_target
    w = 1.0
    Y = 1.0
    G = g_over_y * Y

    c_policy = solve_household(a_grid, y_grid, P, theta_grid, g=G,
                               r=r, w=w, params=params)
    a_next = policy_a_next(c_policy, a_grid, y_grid, r, w)
    D = stationary_distribution_az(a_next, a_grid, P)

    C = (D * c_policy).sum()
    A = (D.sum(axis=1) @ a_grid)

    return A, dict(r=r, w=w, beta=beta, C=C, A=A, Y=Y, G=G,
                   c_policy=c_policy, a_next=a_next, D=D,
                   a_grid=a_grid, y_grid=y_grid, P=P, theta_grid=theta_grid)


def solve_steady_state(theta_S: float, theta_H: float,
                       g_over_y: float = 0.20,
                       rho_e: float = 0.96,
                       sigma_e2: float = 0.015,
                       n_y: int = 5,
                       n_a: int = 120,
                       a_max: float = 20.0,
                       r_target: float = 0.01,
                       A_target: float = 0.56,
                       calibrate_beta: bool = True,
                       beta_init: float = 0.985,
                       tol: float = 1e-3,
                       maxit: int = 30) -> SteadyState:
    """Solve the HANK steady state.

    If calibrate_beta=True, bisect on beta so aggregate assets = A_target
    (annual wealth-to-income ratio; 0.56 is the US liquid wealth target).

    Otherwise, use beta_init as-is.
    """
    if not calibrate_beta:
        A, d = _solve_ss_given_beta(
            beta_init, theta_S, theta_H, a_max, g_over_y,
            rho_e, sigma_e2, n_y, n_a, r_target,
        )
        return SteadyState(**d)

    # Bisection on beta
    lo, hi = 0.94, 0.998
    for it in range(maxit):
        beta = 0.5 * (lo + hi)
        A, d = _solve_ss_given_beta(
            beta, theta_S, theta_H, a_max, g_over_y,
            rho_e, sigma_e2, n_y, n_a, r_target,
        )
        if abs(A - A_target) < tol:
            break
        if A > A_target:
            hi = beta
        else:
            lo = beta
    return SteadyState(**d)


if __name__ == "__main__":
    ss = solve_steady_state(theta_S=0.29, theta_H=-0.76, n_a=80, n_y=5)
    print(f"Steady state (calibrated):")
    print(f"  beta = {ss.beta:.4f}")
    print(f"  r    = {ss.r:.4f}")
    print(f"  C    = {ss.C:.4f}  (Y = {ss.Y}, G = {ss.G})")
    print(f"  A    = {ss.A:.4f}  (target 0.56)")
    # Wealth Gini
    wealth = ss.D.sum(axis=1)
    cum_pop = np.cumsum(wealth)
    cum_wealth = np.cumsum(wealth * ss.a_grid)
    total_wealth = cum_wealth[-1]
    if total_wealth > 1e-9:
        gini = 1 - 2 * np.trapezoid(cum_wealth / total_wealth, cum_pop)
        print(f"  Wealth Gini ~ {abs(gini):.4f}")
