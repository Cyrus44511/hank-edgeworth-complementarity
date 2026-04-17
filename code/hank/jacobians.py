"""
jacobians.py
------------
Compute the partial-equilibrium Jacobians of aggregate consumption with
respect to the paths of {w, r, g}, following Auclert, Bardóczy, Rognlie,
Straub (2021, Econometrica).

The Jacobian J^{C,x}[t, s] gives the impact at horizon t of a unit
perturbation to x at horizon s, holding everything else fixed at the
steady state. We compute it via direct perturbation (not the fake-news
algorithm) -- slower but transparent and dependency-free.

For each input x in {w, r, g} and each shock horizon s in [0, T):
  1. Perturb x by epsilon at horizon s (zero elsewhere).
  2. Solve the household problem BACKWARDS from t = T to t = 0.
  3. Compute the implied path of aggregate consumption.
  4. Numerical derivative gives column s of the Jacobian.

For T-horizon dynamics this is O(T^2) household solves, but each is fast
because we use the steady-state policy as the warm start.
"""

from __future__ import annotations
import numpy as np

from household import (
    HouseholdParams, util_marginal, util_marginal_inv, policy_a_next,
)
from steady_state import SteadyState


def aggregate_consumption_one_path(
        ss: SteadyState, w_path: np.ndarray, r_path: np.ndarray,
        g_path: np.ndarray, params: HouseholdParams) -> np.ndarray:
    """
    Solve household problem backwards along (w_path, r_path, g_path) and
    return the path of aggregate consumption (length T).

    Uses transition-distribution iteration starting from steady-state
    distribution.
    """
    T = len(w_path)
    n_a, n_y = ss.c_policy.shape
    a_grid = ss.a_grid
    y_grid = ss.y_grid
    P = ss.P
    theta_grid = ss.theta_grid

    # 1) Solve household policy backwards
    c_path = [None] * T
    c_next = ss.c_policy.copy()
    for t in range(T - 1, -1, -1):
        c_eff_next = c_next + theta_grid[:, None] * g_path[min(t + 1, T - 1)]
        uc_next = util_marginal(c_eff_next, params.sigma)
        Euc_next = uc_next @ P.T
        uc_today = params.beta * (1.0 + r_path[t]) * Euc_next
        c_eff_today = util_marginal_inv(uc_today, params.sigma)
        c_today = c_eff_today - theta_grid[:, None] * g_path[t]
        c_today = np.maximum(c_today, 1e-9)
        a_today = (c_today + a_grid[:, None] - w_path[t] * y_grid[None, :]
                   ) / (1.0 + r_path[t])
        c_pol = np.zeros_like(c_next)
        for j in range(n_y):
            c_pol[:, j] = np.interp(a_grid, a_today[:, j], c_today[:, j])
            constrained = a_grid < a_today[0, j]
            c_pol[constrained, j] = (
                (1.0 + r_path[t]) * a_grid[constrained]
                + w_path[t] * y_grid[j] - params.a_min
            )
            c_pol[constrained, j] = np.maximum(c_pol[constrained, j], 1e-9)
        c_path[t] = c_pol
        c_next = c_pol

    # 2) Forward-iterate the distribution
    D = ss.D.copy()
    C_agg = np.zeros(T)
    for t in range(T):
        a_next = policy_a_next(c_path[t], a_grid, y_grid, r_path[t], w_path[t])
        C_agg[t] = (D * c_path[t]).sum()
        # Update distribution for t+1 using lottery
        D_new = np.zeros_like(D)
        for j in range(n_y):
            for jp in range(n_y):
                p_jjp = P[j, jp]
                if p_jjp < 1e-12:
                    continue
                ap = a_next[:, j]
                ap = np.clip(ap, a_grid[0], a_grid[-1])
                k = np.searchsorted(a_grid, ap) - 1
                k = np.clip(k, 0, n_a - 2)
                wt = (a_grid[k + 1] - ap) / (a_grid[k + 1] - a_grid[k])
                m = D[:, j] * p_jjp
                np.add.at(D_new[:, jp], k, m * wt)
                np.add.at(D_new[:, jp], k + 1, m * (1 - wt))
        D = D_new

    return C_agg


def compute_jacobian(ss: SteadyState, T: int = 200, eps: float = 1e-4,
                     params: HouseholdParams | None = None) -> dict:
    """
    Compute partial-equilibrium Jacobians of C with respect to {w, r, g}.

    Returns a dict with keys 'w', 'r', 'g', each containing a (T, T) matrix.
    Element [t, s] is dC_t / dx_s.

    NOTE: O(T^2) household solves; for T = 200 and a 200x5 grid this takes
    roughly 1 minute. For sparse Jacobians, use the fake-news algorithm
    (Auclert et al. 2021).
    """
    if params is None:
        params = HouseholdParams()
    w_ss = np.full(T, ss.w)
    r_ss = np.full(T, ss.r)
    g_ss = np.full(T, ss.G)

    # Baseline path
    C_ss = aggregate_consumption_one_path(ss, w_ss, r_ss, g_ss, params)

    J = {'w': np.zeros((T, T)), 'r': np.zeros((T, T)), 'g': np.zeros((T, T))}

    # Loop over horizon of perturbation -- this is the costly step
    for s in range(T):
        # Perturb w at horizon s
        wp = w_ss.copy(); wp[s] += eps
        C_w = aggregate_consumption_one_path(ss, wp, r_ss, g_ss, params)
        J['w'][:, s] = (C_w - C_ss) / eps

        # Perturb r
        rp = r_ss.copy(); rp[s] += eps
        C_r = aggregate_consumption_one_path(ss, w_ss, rp, g_ss, params)
        J['r'][:, s] = (C_r - C_ss) / eps

        # Perturb g
        gp = g_ss.copy(); gp[s] += eps
        C_g = aggregate_consumption_one_path(ss, w_ss, r_ss, gp, params)
        J['g'][:, s] = (C_g - C_ss) / eps

    return J


def compute_jacobian_truncated(ss: SteadyState, T: int = 50,
                               params: HouseholdParams | None = None
                               ) -> dict:
    """Faster version with shorter horizon (good for impact-multiplier purposes).

    For impact effects only we need T = 1; for IRFs T = 30-50 is enough.
    """
    return compute_jacobian(ss, T=T, params=params)


if __name__ == "__main__":
    from steady_state import solve_steady_state
    ss = solve_steady_state(theta_S=0.29, theta_H=-0.76, n_a=80, n_y=5)
    print("Computing Jacobian (T=20, this takes ~30s)...")
    J = compute_jacobian_truncated(ss, T=20)
    print(f"J^{{C,g}}[0,0] = {J['g'][0, 0]:.4f}")
    print(f"J^{{C,w}}[0,0] = {J['w'][0, 0]:.4f}")
    print(f"J^{{C,r}}[0,0] = {J['r'][0, 0]:.4f}")
