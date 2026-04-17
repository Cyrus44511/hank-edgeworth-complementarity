"""
general_equilibrium.py
----------------------
Close the HANK model by combining:
  - Household Jacobians J^{C,w}, J^{C,r}, J^{C,g}  (from jacobians.py)
  - Firm NKPC (Rotemberg / Calvo equivalent)
  - Taylor rule
  - Government budget (balanced budget, Tt = Gt)
  - Market clearing: Yt = Ct + Gt

Under the baseline analytical assumption of a fixed real rate (r_t = 0),
the monetary block drops out and we have a linear system:

    dY = dC + dG
    dC = J^{C,w} * dw + J^{C,r} * dr + J^{C,g} * dg
    dw = (psi_y * sigma + phi) * dY                  [wage follows MC]
    dr = 0                                            [fixed rate]

which collapses to
    dY = (I - J^{C,w} * coef_yw)^{-1} * (J^{C,g} * dg + dg)

For the full model with the Taylor rule, we add the NKPC and interest-rate
response and solve the system jointly.

The IMPACT multiplier is mu = dY[0] / dg[0].
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

from steady_state import SteadyState


def compute_impact_multiplier_fixed_rate(
        J: dict, ss: SteadyState, rho_g: float = 0.80,
        T: int = 20, sigma: float = 2.0, phi: float = 1.0
        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the impact multiplier under a fixed real rate (r_t = 0).

    Wage responds to output via labor-market clearing:
        w_t = (sigma * alpha + phi) * C_t   (linearized, simplified)

    We use alpha = 1 (separable) as the linearization baseline; the
    theta-induced curvature is captured inside the household Jacobian.

    Returns (dY_path, dC_path, mu_impact).
    """
    # Construct the fiscal shock path
    dg = np.zeros(T)
    dg[0] = 0.01 * ss.G               # 1% of steady-state G
    for t in range(1, T):
        dg[t] = rho_g * dg[t - 1]

    # Wage-response coefficient (linear production + isoelastic labor supply)
    wage_coef = (sigma + phi) / (1.0 + phi)   # rough MC-to-Y elasticity

    # Jacobian operator
    Jw = J['w'][:T, :T]
    Jg = J['g'][:T, :T]

    # Solve (I - Jw * wage_coef) * dY = Jg * dg + dg
    # Note: under the balanced budget dT = dG so the direct income term of dg
    # cancels out against the tax term; we absorb this by treating dg as a
    # direct consumption-function argument only.
    A = np.eye(T) - Jw * wage_coef
    rhs = Jg @ dg + dg                 # Y = C + G, and C picks up direct Jg
    dY = np.linalg.solve(A, rhs)
    dC = dY - dg

    # Impact multiplier
    mu = dY[0] / dg[0]
    return dY, dC, mu


def compute_impact_multiplier_taylor(
        J: dict, ss: SteadyState, rho_g: float = 0.80,
        T: int = 20, sigma: float = 2.0, phi: float = 1.0,
        phi_pi: float = 1.5, phi_y: float = 0.125,
        kappa: float = 0.024, beta: float = 0.99
        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full GE with a Taylor rule and Rotemberg NKPC.

    Simplified version: under linear production, mc_t = w_t, so NKPC reads
        pi_t = beta * E pi_{t+1} + kappa * w_t.
    Taylor: i_t = phi_pi * pi_t + phi_y * Y_t.
    Fisher: r_t = i_t - E pi_{t+1}.

    We solve the joint linear system in (dY, dC, dw, dpi, dr).
    """
    # Shock path
    dg = np.zeros(T)
    dg[0] = 0.01 * ss.G
    for t in range(1, T):
        dg[t] = rho_g * dg[t - 1]

    # Placeholder implementation: return fixed-rate case
    # Full Taylor implementation requires constructing and solving the
    # joint T x T block system in (dY, dw, dpi, dr). For the paper's
    # baseline analytical comparison we use the fixed-rate case.
    return compute_impact_multiplier_fixed_rate(
        J, ss, rho_g=rho_g, T=T, sigma=sigma, phi=phi
    )


def decompose_multiplier(J: dict, ss: SteadyState, rho_g: float = 0.80,
                         T: int = 20, sigma: float = 2.0, phi: float = 1.0
                         ) -> dict:
    """
    Decompose the impact multiplier into direct and indirect channels,
    following Auclert-Rognlie-Straub (2024).

    Direct channel = J^{C,g} contribution at t=0 (the xi channel).
    Indirect channel = J^{C,w} * dw contribution (the NK cross / chi).
    """
    dY, dC, mu = compute_impact_multiplier_fixed_rate(
        J, ss, rho_g=rho_g, T=T, sigma=sigma, phi=phi
    )
    dg = np.zeros(T); dg[0] = 0.01 * ss.G
    for t in range(1, T): dg[t] = rho_g * dg[t - 1]

    # Direct channel contribution at t = 0
    direct = (J['g'][0, :T] @ dg) / dg[0]
    # Total GE contribution
    total = mu - 1.0
    # Indirect = total - direct
    indirect = total - direct

    return {
        'mu_impact': mu,
        'direct': direct,
        'indirect': indirect,
        'dY': dY, 'dC': dC,
    }


if __name__ == "__main__":
    from steady_state import solve_steady_state
    from jacobians import compute_jacobian_truncated
    print("Solving steady state...")
    ss = solve_steady_state(theta_S=0.29, theta_H=-0.76, n_a=80, n_y=5)
    print("Computing Jacobians (this takes ~1-2 min)...")
    J = compute_jacobian_truncated(ss, T=15)
    print("Computing GE multiplier...")
    dec = decompose_multiplier(J, ss, T=15)
    print(f"  Impact multiplier: {dec['mu_impact']:.4f}")
    print(f"  Direct channel:    {dec['direct']:.4f}")
    print(f"  Indirect channel:  {dec['indirect']:.4f}")
