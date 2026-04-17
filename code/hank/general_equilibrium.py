"""
general_equilibrium.py
----------------------
Close the HANK model in sequence space:

  Household block:  dC = J^{Cw} dw + J^{Cr} dr + J^{Cg} dg
  Labor market:     dw = (sigma + phi) dY                (linear production)
  NKPC (Rotemberg): dpi = beta * F * dpi + kappa * dw
                    => dpi = (I - beta F)^{-1} * kappa * dw   ==  M_pi * dw
  Taylor rule:      di  = phi_pi * dpi + phi_y * dY
  Fisher:           dr  = di - F * dpi
                       = (phi_pi * I - F) dpi + phi_y dY
                       = M_r * dw + phi_y * dY
  Goods market:     dY  = dC + dg

Collapsing, with M_wY = (sigma+phi) * I:

  dC = (J^{Cw} + J^{Cr} M_r) M_wY dY + J^{Cr} phi_y dY + J^{Cg} dg
     = K * dY + J^{Cg} dg

  dY = dC + dg
     = K dY + (I + J^{Cg}) dg
  => (I - K) dY = (I + J^{Cg}) dg
  => dY = (I - K)^{-1} (I + J^{Cg}) dg.

Two closures are provided:
  * compute_impact_multiplier_fixed_rate : r = 0 baseline (analytical benchmark)
  * compute_impact_multiplier_taylor     : full NKPC + Taylor rule closure

At fixed rate with phi_pi = infinity we recover the analytical THANK multiplier.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

from steady_state import SteadyState


# -----------------------------------------------------------------------------
# Forward-shift and NKPC operators
# -----------------------------------------------------------------------------
def shift_forward(T: int) -> np.ndarray:
    """Operator F such that (F x)[t] = x[t+1] for t < T-1, else 0."""
    F = np.zeros((T, T))
    for t in range(T - 1):
        F[t, t + 1] = 1.0
    return F


def M_pi(T: int, beta: float, kappa: float) -> np.ndarray:
    """NKPC operator: pi = M_pi * w, where (I - beta F) pi = kappa w."""
    F = shift_forward(T)
    return np.linalg.solve(np.eye(T) - beta * F, kappa * np.eye(T))


def M_r(T: int, beta: float, kappa: float, phi_pi: float) -> np.ndarray:
    """
    r = (phi_pi*I - F) pi = (phi_pi*I - F) * M_pi * w + phi_y*I Y
    This returns the coefficient on w only (phi_y*Y term is handled separately).
    """
    F = shift_forward(T)
    Mp = M_pi(T, beta, kappa)
    return (phi_pi * np.eye(T) - F) @ Mp


# -----------------------------------------------------------------------------
# Fiscal shock path
# -----------------------------------------------------------------------------
def fiscal_shock_path(ss: SteadyState, T: int, rho_g: float = 0.80,
                      size: float = 0.01) -> np.ndarray:
    """AR(1) fiscal shock: dg_t = rho_g * dg_{t-1}, dg_0 = size * G."""
    dg = np.zeros(T)
    dg[0] = size * ss.G
    for t in range(1, T):
        dg[t] = rho_g * dg[t - 1]
    return dg


# -----------------------------------------------------------------------------
# Fixed-rate closure (paper's analytical benchmark)
# -----------------------------------------------------------------------------
def compute_impact_multiplier_fixed_rate(
        J: dict, ss: SteadyState, rho_g: float = 0.80,
        T: int = 20, sigma: float = 2.0, phi: float = 1.0,
        wage_coef: float | None = None,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fixed real rate (dr = 0). Wage responds to output via labor-market clearing.

    The effective wage-to-output elasticity depends on the aggregation:
    under inelastic labor and separable preferences, dw/dY = (sigma + phi),
    but in THANK with HtM absorbing labor income, the cleaner elasticity
    (matching Bilbiie 2025) is (sigma + phi)/(1 + phi). We use the latter
    by default since our Jacobians are built around a HANK steady state with
    HtM-like behavior at the bottom of the wealth distribution.
    """
    if wage_coef is None:
        wage_coef = (sigma + phi) / (1.0 + phi)
    dg = fiscal_shock_path(ss, T, rho_g)
    Jw = J['w'][:T, :T]
    Jg = J['g'][:T, :T]
    K = Jw * wage_coef
    A = np.eye(T) - K
    rhs = (np.eye(T) + Jg) @ dg
    dY = np.linalg.solve(A, rhs)
    dC = dY - dg
    mu = dY[0] / dg[0]
    return dY, dC, mu


# -----------------------------------------------------------------------------
# Full NKPC + Taylor closure
# -----------------------------------------------------------------------------
def compute_impact_multiplier_taylor(
        J: dict, ss: SteadyState, rho_g: float = 0.80,
        T: int = 20, sigma: float = 2.0, phi: float = 1.0,
        phi_pi: float = 1.5, phi_y: float = 0.125,
        kappa: float = 0.024, beta: float = 0.99,
        wage_coef: float | None = None,
        ) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """Full GE with NKPC + Taylor + Fisher.

    Returns (dY, dC, mu_impact, paths) where paths is a dict with
    'dw', 'dpi', 'dr', 'di' for IRF plotting.
    """
    dg = fiscal_shock_path(ss, T, rho_g)

    Jw = J['w'][:T, :T]
    Jr = J['r'][:T, :T]
    Jg = J['g'][:T, :T]

    if wage_coef is None:
        wage_coef = (sigma + phi) / (1.0 + phi)
    M_wY = wage_coef * np.eye(T)
    Mr = M_r(T, beta, kappa, phi_pi)

    # dw = M_wY dY;  dr = Mr dw + phi_y dY = Mr M_wY dY + phi_y dY
    M_rY = Mr @ M_wY + phi_y * np.eye(T)

    # dC = Jw dw + Jr dr + Jg dg
    #    = Jw M_wY dY + Jr M_rY dY + Jg dg
    #    = K dY + Jg dg
    K = Jw @ M_wY + Jr @ M_rY

    # dY = dC + dg = K dY + (I + Jg) dg
    A = np.eye(T) - K
    rhs = (np.eye(T) + Jg) @ dg
    dY = np.linalg.solve(A, rhs)

    dw = M_wY @ dY
    dpi = M_pi(T, beta, kappa) @ dw
    di = phi_pi * dpi + phi_y * dY
    dr = M_rY @ dY
    dC = dY - dg

    mu = dY[0] / dg[0]
    return dY, dC, mu, {
        'dw': dw, 'dpi': dpi, 'di': di, 'dr': dr, 'dg': dg,
    }


# -----------------------------------------------------------------------------
# Decomposition: direct vs indirect channel
# -----------------------------------------------------------------------------
def decompose_multiplier(
        J: dict, ss: SteadyState, rho_g: float = 0.80, T: int = 20,
        sigma: float = 2.0, phi: float = 1.0,
        closure: str = 'taylor',
        phi_pi: float = 1.5, phi_y: float = 0.125,
        kappa: float = 0.024, beta: float = 0.99,
) -> dict:
    """
    Decompose the impact multiplier into direct and indirect channels.

    Direct channel = J^{Cg} contribution at t=0 (the xi channel in the paper).
    Indirect channel = general-equilibrium feedback through w and r.
    """
    if closure == 'fixed_rate':
        dY, dC, mu = compute_impact_multiplier_fixed_rate(
            J, ss, rho_g=rho_g, T=T, sigma=sigma, phi=phi,
        )
        paths = {}
    elif closure == 'taylor':
        dY, dC, mu, paths = compute_impact_multiplier_taylor(
            J, ss, rho_g=rho_g, T=T, sigma=sigma, phi=phi,
            phi_pi=phi_pi, phi_y=phi_y, kappa=kappa, beta=beta,
        )
    else:
        raise ValueError(f"Unknown closure: {closure}")

    dg = fiscal_shock_path(ss, T, rho_g)
    # Direct channel: J^Cg at t=0 (the paper's xi channel)
    direct = (J['g'][0, :T] @ dg) / dg[0]
    indirect = (mu - 1.0) - direct

    return {
        'mu_impact': mu,
        'direct': direct,
        'indirect': indirect,
        'dY': dY, 'dC': dC,
        'paths': paths,
    }


if __name__ == "__main__":
    from steady_state import solve_steady_state
    from jacobians import compute_jacobian_truncated

    print("[Fixed-rate vs Taylor-rule closure]")
    for name, tS, tH in [
        ('Bilbiie separable',     0.00,  0.00),
        ('Symmetric complement', -0.50, -0.50),
        ('Heterogeneous',         0.29, -0.76),
    ]:
        ss = solve_steady_state(theta_S=tS, theta_H=tH, n_a=60, n_y=5)
        J = compute_jacobian_truncated(ss, T=10)
        fixed = decompose_multiplier(J, ss, T=10, closure='fixed_rate')
        taylor = decompose_multiplier(J, ss, T=10, closure='taylor')
        print(f"  {name:22s}  mu_fixed={fixed['mu_impact']:.3f}  "
              f"mu_taylor={taylor['mu_impact']:.3f}")
