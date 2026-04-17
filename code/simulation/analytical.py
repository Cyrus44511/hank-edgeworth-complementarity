"""
analytical.py
-------------
Closed-form sufficient statistics and multiplier expressions from
Section 3 of the paper.

This module implements:
  - alpha_j, Gamma_j, psi_j  : steady-state utility objects (Assumption 2)
  - K, D                      : derived structural constants
  - chi(theta_S, theta_H)     : income cyclicality (Proposition 1)
  - xi(theta_S, theta_H)      : direct fiscal channel (Proposition 2)
  - multiplier(theta_S, theta_H) : impact fiscal multiplier (Proposition 3)
  - dominance(theta_S, theta_H)  : distributional dominance test (Corollary 1)

All equations are referenced to the paper by number in comments.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Calibration container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    """Baseline parameter set (quarterly). See Table 1 in the paper."""
    beta:   float = 0.99
    sigma:  float = 2.0
    phi:    float = 1.0
    lam:    float = 0.35          # HtM share (lambda in paper)
    s:      float = 0.94          # saver-to-saver persistence
    mu:     float = 0.25          # tax progressivity
    tau_D:  float = 0.0           # profit redistribution
    g_over_y: float = 0.20
    rho_g:  float = 0.80

    @property
    def g_over_c(self) -> float:
        return self.g_over_y / (1.0 - self.g_over_y)    # = 0.25 at g/y=0.20


BASELINE = Calibration()


# -----------------------------------------------------------------------------
# Steady-state utility objects
# -----------------------------------------------------------------------------
def alpha_j(theta_j: float, cal: Calibration = BASELINE) -> float:
    """alpha^j = c / (c + theta^j * g)."""
    return 1.0 / (1.0 + theta_j * cal.g_over_c)


def Gamma_j(theta_j: float, cal: Calibration = BASELINE) -> float:
    """Gamma^j = theta^j * g / (c + theta^j * g) = 1 - alpha^j."""
    return 1.0 - alpha_j(theta_j, cal)


def psi_j(theta_j: float, cal: Calibration = BASELINE) -> float:
    """psi^j = theta^j * g / c."""
    return theta_j * cal.g_over_c


# -----------------------------------------------------------------------------
# Derived structural objects
# -----------------------------------------------------------------------------
def K(cal: Calibration = BASELINE) -> float:
    """K = phi + 1 - phi * tau_D / lambda (appears in HtM budget)."""
    return cal.phi + 1.0 - cal.phi * cal.tau_D / cal.lam


def D(theta_S: float, theta_H: float, cal: Calibration = BASELINE) -> float:
    """D = (phi + sigma*alpha^H) - K*sigma*lambda*(alpha^H - alpha^S)."""
    aS = alpha_j(theta_S, cal)
    aH = alpha_j(theta_H, cal)
    return (cal.phi + cal.sigma * aH) - K(cal) * cal.sigma * cal.lam * (aH - aS)


def zeta(cal: Calibration = BASELINE) -> float:
    """Bilbiie's transfer elasticity under separability."""
    return cal.phi / (cal.phi + cal.sigma)


def zeta_alpha(theta: float, cal: Calibration = BASELINE) -> float:
    """Modified transfer elasticity under symmetric theta."""
    a = alpha_j(theta, cal)
    return cal.phi / (cal.phi + cal.sigma * a)


# -----------------------------------------------------------------------------
# Sufficient statistics
# -----------------------------------------------------------------------------
def chi(theta_S: float, theta_H: float, cal: Calibration = BASELINE) -> float:
    """
    Income cyclicality (Proposition 1, eq. 21/22).

    chi = K (phi + sigma alpha^S) / D(theta_S, theta_H)

    chi depends only on epsilon^j_cc = sigma*alpha^j, not on epsilon^j_cg.
    """
    aS = alpha_j(theta_S, cal)
    return K(cal) * (cal.phi + cal.sigma * aS) / D(theta_S, theta_H, cal)


def xi(theta_S: float, theta_H: float, cal: Calibration = BASELINE) -> float:
    """
    Direct fiscal channel (Proposition 2, eq. 23/24).

    xi depends on both epsilon^S_cg = sigma*Gamma^S and epsilon^H_cg = sigma*Gamma^H.
    """
    GS = Gamma_j(theta_S, cal)
    GH = Gamma_j(theta_H, cal)
    num = (K(cal) * (cal.phi + cal.sigma * cal.lam * GH +
                     cal.sigma * (1.0 - cal.lam) * GS)
           - cal.sigma * GH - cal.mu * cal.phi / cal.lam)
    return num / D(theta_S, theta_H, cal)


def multiplier(theta_S: float, theta_H: float,
               cal: Calibration = BASELINE) -> float:
    """
    Impact fiscal multiplier (Proposition 3, eq. 1/28).

    mu = 1 + lambda * xi / (1 - lambda * chi).
    """
    ch = chi(theta_S, theta_H, cal)
    xi_val = xi(theta_S, theta_H, cal)
    denom = 1.0 - cal.lam * ch
    if abs(denom) < 1e-10:
        return np.inf
    return 1.0 + cal.lam * xi_val / denom


# -----------------------------------------------------------------------------
# Compounding coefficient (dynamic THANK)
# -----------------------------------------------------------------------------
def delta(theta_S: float, theta_H: float, cal: Calibration = BASELINE) -> float:
    """
    THANK precautionary-saving compounding coefficient.
    delta = 1 + (chi - 1) * (1 - s) / (1 - lambda*chi).
    """
    ch = chi(theta_S, theta_H, cal)
    denom = 1.0 - cal.lam * ch
    if abs(denom) < 1e-10:
        return np.inf
    return 1.0 + (ch - 1.0) * (1.0 - cal.s) / denom


# -----------------------------------------------------------------------------
# Distributional dominance condition (Corollary 1, eq. 30)
# -----------------------------------------------------------------------------
def dominance(theta_S: float, theta_H: float,
              cal: Calibration = BASELINE) -> float:
    """
    Distributional dominance test.
    Returns: (K*lambda - 1) * epsilon^H_cg + K * (1 - lambda) * epsilon^S_cg.
    Positive => multiplier exceeds separable benchmark.
    """
    eps_H_cg = cal.sigma * Gamma_j(theta_H, cal)
    eps_S_cg = cal.sigma * Gamma_j(theta_S, cal)
    return (K(cal) * cal.lam - 1) * eps_H_cg + K(cal) * (1 - cal.lam) * eps_S_cg


# -----------------------------------------------------------------------------
# Special cases: sanity checks
# -----------------------------------------------------------------------------
def bilbiie_benchmark(cal: Calibration = BASELINE) -> Tuple[float, float, float]:
    """Corollary 2: Bilbiie separable benchmark."""
    return chi(0.0, 0.0, cal), xi(0.0, 0.0, cal), multiplier(0.0, 0.0, cal)


def rank_limit(theta_S: float, cal: Calibration = BASELINE) -> float:
    """Corollary 3: RANK limit (lambda -> 0). mu = 1 - psi^S."""
    return 1.0 - psi_j(theta_S, cal)


# -----------------------------------------------------------------------------
# Dispatching routine
# -----------------------------------------------------------------------------
def summary_table(theta_S: float, theta_H: float,
                  cal: Calibration = BASELINE) -> dict:
    """Compute all sufficient statistics for a given (theta_S, theta_H)."""
    return {
        "alpha_S":    alpha_j(theta_S, cal),
        "alpha_H":    alpha_j(theta_H, cal),
        "Gamma_S":    Gamma_j(theta_S, cal),
        "Gamma_H":    Gamma_j(theta_H, cal),
        "K":          K(cal),
        "D":          D(theta_S, theta_H, cal),
        "chi":        chi(theta_S, theta_H, cal),
        "xi":         xi(theta_S, theta_H, cal),
        "delta":      delta(theta_S, theta_H, cal),
        "multiplier": multiplier(theta_S, theta_H, cal),
        "dominance":  dominance(theta_S, theta_H, cal),
    }


if __name__ == "__main__":
    print("Bilbiie separable benchmark (theta=0):")
    chi0, xi0, mu0 = bilbiie_benchmark()
    print(f"  chi = {chi0:.4f}, xi = {xi0:.4f}, mu = {mu0:.4f}")

    print("\nSymmetric complementarity (theta = -0.5):")
    print(f"  mu = {multiplier(-0.5, -0.5):.4f}")

    print("\nHeterogeneous internalization (theta_S=+0.29, theta_H=-0.76):")
    print(f"  summary = {summary_table(0.29, -0.76)}")
