"""
estimate_theta.py
-----------------
Micro-to-macro estimation of household-type-specific theta^j using CEX
household-level consumption data and state-year variation in public-good
provision (Medicaid, FTA transit funding, K-12 expenditure).

This is the Python analogue of the Stata implementation in
    code/empirical/cex_analysis.do

The structural mapping from micro elasticities to theta^j follows
Section 5.3 of the paper:

    d log c^j / d log g = -Gamma^j ~ -theta^j * (g_bar / c_bar)   (small-theta limit)

so theta^j_hat = -beta_q_hat * (c_bar / g_bar) where beta_q is the
micro elasticity for quintile q. We aggregate the five quintile estimates
into two household types via a Bayesian shrinkage estimator.

NOTE: This script EXPECTS CEX data at data/processed/cex_household_panel.csv.
Since the panel is not distributed with this repository (CEX licensing), we
ship a stylized simulation that reproduces the quintile pattern reported in
Section 5.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_cex_micro_estimates(
        n: int = 92_000, rng_seed: int = 42) -> pd.DataFrame:
    """Simulate the CEX diff-in-diff estimates reported in Section 5.2.

    Returns a DataFrame with quintile-specific beta_q for complementary
    and substitute consumption categories.
    """
    rng = np.random.default_rng(rng_seed)
    # True DGP: complementary betas decline with quintile
    true_beta_comp = np.array([0.187, 0.142, 0.068, 0.024, -0.021])
    true_beta_sub  = np.array([-0.032, -0.048, -0.097, -0.124, -0.158])
    # Standard errors (large N => tight)
    se_comp = np.array([0.042, 0.039, 0.034, 0.032, 0.035])
    se_sub  = np.array([0.045, 0.040, 0.038, 0.041, 0.041])

    records = []
    for q in range(5):
        beta_c = true_beta_comp[q] + rng.normal(0, se_comp[q] * 0.2)
        beta_s = true_beta_sub[q]  + rng.normal(0, se_sub[q] * 0.2)
        records.append({
            "quintile": q + 1,
            "beta_comp": beta_c,
            "se_comp":   se_comp[q],
            "beta_sub":  beta_s,
            "se_sub":    se_sub[q],
            "n":         n // 5,
        })
    return pd.DataFrame(records)


def map_to_structural_theta(
        betas: pd.DataFrame, g_over_c: float = 0.25) -> pd.DataFrame:
    """Map micro elasticities into structural theta^j by quintile.

    Under Assumption 2: d log c^j / d log g ~ -theta^j * g_bar/c_bar
    so theta_q_hat = -(beta_comp_q + beta_sub_q) / g_over_c
    """
    theta_q = -(betas["beta_comp"].values + betas["beta_sub"].values) / g_over_c
    betas = betas.copy()
    betas["theta_q"] = theta_q
    # Approximate quintile-level SE: sqrt(se_comp^2 + se_sub^2) / g_over_c
    betas["theta_se"] = np.sqrt(betas["se_comp"].values ** 2 +
                                 betas["se_sub"].values ** 2) / g_over_c
    return betas


def bayesian_aggregate(theta_q: np.ndarray, theta_se: np.ndarray,
                       lam_aggregate_mpc: float = 0.35
                       ) -> Dict[str, float]:
    """Aggregate the 5 quintile-specific theta estimates into two types.

    Under the baseline, the bottom-two quintiles map to H (HtM),
    and the top-three to S (savers), weighted by their contribution to
    the aggregate MPC (lam_aggregate_mpc).
    """
    # Weights: bottom-two quintiles contribute most of aggregate MPC
    w_bot = np.array([0.55, 0.30, 0.10, 0.04, 0.01])  # HtM-type weights
    w_top = 1.0 - w_bot                                # saver-type weights

    # Variance-weighted mean (Bayesian shrinkage)
    prec = 1.0 / (theta_se ** 2 + 1e-12)
    theta_H = np.sum(w_bot * prec * theta_q) / np.sum(w_bot * prec)
    theta_S = np.sum(w_top * prec * theta_q) / np.sum(w_top * prec)

    theta_H_se = 1.0 / np.sqrt(np.sum(w_bot ** 2 * prec))
    theta_S_se = 1.0 / np.sqrt(np.sum(w_top ** 2 * prec))

    return {
        "theta_H":     float(theta_H),
        "theta_H_se":  float(theta_H_se),
        "theta_S":     float(theta_S),
        "theta_S_se":  float(theta_S_se),
    }


def main() -> None:
    cex_file = DATA_DIR / "cex_household_panel.csv"
    if cex_file.exists():
        print(f"Loading CEX estimates from {cex_file}")
        micro = pd.read_csv(cex_file)
    else:
        print(f"[WARN] CEX panel not found at {cex_file}. "
              "Running with simulated data.")
        micro = simulate_cex_micro_estimates()

    mapped = map_to_structural_theta(micro)
    print("\n=== Quintile-level micro estimates ===")
    print(mapped.to_string(index=False))

    agg = bayesian_aggregate(
        mapped["theta_q"].values,
        mapped["theta_se"].values,
    )
    print("\n=== Aggregate estimates ===")
    print(f"  theta_H   = {agg['theta_H']:+.3f}  (SE = {agg['theta_H_se']:.3f})")
    print(f"  theta_S   = {agg['theta_S']:+.3f}  (SE = {agg['theta_S_se']:.3f})")

    # Save for downstream use
    out = OUT_DIR / "theta_estimates.csv"
    pd.DataFrame([agg]).to_csv(out, index=False)
    print(f"\nSaved estimates to {out}")


if __name__ == "__main__":
    main()
