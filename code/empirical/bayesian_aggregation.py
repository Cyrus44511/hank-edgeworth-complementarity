"""
bayesian_aggregation.py
-----------------------
Aggregate quintile-specific DiD estimates from three policy sources into
household-type-specific theta^j (theta_H for HtM, theta_S for savers) via
Bayesian shrinkage, as described in Section 5.3 of the paper.

Mapping from micro elasticity to structural theta under Assumption 2:
    d log c^j / d log g  ~=  -theta^j * (g_bar / c_bar)

So: theta_q = -(beta_comp_q + beta_sub_q) / (g_bar/c_bar)

Aggregation to types:
  Bottom two quintiles -> HtM (weights 0.55, 0.30, 0.10, 0.04, 0.01)
  Top three quintiles  -> savers (complementary weights)

Bayesian update uses inverse-variance weighting with a weakly informative
prior centered at zero, sigma_prior = 1.0.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


# -----------------------------------------------------------------------------
# Structural mapping
# -----------------------------------------------------------------------------
G_OVER_C = 0.25            # steady-state g_bar / c_bar from the paper


def betas_to_theta_q(beta_comp: float, beta_sub: float,
                     g_over_c: float = G_OVER_C) -> float:
    """Map quintile-level (beta_comp, beta_sub) to structural theta_q.

    Under linear effective consumption (Assumption 2 in the paper):
        d log c^j / d log g = -Gamma^j ~= -theta^j * (g_bar/c_bar)

    For the AGGREGATE household elasticity of PRIVATE consumption with
    respect to public spending, we average across complement and substitute
    categories (they're two components of c^j):

        avg_elasticity = 0.5 * (beta_comp + beta_sub)

    Then:  theta_q = -avg_elasticity / g_over_c.
    For HtM (low q): beta_comp > 0 dominates, avg > 0, so theta_q < 0 (complement).
    For savers (high q): beta_sub < 0 dominates, avg < 0, so theta_q > 0 (substitute).
    """
    avg_elasticity = 0.5 * (beta_comp + beta_sub)
    return -avg_elasticity / g_over_c


def aggregate_quintile_thetas(did_out: pd.DataFrame,
                              g_over_c: float = G_OVER_C) -> pd.DataFrame:
    """From the three-source DiD output, compute a single theta_q per
    quintile by averaging across the three policy sources."""
    # Pivot so we have beta_comp, se_comp, beta_sub, se_sub per (source, quintile)
    rows = []
    for q in range(1, 6):
        thetas_by_source = []
        vars_by_source = []
        for source in did_out['source'].unique():
            bc = did_out[(did_out['source'] == source) &
                         (did_out['category_type'] == 'complement') &
                         (did_out['quintile'] == q)]
            bs = did_out[(did_out['source'] == source) &
                         (did_out['category_type'] == 'substitute') &
                         (did_out['quintile'] == q)]
            if len(bc) == 0 or len(bs) == 0: continue
            theta = betas_to_theta_q(float(bc['beta'].iloc[0]),
                                      float(bs['beta'].iloc[0]),
                                      g_over_c)
            # Variance of (bc + bs)/2 = 0.25 * (var_c + var_s)
            # (assuming independence between bc and bs estimates)
            var_theta = 0.25 * (float(bc['se'].iloc[0]) ** 2 +
                                 float(bs['se'].iloc[0]) ** 2) / (g_over_c ** 2)
            thetas_by_source.append(theta)
            vars_by_source.append(var_theta)

        if not thetas_by_source:
            rows.append({'quintile': q, 'theta_q': np.nan, 'theta_se': np.nan,
                         'n_sources': 0}); continue
        # Inverse-variance-weighted mean
        prec = np.array([1.0 / (v + 1e-12) for v in vars_by_source])
        theta_q = float(np.array(thetas_by_source) @ prec / prec.sum())
        theta_se = float(1.0 / np.sqrt(prec.sum()))
        rows.append({'quintile': q, 'theta_q': theta_q, 'theta_se': theta_se,
                     'n_sources': len(thetas_by_source)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Aggregate to theta^H and theta^S
# -----------------------------------------------------------------------------
HTM_WEIGHTS    = np.array([0.55, 0.30, 0.10, 0.04, 0.01])
SAVER_WEIGHTS  = 1.0 - HTM_WEIGHTS


def bayesian_aggregate_to_types(
        quintile_df: pd.DataFrame,
        sigma_prior: float = 1.0,
        prior_mean_H: float = 0.0,
        prior_mean_S: float = 0.0,
) -> Dict[str, float]:
    """Aggregate the 5 quintile theta estimates into theta^H and theta^S
    using weighted Bayesian shrinkage."""
    theta_q = quintile_df['theta_q'].values
    var_q = quintile_df['theta_se'].values ** 2

    def _posterior(weights: np.ndarray, prior_mean: float
                   ) -> tuple[float, float]:
        # Posterior mean = weighted avg with inverse-variance weighting,
        # shrunk toward prior_mean with weight 1/sigma_prior^2
        prec_data = weights ** 2 / (var_q + 1e-12)
        prec_prior = 1.0 / sigma_prior ** 2
        num = (weights @ (theta_q * weights / (var_q + 1e-12))
               + prior_mean * prec_prior)
        den = prec_data.sum() + prec_prior
        mean = num / den
        var = 1.0 / den
        return float(mean), float(np.sqrt(var))

    theta_H, se_H = _posterior(HTM_WEIGHTS, prior_mean_H)
    theta_S, se_S = _posterior(SAVER_WEIGHTS, prior_mean_S)
    return {
        'theta_H': theta_H, 'theta_H_se': se_H,
        'theta_S': theta_S, 'theta_S_se': se_S,
    }


if __name__ == "__main__":
    from cex_data import load_cex_with_policy
    from did_estimator import run_all_did

    print("Loading CEX + policy data...")
    df = load_cex_with_policy()

    print("Running DiD estimation (three sources x 2 categories)...")
    did_out = run_all_did(df)
    print(did_out.pivot_table(index=['source', 'category_type'],
                               columns='quintile', values='beta').round(3))

    print("\nAggregating to quintile thetas...")
    qth = aggregate_quintile_thetas(did_out)
    print(qth.round(3))

    print("\nBayesian aggregation to theta^H and theta^S...")
    result = bayesian_aggregate_to_types(qth)
    print(f"  theta^H = {result['theta_H']:+.3f}  "
          f"(SE = {result['theta_H_se']:.3f})")
    print(f"  theta^S = {result['theta_S']:+.3f}  "
          f"(SE = {result['theta_S_se']:.3f})")

    # Save for downstream use
    out = PROCESSED_DIR / "theta_estimates_full.csv"
    pd.DataFrame([result]).to_csv(out, index=False)
    did_out.to_csv(PROCESSED_DIR / "did_estimates_by_source.csv", index=False)
    qth.to_csv(PROCESSED_DIR / "quintile_thetas.csv", index=False)
    print(f"\nSaved results to {PROCESSED_DIR}")
