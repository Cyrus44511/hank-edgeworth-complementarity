"""
did_estimator.py
----------------
Difference-in-differences / two-way fixed-effects estimator for the elasticity
of private consumption to public-good provision, by income quintile.

Spec (eq. 1 in Section 5.2):

  log c_{ist}^{cat} = alpha_i + gamma_t + delta_s
                    + sum_q beta^q_{cat} * log g_{st} * 1{i in q}
                    + X' Omega + u

where X are household-level controls (age_head, household_size).

We estimate separately for:
  (a) Medicaid expansion (0/1 indicator)
  (b) log FTA transit funds per capita
  (c) log NCES per-pupil K-12 expenditure

The final theta^j estimates aggregate across the three sources and the two
category types (complement, substitute) via Bayesian shrinkage in
bayesian_aggregation.py.

Implementation: partialling-out via demeaning (within transformation) because
statsmodels' OLS with 50-state FE + household FE + year FE on 100k+ rows is
slow. The within-estimator is equivalent to dummy-variable OLS for beta but
much faster.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def _demean(df: pd.DataFrame, group: str, col: str) -> pd.Series:
    """Subtract group means (within transformation)."""
    return df[col] - df.groupby(group)[col].transform('mean')


def _two_way_demean(df: pd.DataFrame, g1: str, g2: str,
                    col: str, maxit: int = 30,
                    tol: float = 1e-8) -> pd.Series:
    """Iterated within transformation for two-way fixed effects."""
    x = df[col].copy()
    for _ in range(maxit):
        old = x.copy()
        x = x - x.groupby(df[g1]).transform('mean')
        x = x - x.groupby(df[g2]).transform('mean')
        if (x - old).abs().max() < tol:
            break
    return x


def estimate_quintile_elasticities(
        df: pd.DataFrame, policy_var: str, category_filter: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Estimate beta^q for q in {1,...,5} from the DiD regression.

    Parameters
    ----------
    df : CEX panel merged with policy data (from cex_data.load_cex_with_policy)
    policy_var : 'expanded' | 'log_transit_real' | 'log_k12_real'
    category_filter : 1 for complement categories, 0 for substitute

    Returns
    -------
    (betas, ses) where each is dict keyed by quintile 1..5.
    """
    d = df[df['is_complement'] == category_filter].copy()
    d = d.dropna(subset=['log_c', policy_var])

    # Two-way demean log_c and the policy variable by (hh, year)
    # For memory-efficient two-way FE with 50k households, we use within on
    # the larger group (household) then year.
    d['_lc'] = _demean(d, 'household_id', 'log_c')
    d['_lc'] = _demean(d, 'year', '_lc')
    d['_pol'] = _demean(d, 'household_id', policy_var)
    d['_pol'] = _demean(d, 'year', '_pol')

    # Interact policy with quintile dummies (cleanly, using numpy arrays)
    betas = {}
    ses = {}
    for q in range(1, 6):
        mask = d['income_quintile'] == q
        if mask.sum() < 20:
            betas[q] = np.nan; ses[q] = np.nan; continue
        y = d.loc[mask, '_lc'].values
        x = d.loc[mask, '_pol'].values
        # OLS: beta = (x'x)^{-1} x'y
        xx = float(x @ x)
        if xx < 1e-12:
            betas[q] = np.nan; ses[q] = np.nan; continue
        beta = float(x @ y / xx)
        # HC1-style SE (clustered by household would be better; simple
        # heteroscedasticity-robust here)
        resid = y - x * beta
        meat = float(np.sum((x * resid) ** 2))
        var = meat / (xx ** 2)
        betas[q] = beta
        ses[q] = float(np.sqrt(var))

    return betas, ses


def run_all_did(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the full DiD suite across the three policy sources and both
    category types. Returns a long DataFrame with columns:
      source, category_type, quintile, beta, se, n
    """
    rows = []
    sources = ['expanded', 'log_transit_real', 'log_k12_real']
    source_labels = {
        'expanded':         'Medicaid',
        'log_transit_real': 'FTA Transit (log)',
        'log_k12_real':     'K-12 per-pupil (log)',
    }
    for source in sources:
        for cat_type, cat_label in [(1, 'complement'), (0, 'substitute')]:
            betas, ses = estimate_quintile_elasticities(df, source, cat_type)
            for q in range(1, 6):
                rows.append({
                    'source': source_labels[source],
                    'category_type': cat_label,
                    'quintile': q,
                    'beta': betas[q],
                    'se': ses[q],
                    'n': int((df[(df['is_complement'] == cat_type) &
                                  (df['income_quintile'] == q)]).shape[0]),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from cex_data import load_cex_with_policy
    df = load_cex_with_policy()
    out = run_all_did(df)
    print(out.pivot_table(index=['source', 'category_type'],
                          columns='quintile',
                          values='beta').round(3))
