"""
cex_data.py
-----------
Loader for the Consumer Expenditure Survey (CEX) household panel.

Real data layout expected at data/raw/cex_household_panel.csv with columns:
    household_id, year, state, income_quintile, weight,
    cat_comp_food_home, cat_comp_transit, cat_comp_school,
    cat_sub_restaurant, cat_sub_private_health, cat_sub_private_school,
    age_head, household_size, ...

If the file is not present, a vectorized synthetic generator produces a panel
whose quintile-by-policy-source betas match the patterns reported in
Barthel & Francois (2025) and Francois & Keinsley (2023).

See docs/data_download_guide.md for how to build the real extract.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from policy_data import US_STATES, YEARS, load_all_policies

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


COMP_CATEGORIES = [
    'cat_comp_food_home',
    'cat_comp_transit',
    'cat_comp_school',
]
SUB_CATEGORIES = [
    'cat_sub_restaurant',
    'cat_sub_private_health',
    'cat_sub_private_school',
]
ALL_CATS = COMP_CATEGORIES + SUB_CATEGORIES


# True quintile-specific betas for the synthetic DGP
_TRUE_BETAS = {
    'comp': {
        'transit':  {1: +0.23, 2: +0.17, 3: +0.08, 4: +0.03, 5: -0.03},
        'k12':      {1: +0.19, 2: +0.15, 3: +0.07, 4: +0.02, 5: -0.02},
        'medicaid': {1: +0.15, 2: +0.12, 3: +0.05, 4: +0.02, 5: -0.02},
    },
    'sub': {
        'transit':  {1: -0.02, 2: -0.04, 3: -0.09, 4: -0.12, 5: -0.16},
        'k12':      {1: -0.03, 2: -0.05, 3: -0.10, 4: -0.13, 5: -0.17},
        'medicaid': {1: -0.02, 2: -0.04, 3: -0.08, 4: -0.10, 5: -0.13},
    },
}


def _synthetic_cex(n_households: int = 15_000,
                   avg_years_per_hh: int = 2,
                   rng_seed: int = 101) -> pd.DataFrame:
    """Vectorized synthetic CEX generator.

    For speed, we build the full panel as numpy arrays and convert to
    DataFrame only at the end.
    """
    rng = np.random.default_rng(rng_seed)
    policies = load_all_policies()

    # State-year lookup for policy variables
    pol_idx = policies.set_index(['state', 'year'])
    g_transit_all = pol_idx['log_transit_real'].values - 4.0     # centered
    g_k12_all     = pol_idx['log_k12_real'].values - 9.3         # centered
    g_medic_all   = pol_idx['expanded'].values.astype(float)

    # Map (state, year) -> index into pol_idx
    sy_tuples = list(pol_idx.index)
    n_sy = len(sy_tuples)

    # Assign households uniformly to states
    hh_state = rng.integers(0, len(US_STATES), size=n_households)
    hh_quintile = np.tile(np.arange(1, 6), n_households // 5 + 1)[:n_households]
    rng.shuffle(hh_quintile)
    hh_fe = rng.normal(0, 0.3, size=n_households)

    # State FE
    state_fe = rng.normal(0, 0.1, size=len(US_STATES))

    # For each household, pick its observed years
    n_cats = len(ALL_CATS)
    rows = []
    for hh in range(n_households):
        q = int(hh_quintile[hh])
        s_idx = int(hh_state[hh])
        s = US_STATES[s_idx]
        # Number of years this hh is in the panel
        n_years = max(1, int(rng.integers(1, avg_years_per_hh * 2 + 1)))
        yrs = rng.choice(YEARS, size=min(n_years, len(YEARS)), replace=False)
        for y in yrs:
            try:
                g_t = pol_idx.loc[(s, y), 'log_transit_real'] - 4.0
                g_k = pol_idx.loc[(s, y), 'log_k12_real'] - 9.3
                g_m = float(pol_idx.loc[(s, y), 'expanded'])
            except KeyError:
                continue
            mean_lc = {1:8.5,2:9.1,3:9.6,4:10.1,5:10.8}[q] + hh_fe[hh] + state_fe[s_idx]
            # Complement categories
            for cat in COMP_CATEGORIES:
                signal = (_TRUE_BETAS['comp']['transit'][q] * g_t +
                           _TRUE_BETAS['comp']['k12'][q]     * g_k +
                           _TRUE_BETAS['comp']['medicaid'][q]* g_m)
                noise = rng.normal(0, 0.22)
                rows.append((hh, y, s, q, cat, 1,
                              mean_lc - 2.3 + signal + noise,
                              45, 3))
            for cat in SUB_CATEGORIES:
                signal = (_TRUE_BETAS['sub']['transit'][q] * g_t +
                           _TRUE_BETAS['sub']['k12'][q]     * g_k +
                           _TRUE_BETAS['sub']['medicaid'][q]* g_m)
                noise = rng.normal(0, 0.24)
                rows.append((hh, y, s, q, cat, 0,
                              mean_lc - 3.0 + signal + noise,
                              45, 3))

    cols = ['household_id', 'year', 'state', 'income_quintile',
            'category', 'is_complement', 'log_c', 'age_head', 'household_size']
    df = pd.DataFrame(rows, columns=cols)
    return df


def load_cex(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "cex_household_panel.csv")
    if path.exists():
        print(f"  [real] Loading CEX from {path}")
        return pd.read_csv(path)
    print(f"  [synthetic] CEX panel not found; generating synthetic data...")
    df = _synthetic_cex()
    synth_out = PROCESSED_DIR / "cex_panel_synthetic.csv"
    df.to_csv(synth_out, index=False)
    print(f"  Saved synthetic panel ({len(df):,} rows) to {synth_out}")
    return df


def load_cex_with_policy() -> pd.DataFrame:
    cex = load_cex()
    pol = load_all_policies()
    df = cex.merge(pol, on=['state', 'year'], how='left')
    return df


if __name__ == "__main__":
    df = load_cex_with_policy()
    print(f"\nCEX panel: {len(df):,} rows, "
          f"{df['household_id'].nunique():,} households.")
    print(df.groupby(['income_quintile', 'is_complement'])['log_c']
            .agg(['mean', 'std', 'count']).round(3))
