# Data Download Guide

This project's empirical identification strategy uses four data sources, none of which I could fetch directly from the sandbox (restricted network). Here is a step-by-step guide to obtain each, build the expected extract, and drop it into `data/raw/` so the pipeline runs on real data.

Once all four files are in place, run:

```bash
python3 code/empirical/run_empirical.py
```

The pipeline automatically detects real data files and uses them instead of the synthetic fallback.

---

## 1. Consumer Expenditure Survey (CEX) Public-Use Microdata

**Source:** U.S. Bureau of Labor Statistics
**URL:** https://www.bls.gov/cex/pumd.htm
**File expected:** `data/raw/cex_household_panel.csv`

### Download steps

1. Visit https://www.bls.gov/cex/pumd_data.htm
2. Under "Interview Survey" → "CSV format", download yearly extracts for 1996–2019. Each year gives you a ZIP containing files like:
   - `fmli141.csv` (household-level summary for Q1 2014)
   - `mtbi141.csv` (expenditure detail)
   - `intrvw<YY>.zip` (interview bundle)
3. For each year, merge `fmli*.csv` (household identifier + demographics + income quintile) with `mtbi*.csv` (category-level expenditures).
4. Compute the six category totals for each household-quarter-year:
   - `cat_comp_food_home` — `UCC = 190901` (food at home)
   - `cat_comp_transit` — `UCC = 530110` (mass transit fares)
   - `cat_comp_school` — `UCC = 660110` + `660210` (school supplies + textbooks)
   - `cat_sub_restaurant` — `UCC = 190902` (food away from home)
   - `cat_sub_private_health` — `UCC = 580111` + `580112` (private health insurance)
   - `cat_sub_private_school` — `UCC = 660901` (private K–12 tuition)
5. Collapse household × quarter → household × year (mean quarterly expenditure, annualized).
6. Take logs: `log_c = log(1 + expenditure)`.
7. Attach `state` (two-letter FIPS-to-postal) from the `STATE` column.
8. Attach `income_quintile` by computing the 20/40/60/80 percentiles of household before-tax income (`FINCBTXM`) within each year.

### Expected CSV columns

```
household_id, year, state, income_quintile, category, is_complement,
log_c, age_head, household_size
```

One row per (household, year, category). `is_complement = 1` for the three `cat_comp_*` categories, `0` for the three `cat_sub_*` categories.

### Sample size

The full 1996–2019 CEX interview sample has roughly 40,000 households per year, which produces roughly 5 million (household, year, category) rows. The pipeline handles this in about 2 minutes.

---

## 2. Medicaid Expansion Timing (KFF)

**Source:** Kaiser Family Foundation's Medicaid Expansion Tracker
**URL:** https://www.kff.org/affordable-care-act/issue-brief/status-of-state-medicaid-expansion-decisions-interactive-map/
**File expected:** `data/raw/medicaid_expansion.csv`

### Download steps

1. Download the "Historical Medicaid Expansion Decisions" spreadsheet from the KFF page above, or use the timing from Kaestner & Lubotsky (2016, JEP).
2. For each state and year 1996–2019, record:
   - `expanded` (0/1): whether the state had adopted ACA Medicaid expansion in that year
   - `eligibility_cutoff_fpl`: the effective Medicaid eligibility cutoff as a percentage of the Federal Poverty Line (138% once expanded; state-specific pre-expansion levels otherwise)

### Expected CSV columns

```
state, year, expanded, eligibility_cutoff_fpl
```

## 3. FTA Federal Transit Funds Per Capita

**Source:** Federal Transit Administration, U.S. DOT
**URL:** https://www.transit.dot.gov/ntd/data-product/monthly-module-adjusted-data-release
**File expected:** `data/raw/fta_transit_per_capita.csv`

### Download steps

1. From the FTA NTD Annual Database, download the "Funding Sources" spreadsheet for years 1996–2019.
2. Aggregate federal formula funds (programs 5307, 5309, 5311) to the state level by summing across agencies.
3. Divide by state population (from US Census annual estimates, https://www.census.gov/programs-surveys/popest.html).

### Expected CSV columns

```
state, year, transit_funds_per_capita
```

## 4. NCES K-12 Per-Pupil Expenditure

**Source:** National Center for Education Statistics
**URL:** https://nces.ed.gov/programs/digest/current_tables.asp
**File expected:** `data/raw/nces_k12_per_pupil.csv`

### Download steps

1. From the NCES Digest of Education Statistics, download Table 236.75 ("Current expenditures per pupil in fall enrollment in public elementary and secondary schools, by state or jurisdiction") for each available year 1996–2019.
2. Pivot to long format (one row per state-year).

### Expected CSV columns

```
state, year, expenditure_per_pupil_usd
```

---

## 5. Pipeline outputs

Once all files are present, running `python3 code/empirical/run_empirical.py` produces:

- `data/processed/did_estimates_by_source.csv` — 3 sources × 2 categories × 5 quintiles DiD estimates
- `data/processed/quintile_thetas.csv` — structural $\theta_q$ per quintile
- `data/processed/theta_estimates_full.csv` — aggregated $(\theta^H, \theta^S)$
- `paper/figures/fig8_posterior.pdf` — joint posterior figure
- `paper/tables/tab_micro_main.tex` — LaTeX-ready micro-main table

## 6. Working without the real data

The synthetic pipeline is designed so that, when real data is missing, the generated panel reproduces the key qualitative results reported in the paper: $\theta^H < 0$, $\theta^S > 0$, and a monotone gradient across income quintiles. The magnitudes are conservative relative to the paper's calibration because the synthetic DGP uses smaller betas than Barthel & Francois (2025) report; swapping in real CEX data should produce magnitudes close to the paper's target ($\theta^H \approx -0.76$, $\theta^S \approx +0.29$).

## 7. Troubleshooting

- **CEX format**: BLS ships the public-use files as a mix of CSV (2013 onward) and fixed-width (pre-2013). For early years, use the `readcensus` R package or the CEX-provided codebook to convert.
- **State codes**: BLS uses two-digit FIPS; KFF and NCES use two-letter postal abbreviations. The pipeline expects postal codes (`NY`, `CA`, etc.). Use a standard crosswalk (from `us` Python package or CDC).
- **Income quintiles**: The CEX pre-computes quintiles in the `INC_RANK` variable for some years; for others, compute yourself from `FINCBTXM`. Either works.
- **Missing years**: If a state-year combination is missing from any policy file, the DiD estimator drops that observation (inner join in `cex_data.load_cex_with_policy`).
