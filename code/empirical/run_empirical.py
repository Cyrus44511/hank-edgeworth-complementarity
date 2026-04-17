"""
run_empirical.py
----------------
Master script for the empirical identification of theta^H and theta^S.

Pipeline:
  1. Load (or synthesize) CEX household panel
  2. Load state-year policy variation: Medicaid expansion, FTA transit,
     NCES K-12 per-pupil expenditure
  3. Run diff-in-diff for each (policy source) x (category type) x quintile
  4. Map quintile betas to structural theta_q
  5. Bayesian aggregate to theta^H and theta^S
  6. Generate Figure 8: joint posterior over (theta^H, theta^S)
  7. Save Tables 5 (DiD estimates) and 6 (aggregated thetas)

Run:
    python3 run_empirical.py
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cex_data import load_cex_with_policy
from did_estimator import run_all_did
from bayesian_aggregation import (
    aggregate_quintile_thetas, bayesian_aggregate_to_types,
    HTM_WEIGHTS, SAVER_WEIGHTS, G_OVER_C,
)

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "paper" / "figures"


# -----------------------------------------------------------------------------
# Figure 8: joint posterior over (theta^H, theta^S)
# -----------------------------------------------------------------------------
def plot_posterior(theta_H: float, theta_H_se: float,
                   theta_S: float, theta_S_se: float,
                   n_draws: int = 5000) -> None:
    """Sample from the posterior and generate a joint-density plot.

    Uses a bivariate normal posterior with negative correlation reflecting
    the common identifying variation across quintiles.
    """
    rng = np.random.default_rng(42)
    # Negative correlation from shared identifying variation
    corr = -0.4
    cov = np.array([
        [theta_H_se ** 2, corr * theta_H_se * theta_S_se],
        [corr * theta_H_se * theta_S_se, theta_S_se ** 2],
    ])
    draws = rng.multivariate_normal([theta_H, theta_S], cov, size=n_draws)

    fig = plt.figure(figsize=(7.5, 6))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax)

    # Scatter + kde
    ax.scatter(draws[:, 0], draws[:, 1], s=4, alpha=0.2, color='steelblue')
    ax.axvline(0, color='grey', ls='--', lw=0.7)
    ax.axhline(0, color='grey', ls='--', lw=0.7)

    # Posterior mean
    ax.scatter([theta_H], [theta_S], color='darkred', s=80,
               marker='x', zorder=3, label='Posterior mean')

    # Bilbiie separable benchmark
    ax.scatter([0.0], [0.0], color='black', s=70,
               marker='o', zorder=3, label='Bilbiie separable')

    # Paper's calibration target
    ax.scatter([-0.76], [0.29], color='darkorange', s=80,
               marker='*', zorder=3, label='Paper calibration')

    # Dashed line at theta^H = theta^S (symmetric cases)
    lo, hi = -1.2, 0.8
    ax.plot([lo, hi], [lo, hi], color='grey', lw=0.6, ls=':',
            label=r'$\theta^H = \theta^S$')

    ax.set_xlabel(r'$\theta^H$ (hand-to-mouth)')
    ax.set_ylabel(r'$\theta^S$ (savers)')
    ax.set_xlim(-1.4, 0.6)
    ax.set_ylim(-0.6, 0.8)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax_top.hist(draws[:, 0], bins=40, color='steelblue', alpha=0.6)
    ax_top.axis('off')
    ax_right.hist(draws[:, 1], bins=40, color='steelblue', alpha=0.6,
                  orientation='horizontal')
    ax_right.axis('off')

    fig.suptitle(
        r'Figure 8. Joint posterior over $(\theta^H, \theta^S)$',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig8_posterior.pdf')
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig8_posterior.pdf'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 66)
    print("  Empirical identification of theta^H and theta^S")
    print("=" * 66)

    print("\n[1/5] Loading CEX + policy data...")
    df = load_cex_with_policy()
    print(f"       {len(df):,} rows, {df['household_id'].nunique():,} households")
    print(f"       {df['state'].nunique()} states, "
          f"{df['year'].nunique()} years")

    print("\n[2/5] Running DiD across 3 policy sources x 2 categories x 5 quintiles...")
    did_out = run_all_did(df)
    did_out.to_csv(PROCESSED / "did_estimates_by_source.csv", index=False)

    # Print DiD table
    pivot = did_out.pivot_table(
        index=['source', 'category_type'],
        columns='quintile', values='beta'
    ).round(3)
    print("\n     DiD betas:")
    print(pivot.to_string())

    print("\n[3/5] Aggregating to structural theta_q per quintile...")
    qth = aggregate_quintile_thetas(did_out)
    qth.to_csv(PROCESSED / "quintile_thetas.csv", index=False)
    print(qth.round(3).to_string(index=False))

    print("\n[4/5] Bayesian aggregation to theta^H, theta^S...")
    result = bayesian_aggregate_to_types(qth)
    pd.DataFrame([result]).to_csv(
        PROCESSED / "theta_estimates_full.csv", index=False)
    print(f"     theta^H = {result['theta_H']:+.3f}  "
          f"(SE = {result['theta_H_se']:.3f})")
    print(f"     theta^S = {result['theta_S']:+.3f}  "
          f"(SE = {result['theta_S_se']:.3f})")

    print("\n[5/5] Generating posterior figure...")
    plot_posterior(
        theta_H=result['theta_H'], theta_H_se=result['theta_H_se'],
        theta_S=result['theta_S'], theta_S_se=result['theta_S_se'],
    )

    # Write LaTeX-ready micro-main table
    print("\n     Writing paper/tables/tab_micro_main.tex ...")
    write_latex_micro_table(did_out)

    print("\n" + "=" * 66)
    print("  Done.")
    print("=" * 66)


def write_latex_micro_table(did_out: pd.DataFrame) -> None:
    """Write a clean LaTeX fragment for Table 5 (micro-main)."""
    out_path = ROOT / "paper" / "tables"
    out_path.mkdir(parents=True, exist_ok=True)
    # Average betas across the three sources
    avg = did_out.groupby(['category_type', 'quintile']).agg(
        beta=('beta', 'mean'), se=('se', lambda s: np.sqrt((s ** 2).mean()))
    ).reset_index()

    lines = [
        r'\begin{tabular}{@{}crrrr@{}}',
        r'\toprule',
        r'Quintile & {$\beta^q$ comp.} & {(SE)} & {$\beta^q$ sub.} & {(SE)} \\',
        r'\midrule',
    ]
    for q in range(1, 6):
        cb = avg[(avg['category_type'] == 'complement') & (avg['quintile'] == q)]
        sb = avg[(avg['category_type'] == 'substitute') & (avg['quintile'] == q)]
        if len(cb) == 0 or len(sb) == 0: continue
        lines.append(
            f"{q} & {cb['beta'].iloc[0]:+.3f} & {cb['se'].iloc[0]:.3f} & "
            f"{sb['beta'].iloc[0]:+.3f} & {sb['se'].iloc[0]:.3f} \\\\"
        )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    (out_path / "tab_micro_main.tex").write_text('\n'.join(lines) + '\n')


if __name__ == "__main__":
    main()
