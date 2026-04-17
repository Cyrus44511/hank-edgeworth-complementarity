"""
run_hank.py
-----------
Master script for the HANK quantitative analysis. Reproduces:
  - Table 4 of the paper: impact multipliers across configurations
  - Figure 6: IRFs of (Y, C, w) to a 1% G shock under each configuration
  - Figure 7: contribution-by-wealth-quintile decomposition

Usage:
    python3 run_hank.py                # full reproduction (slow: ~10 min)
    python3 run_hank.py --quick        # smaller grid, faster (~2 min)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from steady_state import solve_steady_state
from jacobians import compute_jacobian_truncated
from general_equilibrium import (
    compute_impact_multiplier_fixed_rate, decompose_multiplier,
)
from household import HouseholdParams, theta_by_wealth

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Configurations to run
# -----------------------------------------------------------------------------
CONFIGS = {
    "Bilbiie separable":              dict(theta_S= 0.00, theta_H= 0.00),
    "Symmetric complementarity":      dict(theta_S=-0.50, theta_H=-0.50),
    "Symmetric substitutability":     dict(theta_S= 0.30, theta_H= 0.30),
    "Heterogeneous internalization":  dict(theta_S= 0.29, theta_H=-0.76),
    "Estimated upper bound":          dict(theta_S= 0.38, theta_H=-0.90),
}


def compute_one_configuration(name: str, params: dict, T: int, n_a: int,
                              n_y: int) -> dict:
    """Solve SS, compute Jacobians, compute GE multiplier for one config."""
    print(f"\n[{name}]  theta_S={params['theta_S']:+.2f}, "
          f"theta_H={params['theta_H']:+.2f}")
    print("  (1/3) Solving steady state...", end="", flush=True)
    ss = solve_steady_state(theta_S=params['theta_S'],
                            theta_H=params['theta_H'],
                            n_a=n_a, n_y=n_y)
    print(f" C={ss.C:.3f}, A={ss.A:.3f}")

    print(f"  (2/3) Computing Jacobian (T={T})...", end="", flush=True)
    J = compute_jacobian_truncated(ss, T=T)
    print(" done")

    print("  (3/3) Computing GE multiplier...", end="", flush=True)
    dec = decompose_multiplier(J, ss, T=T)
    print(f" mu={dec['mu_impact']:.4f}")

    return {
        "name": name, "params": params,
        "C": float(ss.C), "A": float(ss.A),
        "mu": float(dec['mu_impact']),
        "direct": float(dec['direct']),
        "indirect": float(dec['indirect']),
        "dY_path": dec['dY'].tolist(),
        "dC_path": dec['dC'].tolist(),
    }


# -----------------------------------------------------------------------------
# Figure 6: IRFs
# -----------------------------------------------------------------------------
def plot_irfs(results: List[dict], rho_g: float = 0.80) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    T_plot = min(20, len(results[0]['dY_path']))
    horizon = np.arange(T_plot)

    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(results)))
    for r, color in zip(results, colors):
        axes[0].plot(horizon, r['dY_path'][:T_plot], color=color, lw=1.7,
                     label=r['name'])
        axes[1].plot(horizon, r['dC_path'][:T_plot], color=color, lw=1.7,
                     label=r['name'])
    for ax, title in zip(axes, ["Output dY/G", "Consumption dC/G"]):
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.set_xlabel("Horizon (quarters)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Figure 6. IRFs to a 1% G shock (rho_g = {rho_g})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_irfs.pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig6_irfs.pdf'}")


# -----------------------------------------------------------------------------
# Figure 7: wealth-decomposition (uses full SS for the heterogeneous case)
# -----------------------------------------------------------------------------
def plot_wealth_decomposition() -> None:
    """Decompose impact dY by wealth quintile under heterogeneous internalization."""
    ss = solve_steady_state(theta_S=0.29, theta_H=-0.76, n_a=200, n_y=5)
    n_a = len(ss.a_grid)

    # Quintile boundaries by wealth
    wealth = ss.D.sum(axis=1)
    cum = np.cumsum(wealth)
    q_bounds = [0]
    for q in range(1, 6):
        idx = np.searchsorted(cum, q / 5.0)
        q_bounds.append(min(idx, n_a))
    q_bounds[-1] = n_a

    # Contribution to aggregate consumption by quintile
    quintile_C = np.zeros(5)
    for q in range(5):
        quintile_C[q] = (ss.D[q_bounds[q]:q_bounds[q+1], :] *
                         ss.c_policy[q_bounds[q]:q_bounds[q+1], :]).sum()
    # Contribution to consumption response (proxy: weight by mean theta in each)
    quintile_theta = np.zeros(5)
    for q in range(5):
        sl = ss.theta_grid[q_bounds[q]:q_bounds[q+1]]
        quintile_theta[q] = sl.mean() if len(sl) > 0 else ss.theta_grid[q_bounds[q]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    qs = np.arange(1, 6)
    axes[0].bar(qs, quintile_C / quintile_C.sum(), color="steelblue",
                edgecolor="black")
    axes[0].set_xlabel("Wealth quintile")
    axes[0].set_ylabel("Share of aggregate C")
    axes[0].set_title("Panel A. Consumption shares by wealth quintile")

    axes[1].bar(qs, quintile_theta, color="firebrick", edgecolor="black")
    axes[1].axhline(0, color='black', lw=0.6)
    axes[1].set_xlabel("Wealth quintile")
    axes[1].set_ylabel(r"Average $\theta$")
    axes[1].set_title(r"Panel B. $\theta$ profile by wealth quintile")

    fig.suptitle("Figure 7. Wealth-based decomposition (heterogeneous internalization)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_wealth_decomposition.pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig7_wealth_decomposition.pdf'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Smaller grid for fast turnaround")
    ap.add_argument("--T", type=int, default=15,
                    help="Time horizon for IRFs/Jacobian")
    ap.add_argument("--n_a", type=int, default=None)
    ap.add_argument("--n_y", type=int, default=None)
    args = ap.parse_args()

    n_a = args.n_a or (60 if args.quick else 120)
    n_y = args.n_y or (5 if args.quick else 7)
    T = args.T

    print("=" * 60)
    print(f"  HANK reproduction  --  T={T}, n_a={n_a}, n_y={n_y}")
    print("=" * 60)

    results = []
    for name, params in CONFIGS.items():
        try:
            r = compute_one_configuration(name, params, T, n_a, n_y)
            results.append(r)
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")
            results.append({"name": name, "params": params,
                            "mu": float('nan'),
                            "dY_path": [0.0] * T, "dC_path": [0.0] * T})

    # Print Table 4
    print("\n" + "=" * 60)
    print("  Table 4: Impact fiscal multipliers (HANK)")
    print("=" * 60)
    print(f"  {'Configuration':<35s} {'mu':>8s} {'direct':>8s} {'indirect':>10s}")
    for r in results:
        print(f"  {r['name']:<35s} {r['mu']:>8.4f} "
              f"{r.get('direct', float('nan')):>8.4f} "
              f"{r.get('indirect', float('nan')):>10.4f}")

    # Save results to JSON for the paper
    out = ROOT / "data" / "processed" / "hank_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved table 4 results to {out}")

    # Generate figures
    print("\n  Generating Figure 6 (IRFs)...")
    plot_irfs(results)
    print("  Generating Figure 7 (wealth decomposition)...")
    plot_wealth_decomposition()
    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
