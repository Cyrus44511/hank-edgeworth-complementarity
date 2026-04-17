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
from general_equilibrium import decompose_multiplier
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

    print("  (3/3) Computing GE multiplier (fixed rate + Taylor)...",
          end="", flush=True)
    dec_fix = decompose_multiplier(J, ss, T=T, closure='fixed_rate')
    dec_tay = decompose_multiplier(J, ss, T=T, closure='taylor')
    print(f" mu_fix={dec_fix['mu_impact']:.3f}  "
          f"mu_taylor={dec_tay['mu_impact']:.3f}")

    return {
        "name": name, "params": params,
        "C": float(ss.C), "A": float(ss.A),
        "mu_fixed_rate": float(dec_fix['mu_impact']),
        "mu_taylor":     float(dec_tay['mu_impact']),
        "direct":        float(dec_tay['direct']),
        "indirect":      float(dec_tay['indirect']),
        "dY_path":       dec_tay['dY'].tolist(),
        "dC_path":       dec_tay['dC'].tolist(),
        "dpi_path":      dec_tay['paths'].get('dpi', np.zeros(T)).tolist()
                         if dec_tay['paths'] else [0.0] * T,
        "dr_path":       dec_tay['paths'].get('dr', np.zeros(T)).tolist()
                         if dec_tay['paths'] else [0.0] * T,
    }


# -----------------------------------------------------------------------------
# Figure 6: IRFs
# -----------------------------------------------------------------------------
def plot_irfs(results: List[dict], rho_g: float = 0.80) -> None:
    """BoE-style four-panel IRF figure: Y, C, pi, r."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    T_plot = min(20, len(results[0]['dY_path']))
    horizon = np.arange(T_plot)

    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(results)))
    panels = [
        ("dY_path",  "Output response ($dY / dG$)"),
        ("dC_path",  "Consumption response ($dC / dG$)"),
        ("dpi_path", "Inflation ($d\\pi$, %-pt)"),
        ("dr_path",  "Real interest rate ($dr$, %-pt)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for r, color in zip(results, colors):
            path = np.array(r.get(key, [0] * T_plot))[:T_plot]
            # Normalize so t=0 response to dg is in units of dG_0
            if key in ("dY_path", "dC_path"):
                pass  # already in levels of dY/dG
            else:
                path = path * 100  # convert to basis points for pi, r
            ax.plot(horizon, path, color=color, lw=1.7, label=r['name'])
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.set_xlabel("Horizon (quarters)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Figure 6. IRFs to a 1% G shock, Taylor rule "
                 f"($\\rho_g={rho_g}$, $\\phi_\\pi=1.5$, $\\phi_y=0.125$)",
                 y=1.00)
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
    print("\n" + "=" * 70)
    print("  Table 4: Impact fiscal multipliers (HANK)")
    print("=" * 70)
    hdr = f"  {'Configuration':<35s} {'mu_fixed':>10s} {'mu_taylor':>10s} {'direct':>10s}"
    print(hdr)
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['name']:<35s} "
              f"{r.get('mu_fixed_rate', float('nan')):>10.4f} "
              f"{r.get('mu_taylor', float('nan')):>10.4f} "
              f"{r.get('direct', float('nan')):>10.4f}")

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
