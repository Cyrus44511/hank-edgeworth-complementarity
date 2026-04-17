"""
figures.py
----------
Generate all figures referenced in the paper.

Usage:
    python3 figures.py                 # generates all figures
    python3 figures.py --fig 1         # generates Figure 1 only
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytical import BASELINE, Calibration, multiplier, xi, chi

# -----------------------------------------------------------------------------
# Paths and style
# -----------------------------------------------------------------------------
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def boe_style() -> None:
    """Set matplotlib style inspired by the BoE technical paper."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# -----------------------------------------------------------------------------
# Figure 1: Multiplier vs lambda (symmetric)
# -----------------------------------------------------------------------------
def figure_mult_vs_lambda() -> None:
    boe_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lams = np.linspace(0.05, 0.65, 200)
    thetas = [+0.3, 0.0, -0.2, -0.5, -0.8]
    labels = [r"$\theta=+0.3$ (substitute)",
              r"$\theta=0$ (Bilbiie separable)",
              r"$\theta=-0.2$ (weak complement)",
              r"$\theta=-0.5$ (moderate complement)",
              r"$\theta=-0.8$ (strong complement)"]
    for th, lbl in zip(thetas, labels):
        mus = []
        for lam in lams:
            cal = Calibration(lam=lam, mu=lam)  # uniform mu = lam
            mus.append(multiplier(th, th, cal))
        ax.plot(lams, mus, lw=1.8, label=lbl)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8, label=r"$\mu=1$")
    ax.set_xlabel(r"Hand-to-mouth share $\lambda$")
    ax.set_ylabel(r"Fiscal multiplier $\mu(\lambda,\theta)$")
    ax.set_title(r"Figure 1. Multiplier and the Hand-to-Mouth Share"
                 r" (Symmetric: $\theta^S=\theta^H=\theta$, $\chi>1$)")
    ax.set_ylim(-1.2, 5.2)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mult_vs_lambda.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: Multiplier vs theta (symmetric)
# -----------------------------------------------------------------------------
def figure_mult_vs_theta() -> None:
    boe_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    thetas = np.linspace(-0.85, 0.85, 200)
    lams = [0.20, 0.35, 0.50]
    for lam in lams:
        cal = Calibration(lam=lam, mu=lam)
        mus = [multiplier(th, th, cal) for th in thetas]
        ax.plot(thetas, mus, lw=1.8, label=rf"$\lambda={lam:.2f}$")
    ax.axvline(0.0, color="grey", ls="--", lw=0.8)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.annotate("More complementarity\nlowers static multiplier",
                xy=(-0.55, 2.2), fontsize=9, ha="center", color="steelblue")
    ax.annotate("More substitutability\nraises static multiplier",
                xy=(0.45, 3.8), fontsize=9, ha="center", color="darkorange")
    ax.set_xlabel(r"Complementarity parameter $\theta$")
    ax.set_ylabel(r"Fiscal multiplier $\mu(\theta)$")
    ax.set_title(r"Figure 2. Multiplier and Complementarity")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_mult_vs_theta.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: 3D surface (heterogeneous internalization)
# -----------------------------------------------------------------------------
def figure_3d_surface() -> None:
    boe_style()
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return
    thetas_S = np.linspace(-0.75, 0.75, 40)
    thetas_H = np.linspace(-0.75, 0.75, 40)
    TS, TH = np.meshgrid(thetas_S, thetas_H)
    MU = np.vectorize(lambda ts, th: multiplier(ts, th))(TS, TH)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(TS, TH, MU, cmap="viridis", edgecolor="none",
                           alpha=0.90)
    # Bilbiie point (theta=0)
    ax.scatter([0], [0], [multiplier(0, 0)], color="black", s=40,
               label="Bilbiie separable")
    ax.set_xlabel(r"$\theta^S$ (savers)")
    ax.set_ylabel(r"$\theta^H$ (hand-to-mouth)")
    ax.set_zlabel(r"Multiplier $\mu(\theta^S,\theta^H)$")
    ax.set_title(r"Figure 3. Multiplier Surface under Heterogeneous Internalization"
                 r" ($\lambda=0.35$)")
    fig.colorbar(surf, ax=ax, shrink=0.55, label=r"$\mu$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_mult_surface.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 4: Direct fiscal channel xi (two-panel)
# -----------------------------------------------------------------------------
def figure_xi_two_panel() -> None:
    boe_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel A: xi vs theta (symmetric)
    thetas = np.linspace(-0.8, 0.8, 200)
    for lam in [0.20, 0.35, 0.50]:
        cal = Calibration(lam=lam, mu=lam)
        xis = [xi(th, th, cal) for th in thetas]
        axes[0].plot(thetas, xis, lw=1.8, label=rf"$\lambda={lam:.2f}$")
    axes[0].axvline(0.0, color="grey", ls="--", lw=0.8)
    axes[0].set_xlabel(r"Complementarity parameter $\theta$")
    axes[0].set_ylabel(r"Direct fiscal channel $\xi(\theta)$")
    axes[0].set_title(r"Panel A: $\xi$ under symmetric $\theta^S=\theta^H=\theta$")
    axes[0].legend(loc="upper left", frameon=False)

    # Panel B: xi vs theta^H for varying theta^S
    thetas_H = np.linspace(-0.8, 0.8, 200)
    for tS in [-0.3, 0.0, 0.29, 0.6]:
        xis = [xi(tS, tH) for tH in thetas_H]
        axes[1].plot(thetas_H, xis, lw=1.8, label=rf"$\theta^S={tS:+.2f}$")
    axes[1].axvline(0.0, color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel(r"HTM complementarity $\theta^H$")
    axes[1].set_ylabel(r"Direct fiscal channel $\xi(\theta^S,\theta^H)$")
    axes[1].set_title(r"Panel B: $\xi$ vs $\theta^H$ for varying $\theta^S$"
                      r" ($\lambda=0.35$)")
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle("Figure 4. The direct fiscal channel: a new sufficient statistic"
                 " (Proposition 2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_xi_two_panel.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 5: Heatmap of distributional dominance
# -----------------------------------------------------------------------------
def figure_dominance_heatmap() -> None:
    boe_style()
    from analytical import dominance
    thetas_S = np.linspace(-0.8, 0.8, 80)
    thetas_H = np.linspace(-0.8, 0.8, 80)
    TS, TH = np.meshgrid(thetas_S, thetas_H)
    DOM = np.vectorize(dominance)(TS, TH)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.pcolormesh(TS, TH, DOM, cmap="RdBu_r",
                       shading="auto",
                       vmin=-np.max(np.abs(DOM)),
                       vmax=np.max(np.abs(DOM)))
    cs = ax.contour(TS, TH, DOM, levels=[0], colors="black", linewidths=1.5)
    ax.clabel(cs, fmt="dominance = 0")
    ax.scatter([0.29], [-0.76], color="black", s=60, zorder=3,
               label="Baseline (estimated)")
    ax.scatter([0.0], [0.0], color="white", edgecolor="black", s=50, zorder=3,
               label="Bilbiie separable")
    ax.set_xlabel(r"$\theta^S$ (savers)")
    ax.set_ylabel(r"$\theta^H$ (hand-to-mouth)")
    ax.set_title(r"Figure 5. Distributional Dominance Region (Corollary 1)"
                 "\nBlue: $\\mu < \\mu^\\mathrm{Bilbiie}$; Red: $\\mu > \\mu^\\mathrm{Bilbiie}$")
    fig.colorbar(im, ax=ax, label="Dominance expression")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dominance_heatmap.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
ALL_FIGS = {
    1: ("Multiplier vs lambda", figure_mult_vs_lambda),
    2: ("Multiplier vs theta",  figure_mult_vs_theta),
    3: ("Multiplier 3D surface", figure_3d_surface),
    4: ("Direct fiscal channel xi", figure_xi_two_panel),
    5: ("Dominance heatmap", figure_dominance_heatmap),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--fig", type=int, default=None,
                        help="Figure number to generate (default: all)")
    args = parser.parse_args()

    if args.fig is None:
        for n, (name, f) in ALL_FIGS.items():
            print(f"Generating Figure {n}: {name} ...")
            f()
    else:
        name, f = ALL_FIGS[args.fig]
        print(f"Generating Figure {args.fig}: {name} ...")
        f()
    print(f"All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
