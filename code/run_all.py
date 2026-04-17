"""
run_all.py
----------
Master orchestration script. Reproduces all quantitative results and figures
in the paper. Run from the repository root:

    python3 code/run_all.py
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"    [FAIL] exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sim = root / "code" / "simulation"
    emp = root / "code" / "empirical"
    hnk = root / "code" / "hank"

    print("=" * 60)
    print("  HANK + Edgeworth Complementarity -- reproduction suite")
    print("=" * 60)

    # 1. Compute closed-form multipliers
    run(["python3", "analytical.py"], cwd=sim)

    # 2. Estimate theta^j from CEX micro data
    run(["python3", "estimate_theta.py"], cwd=emp)

    # 3. HANK solver
    run(["python3", "hank_jacobian.py"], cwd=hnk)

    # 4. Generate figures
    run(["python3", "figures.py"], cwd=sim)

    print("\n" + "=" * 60)
    print("  Done. Figures written to paper/figures/.")
    print("=" * 60)


if __name__ == "__main__":
    main()
