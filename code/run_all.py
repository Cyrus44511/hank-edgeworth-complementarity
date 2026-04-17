"""
run_all.py
----------
Master orchestration script. Reproduces all quantitative and empirical
results in the paper:
  1. Analytical multipliers (simulation/analytical.py)
  2. Full HANK solver (hank/run_hank.py)
  3. Empirical identification of theta^H, theta^S (empirical/run_empirical.py)
  4. All figures (simulation/figures.py)

Run from repo root:
    python3 code/run_all.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"    [FAIL] exit code {r.returncode}")
        sys.exit(r.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sim = root / "code" / "simulation"
    emp = root / "code" / "empirical"
    hnk = root / "code" / "hank"

    print("=" * 66)
    print("  HANK + Edgeworth Complementarity -- full reproduction pipeline")
    print("=" * 66)

    # 1. Analytical closed-form multipliers
    run(["python3", "analytical.py"], cwd=sim)

    # 2. Full HANK SSJ solver
    run(["python3", "run_hank.py", "--quick"], cwd=hnk)

    # 3. Empirical identification of theta^H, theta^S
    run(["python3", "run_empirical.py"], cwd=emp)

    # 4. Closed-form figures
    run(["python3", "figures.py"], cwd=sim)

    print("\n" + "=" * 66)
    print("  Done. See paper/figures/, paper/tables/, data/processed/")
    print("=" * 66)


if __name__ == "__main__":
    main()
