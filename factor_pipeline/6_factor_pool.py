"""
Factor Pool Integration (Step 6)

According to RUN_DIR, the following products are integrated into the pool/<RUN_DIR>/ directory:
- All files and subdirectories under out/<RUN_DIR>/ (if they exist)
- analysis/<RUN_DIR>/select/factor_select.csv (if exists)

Usage (standalone operation):
  python factor_pipeline/6_factor_pool.py --run-dir dcgan_netp_v1
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# ============================== [CONFIG] ==============================

RUN_DIR = 'test'
MODE = 'dft'

# ======================================================================


def _copy_tree(src: Path, dst: Path):
    if not src.exists():
        return
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            d = target_root / f
            shutil.copy2(s, d)


def integrate_pool(run_dir: str, mode: str = MODE) -> str:
    """
    Merge out/<dir>/ with analysis/<run_dir>/select/factor_stats.csv and write pool/<run_dir>/.
    - mode=’dft’: analysis uses run_dir; out uses f”{run_dir}_0”
    - mode='opt': analysis uses run_dir; out uses the trailing-underscore directory.
    Returns the target directory path string.
    """
    run_dir = str(run_dir).strip()
    mode = (mode or 'dft').strip().lower()
    project_root = Path(__file__).resolve().parents[1]

    # dft: Use out/<RUN_DIR>_0
    # opt: Use out/<RUN_DIR>_ (consistent with opt mode of 5_factor_builder.py)
    effective_out = (f"{run_dir}_" if mode == 'opt' else f"{run_dir}_0")
    out_dir = project_root / 'out' / effective_out
    # Analysis result source: dft -> analysis/<RUN_DIR>/select/factor_stats.csv
    #              opt -> analysis/<RUN_DIR>/select/factor_stats.csv
    effective_analysis = run_dir
    analysis_stats_csv = project_root / 'analysis' / effective_analysis / 'select' / 'factor_stats.csv'
    # Target pool directory name: dft -> <RUN_DIR>_0, opt -> <RUN_DIR>
    effective_pool = (run_dir if mode == 'opt' else f"{run_dir}_0")
    pool_dir = project_root / 'pool' / effective_pool

    print(f"[POOL] run_dir = {run_dir}  mode={mode}")
    print(f"[POOL] out from: {out_dir} (name={effective_out})")
    print(f"[POOL] stats csv: {analysis_stats_csv} (analysis name={effective_analysis})")
    print(f"[POOL] dest: {pool_dir} (name={effective_pool})")

    pool_dir.mkdir(parents=True, exist_ok=True)

    # Copy the contents of the out directory
    if out_dir.exists():
        _copy_tree(out_dir, pool_dir)
        print(f"[POOL] copied out/ -> {pool_dir}")
    else:
        print(f"[POOL] warn: out directory does not exist, skip: {out_dir}")

    # Copy factor_stats.csv to the target root directory
    if analysis_stats_csv.exists():
        shutil.copy2(analysis_stats_csv, pool_dir / 'factor_stats.csv')
        print(f"[POOL] copied factor_stats.csv -> {pool_dir / 'factor_stats.csv'}")
    else:
        print(f"[POOL] warn: No statistics file found: {analysis_stats_csv}")

    return str(pool_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=RUN_DIR, help='analysis directory name, example dcgan_netp_v1_')
    parser.add_argument('--mode', default=MODE, choices=['dft', 'opt'], help='Integration mode: dft or opt')
    args = parser.parse_args()
    run_dir = (args.run_dir or RUN_DIR).strip()
    mode = (args.mode or MODE).strip().lower()
    integrate_pool(run_dir, mode=mode)


if __name__ == '__main__':
    main()
