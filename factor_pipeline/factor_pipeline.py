"""
Run the six-stage factor evaluation and pool-building pipeline.

1. Calculate factor series and inventory under analysis/<RUN_DIR>/.
2. Backtest factors and export results for the configured data splits.
3. Rank and select factors, then export statistics.
4. Export selected expressions to JSON.
5. Rebuild factor artifacts under out/<RUN_DIR>_0 (dft) or out/<RUN_DIR>_ (opt).
6. Integrate artifacts into pool/<RUN_DIR>_0 (dft) or pool/<RUN_DIR> (opt).

Examples:
  python factor_pipeline/factor_pipeline.py --run-dir YOUR_RUN --steps 1:6
  python factor_pipeline/factor_pipeline.py --run-dir YOUR_RUN --steps 3:6 --mode opt

Each stage uses its own configuration defaults. RUN_DIR selects the common run.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================== [CONFIG] ==============================
RUN_DIR = 'aff_all_2022-2024_0_lstm_netp_v2'   # Run directory (out/<RUN_DIR>)
STEPS = '3:6'      # Run steps: ’1:5’ or ’3:5’ or ’6:6’
MODE = 'dft'       # Operating mode: ’dft’ | ’opt’
# ======================================================================

def parse_steps(s: str) -> list[int]:
    s = s.strip()
    if ":" in s:
        parts = s.split(":", 1)
        a = int(parts[0]) if parts[0] else 1
        b = int(parts[1]) if parts[1] else 5
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    else:
        n = int(s)
        if n < 1 or n > 5:
            raise ValueError("steps must be in the range of 1..5")
        return [n]


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {module_name} <- {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_step_1(underscored_dir: str):
    path = str(Path(PROJECT_ROOT) / 'factor_pipeline' / '1_factor_process.py')
    m = load_module_from_path('fp_step1', path)
    setattr(m, 'MODE', MODE)
    m.run_factor_analysis_pipeline(specified_dir=underscored_dir)


def run_step_2(underscored_dir: str):
    path = str(Path(PROJECT_ROOT) / 'factor_pipeline' / '2_factor_backtest.py')
    m = load_module_from_path('fp_step2', path)
    setattr(m, 'MODE', MODE)
    m.run_backtest(backtest_dir=underscored_dir)


def run_step_3(run_dir: str, underscored_dir: str, mode: str):
    path = str(Path(PROJECT_ROOT) / 'factor_pipeline' / '3_factor_select.py')
    m = load_module_from_path('fp_step3', path)
    setattr(m, 'MODE', mode)
    scores = m.compute_overall_scores(run_dir=run_dir,
                                      metric_weights=m.METRIC_WEIGHTS,
                                      table_weights=m.TABLE_WEIGHTS)
    m.select_top_factors(run_dir=run_dir, scores=scores, top_k=m.TOP_K, mode=mode)


def run_step_4(run_dir: str, underscored_dir: str):
    path = str(Path(PROJECT_ROOT) / 'factor_pipeline' / '4_factor_json.py')
    m = load_module_from_path('fp_step4', path)
    selected = m.load_selected_factors(run_dir)
    m.export_json(run_dir, selected)


def run_step_5(run_dir: str, mode: str):
    path = str(Path(PROJECT_ROOT) / 'factor_pipeline' / '5_factor_builder.py')
    m = load_module_from_path('fp_step5', path)
    setattr(m, 'RUN_DIR', run_dir)
    setattr(m, 'MODE', mode)
    m.main()


def run_step_6(run_dir: str, mode: str):
    path = str(Path(PROJECT_ROOT) / 'factor_pipeline' / '6_factor_pool.py')
    m = load_module_from_path('fp_step6', path)
    setattr(m, 'MODE', mode)
    m.integrate_pool(run_dir, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=RUN_DIR, help='Training run directory name under out/, such as dcgan_netp_v1')
    parser.add_argument('--steps', default=STEPS, help='Step range such as 1:6 or 3:6, or a single step from 1 to 5')
    parser.add_argument('--mode', default=MODE, choices=['dft','opt'], help='Operating mode: dft or opt')
    args = parser.parse_args()

    run_dir: str = (args.run_dir or RUN_DIR).strip()
    underscored_dir = run_dir
    steps = parse_steps(args.steps or STEPS)
    mode = (args.mode or MODE).strip().lower()

    print(f"\n[PIPELINE] RUN_DIR = {run_dir}")

    for step in steps:
        print(f"\n============================== [ STEP {step} ] ==============================\n")
        if step == 1:
            run_step_1(underscored_dir)
        elif step == 2:
            run_step_2(underscored_dir)
        elif step == 3:
            run_step_3(run_dir, underscored_dir, mode)
        elif step == 4:
            run_step_4(run_dir, underscored_dir)
        elif step == 5:
            run_step_5(run_dir, mode)
        elif step == 6:
            run_step_6(run_dir, mode)
        else:
            raise ValueError('Invalid step number')

    print(f"\n[PIPELINE] DONE\n")


if __name__ == '__main__':
    main()
