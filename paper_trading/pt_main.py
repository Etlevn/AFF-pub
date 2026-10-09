"""Calculate research signals from local data and export stock pools to CSV."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_trading.local_data import LocalCSVData
from paper_trading.pt_config import PipelineConfig
from paper_trading.pt_fetcher import PTFetcher


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing <instrument>.csv files")
    parser.add_argument("--instruments", required=True, help="Local instrument metadata CSV")
    parser.add_argument("--suspensions", help="Optional CSV with time and JSON-array tp_codes columns")
    parser.add_argument("--output", default="exports", help="Local output directory")
    parser.add_argument("-m", choices=["sgl", "mpl"], required=True, help="Single or multiple factor mode")
    parser.add_argument("-d", required=True, help="Directory name under pool/")
    parser.add_argument("-n", help="Factor index for single-factor mode")
    parser.add_argument("-s", required=True, help="Pool name or multiple-factor naming rule")
    parser.add_argument("--sort", help="Statistics column for multiple-factor selection")
    parser.add_argument("--desc", choices=["true", "false"], default="true")
    parser.add_argument("--limit", type=int, help="Maximum factors in multiple-factor mode")
    return parser


def export_pool(pool_name, frame, directory):
    """Write one generated pool without allowing a name to escape the output directory."""
    pool_name = str(pool_name)
    if Path(pool_name).name != pool_name or pool_name in ("", ".", "..") or "\\" in pool_name:
        raise ValueError("Stock-pool names must be plain file names")
    output = Path(directory).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{pool_name}.csv"
    frame.to_csv(path, index=False)
    return path


def main():
    args = build_parser().parse_args()
    config = PipelineConfig()
    info = pd.read_csv(args.instruments, dtype={"tr_code": str, "bond_code": str, "stock_code": str})
    suspensions = None
    if args.suspensions:
        import json
        suspensions = pd.read_csv(args.suspensions)
        suspensions["tp_codes"] = suspensions["tp_codes"].map(json.loads)

    # Load calculation dependencies only after parsing, so --help works offline.
    from paper_trading.pt_pipeline import TradingPipeline
    pipeline = TradingPipeline()
    pipeline.fetcher = PTFetcher(LocalCSVData(args.data_dir))
    saved = []

    def save_pool(name, frame):
        path = export_pool(name, frame, args.output)
        saved.append(path)
        print(f"Saved stock pool: {path}")

    pipeline.run(df_info=info, config=config, tp_info=suspensions, on_pool=save_pool)
    if not saved:
        raise RuntimeError("The pipeline did not produce a stock pool")


if __name__ == "__main__":
    main()
