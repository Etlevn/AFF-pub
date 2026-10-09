"""
Factor JSON Exporter

Read analysis/<RUN_DIR>/backtest/factor_select.csv and export JSON (only including idx, expr, enabled=true).
JSON files are output to analysis/<RUN_DIR>/backtest/<JSON_FILENAME> (default factor.json).

JSON structure reference llm_opt/factor.json:
[
  {"idx": "40", "expr": "<expression>", "enabled": true},
  ...
]
"""

from __future__ import annotations

import os
import sys
import json
import pandas as pd
from typing import List, Dict

# ============================== [CONFIG] ==============================

RUN_DIR = "test"

# ======================================================================

def _backtest_dir(run_dir: str) -> str:
    return os.path.join("analysis", run_dir, "backtest")

def _select_dir(run_dir: str) -> str:
    return os.path.join("analysis", run_dir, "select")


def load_selected_factors(run_dir: str) -> pd.DataFrame:
    select_dir = _select_dir(run_dir)
    csv_path = os.path.join(select_dir, "factor_select.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"factor_select.csv does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    # Compatible column names: factor (index/number), expression or Expression (expression)
    if "factor" not in df.columns:
        raise ValueError("factor_select.csv missing column ’factor’")
    expr_col = "expression" if "expression" in df.columns else ("Expression" if "Expression" in df.columns else None)
    if expr_col is None:
        raise ValueError("factor_select.csv Missing expression column ’expression’ or ’Expression’")

    df["factor"] = df["factor"].astype(str)
    df[expr_col] = df[expr_col].astype(str)
    return df[["factor", expr_col]].rename(columns={expr_col: "expr"})


def export_json(run_dir: str, df: pd.DataFrame) -> str:
    select_dir = _select_dir(run_dir)
    os.makedirs(select_dir, exist_ok=True)
    save_path = os.path.join(select_dir, "factor_select.json")

    records: List[Dict] = []
    for row in df.itertuples(index=False):
        records.append({
            "enabled": True,
            "idx": getattr(row, "factor"),
            "expr": getattr(row, "expr"),
            "info": {
                "name": "",
                "desc": ""
            }
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Print output summary
    try:
        print(f"[JSON] JSON has been generated: {save_path} ({len(df)} factors)")
    except Exception:
        print(f"[JSON] JSON has been generated: {save_path}")

    return save_path


def main():
    print("🎯 Factor JSON export module")
    print("=" * 40)
    print(f"RUN_DIR: {RUN_DIR}")

    try:
        selected = load_selected_factors(RUN_DIR)
    except Exception as e:
        print(f"❌ Failed to read factor_select.csv: {e}")
        sys.exit(1)

    try:
        out_path = export_json(RUN_DIR, selected)
    except Exception as e:
        print(f"❌ Export JSON failed: {e}")
        sys.exit(1)

    print(f"💾 Exported JSON: {out_path}")
    print(f"✅ Total writes {len(selected)} strip factor")


if __name__ == "__main__":
    main()
