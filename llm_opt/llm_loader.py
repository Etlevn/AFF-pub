import json
from pathlib import Path
from typing import Any, Dict


def load_factor_select(run_dir: str) -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[1]
    json_path = base / "analysis" / run_dir / "select" / "factor_select.json"
    if not json_path.exists():
        raise FileNotFoundError(f"factor_select.json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_llm_factor_template() -> Dict[str, Any]:
    base = Path(__file__).resolve().parent
    json_path = base / "llm_factor.json"
    if not json_path.exists():
        raise FileNotFoundError(f"llm_factor.json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_operators_doc() -> str:
    base = Path(__file__).resolve().parents[1]
    md_path = base / "md" / "operators.md"
    if not md_path.exists():
        raise FileNotFoundError(f"operators.md not found: {md_path}")
    return md_path.read_text(encoding="utf-8")


def load_backtest_stats_text(run_dir: str) -> str:
    """Reading backtest data performance (CSV plain text injection): analysis/<run_dir>/select/factor_stats.csv"""
    base = Path(__file__).resolve().parents[1]
    csv_path = base / "analysis" / run_dir / "select" / "factor_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"factor_stats.csv not found: {csv_path}")
    return csv_path.read_text(encoding="utf-8", errors="ignore")


def load_backtest_metrics_doc() -> str:
    """Read backtest indicator description (Markdown text injection): md/backtest.md"""
    base = Path(__file__).resolve().parents[1]
    md_path = base / "md" / "backtest.md"
    if not md_path.exists():
        raise FileNotFoundError(f"backtest.md not found: {md_path}")
    return md_path.read_text(encoding="utf-8")
