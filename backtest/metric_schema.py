"""Read historical backtest headers using the current English metric schema."""

import pandas as pd


# Encoded aliases are protocol compatibility data, not display text. New exports
# always use the English names; existing local CSV files remain readable.
LEGACY_METRIC_NAMES = {
    "\u8d85\u989dIC": "ExcessIC",
    "\u5e74\u5316": "annualized",
    "\u7d2f\u8ba1": "cumulative",
    "\u590f\u666e": "sharpe",
    "\u80dc\u7387": "win_rate",
    "\u56de\u64a4": "drawdown",
}


def normalize_metric_name(name: str) -> str:
    """Translate a historical metric, including train/valid/test prefixes."""
    if not isinstance(name, str):
        return name
    for old, new in LEGACY_METRIC_NAMES.items():
        name = name.replace(old, new)
    return name


def normalize_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with English headers, rejecting ambiguous duplicate names."""
    columns = [normalize_metric_name(name) for name in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate metric columns after English normalization")
    normalized = frame.copy()
    normalized.columns = columns
    return normalized
