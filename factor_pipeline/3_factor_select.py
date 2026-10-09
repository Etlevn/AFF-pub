"""
Factor Select Module
According to the given indicators and weights, the backtest results (TRAIN/VALID/TEST) are divided into tables and overall weighted rankings,
Supports two selection modes:
- dft: Sort by global overall_rank, take the first TOP_K factors;
- opt: Filter by ”tree grouping” (factor numbers with the same integer digit are classified as the same tree, and those with the decimal part 0 are regarded as root of the tree).

opt pattern rules:
1) First calculate overall_rank with the same logic;
2) Group the factors by tree;
3) In each tree, find root (the decimal part is the factor of 0), and record its overall_rank as the threshold;
   If root does not exist, the minimum overall_rank in the tree is used as the threshold;
4) Select the factors (including root) within tree that overall_rank is less than or equal to the threshold, and then select the smallest overall_rank and at most 3 factors;
5) merge the results of all tree and write them out.

Enter the directory structure (take run_dir=”test” as an example):
  - analysis/test/backtest/backtest_train.csv
  - analysis/test/backtest/backtest_valid.csv (optional)
  - analysis/test/backtest/backtest_test.csv
  - analysis/test/backtest/factor_list.csv

Output:
  - analysis/test/select/factor_select.csv
"""

from __future__ import annotations

import os
import sys
import math
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Support direct execution as well as loading through the orchestrator.
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.metric_schema import normalize_metric_columns

# ============================== [CONFIG] ==============================

# Backtest running directory
RUN_DIR = "test"

# The selected indicator column and its weight
METRIC_WEIGHTS: Dict[str, float] = {
    "ExcessIC": 1,
    "annualized_long_excess": 1,
    "sharpe_long_excess": 1,
}

# The weight between the three tables (TRAIN/VALID/TEST) (non-existing tables will be automatically ignored)
TABLE_WEIGHTS: Dict[str, float] = {
    "TRAIN": 1,
    "TEST": 1,
}

# Select the first few factors
TOP_K = 10

# Select mode: ’dft’ | ’opt’
MODE = "dft"

# opt mode: Whether to unconditionally retain the root factors of each tree
# is True: each tree contains at least its root; in addition to root, at most 3 are retained
# is False: keep the original logic (filter according to the root threshold and take at most 3, root is not forced to be retained) When
KEEP_ROOT_ALWAYS = True

# Whether to export factor summary statistics table factor_sum.csv
# is True: factor_sum.csv is generated while saving factor_stats.csv
# factor_sum.csv contains statistical information of all indicators: mean, std, med, min, max
EXPORT_FACTOR_SUMMARY = True

# ======================================================================


def _backtest_dir(run_dir: str) -> str:
    return os.path.join("analysis", run_dir, "backtest")


def _load_table(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        return None
    try:
        df = normalize_metric_columns(pd.read_csv(file_path))
        # Key column standardization
        if "factor" not in df.columns:
            raise ValueError(f"Missing column ’factor’: {file_path}")
        df["factor"] = df["factor"].astype(str)
        return df
    except Exception as e:
        print(f"❌ Failed to read table: {file_path} -> {e}")
        return None


def _rank_scores(df: pd.DataFrame, metric_weights: Dict[str, float]) -> pd.Series:
    """
    Sort the specified metric column of df to get the ranking (1 is the best),
    Then calculate the weighted average ranking (rank) of the table according to the weight.

    Rules:
    - For IC, excess IC, RankIC: sort by absolute value (the larger the absolute value, the better)
    - For other indicators: sort by original value (the larger the value, the better)
    - For missing values: use the worst ranking (len(df) + 1).
    - Finally return a Series: index=factor, values=weighted_rank.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    n = len(df)
    worst_rank = float(n + 1)

    # Make sure factor is the index
    work = df.copy()
    work["factor"] = work["factor"].astype(str)
    work = work.set_index("factor", drop=False)

    per_metric_rank: Dict[str, pd.Series] = {}
    total_w = 0.0

    for metric, w in metric_weights.items():
        if w is None or w == 0:
            continue
        if metric not in work.columns:
            print(f"[SELECT] The indicator column is missing, skip: {metric}")
            continue

        # For IC, excess IC, RankIC, use absolute value sorting, and other indicators use original value sorting
        if metric in ["IC", "ExcessIC", "RankIC"]:
            # Ranking: The bigger the absolute value, the better -> rank (ascending=False)
            ranks = work[metric].abs().rank(ascending=False, method="min")
        else:
            # Ranking: The bigger the value, the better -> rank (ascending=False)
            ranks = work[metric].rank(ascending=False, method="min")

        # Missing values are set to the worst ranking
        ranks = ranks.fillna(worst_rank)
        per_metric_rank[metric] = ranks
        total_w += float(w)

    if total_w == 0 or not per_metric_rank:
        # No valid index, return empty
        return pd.Series(dtype=float)

    # Weighted average ranking
    weighted_sum = None
    for metric, ranks in per_metric_rank.items():
        w = float(metric_weights.get(metric, 0.0))
        if w == 0:
            continue
        contrib = ranks * w
        weighted_sum = contrib if weighted_sum is None else (weighted_sum + contrib)

    weighted_rank = weighted_sum / total_w
    weighted_rank.name = "table_rank"
    return weighted_rank


def compute_overall_scores(run_dir: str,
                           metric_weights: Dict[str, float],
                           table_weights: Dict[str, float]) -> pd.DataFrame:
    """
    Read backtest_{TRAIN,VALID,TEST}.csv, calculate the weighted rank of each table,
    Then weight according to the table weight to obtain overall_rank.

    Return DataFrame: columns=[TRAIN, VALID, TEST, overall_rank]
    """
    bdir = _backtest_dir(run_dir)
    paths = {
        "TRAIN": os.path.join(bdir, "backtest_train.csv"),
        "VALID": os.path.join(bdir, "backtest_valid.csv"),
        "TEST":  os.path.join(bdir, "backtest_test.csv"),
    }

    table_ranks: Dict[str, pd.Series] = {}
    # Only check the tables in TABLE_WEIGHTS with weight > 0, other tables are completely ignored (no prompts are printed)
    enabled_tables = {k for k, w in table_weights.items() if w}
    for name, p in paths.items():
        if name not in enabled_tables:
            continue
        df = _load_table(p)
        if df is None:
            # The table is enabled but the file is missing, prompting
            print(f"[SELECT] Not found {name} table, ignore: {p}")
            continue
        ranks = _rank_scores(df, metric_weights)
        if ranks.empty:
            print(f"[SELECT] {name} table has no valid index, ignore")
            continue
        table_ranks[name] = ranks

    if not table_ranks:
        raise RuntimeError("No backtest_* table available or no valid indicators, ranking cannot be calculated")

    # Merge rank of all tables
    all_index = None
    for s in table_ranks.values():
        all_index = s.index if all_index is None else all_index.union(s.index)

    result = pd.DataFrame(index=all_index)
    for name, s in table_ranks.items():
        result[name] = s

    # Only ”actually existing tables” are weighted. Missing tables do not generate columns and do not participate in weighting.
    effective = [(name, float(w)) for name, w in table_weights.items() if w and name in result.columns]
    if not effective:
        raise RuntimeError("No table available for weighting (check TABLE_WEIGHTS vs. actual existing backtest_* files)")

    # Calculate the worst ranking for each actually existing table and fill in the missing values in the list
    worst_by_table = {}
    for name, _ in effective:
        col = result[name]
        worst_by_table[name] = (col.max(skipna=True) if col.notna().any() else 0) + 1

    weight_sum = sum(w for _, w in effective)
    weighted_sum = None
    for name, w in effective:
        worst = worst_by_table[name]
        col = result[name].fillna(worst)
        contrib = col * (w / weight_sum)
        weighted_sum = contrib if weighted_sum is None else (weighted_sum + contrib)

    result["overall_rank"] = weighted_sum
    # All fraction fields retain two decimal places.
    numeric_cols = result.select_dtypes(include="number").columns
    result[numeric_cols] = result[numeric_cols].round(2)
    result = result.sort_values("overall_rank", ascending=True)
    result.index.name = "factor"
    return result


def _parse_tree_and_root_flag(factor_label: str) -> Tuple[str, bool]:
    """
    Analyze the factor number and return (tree_key, is_root).
    - tree_key: a string of integer digits (if it is not a number, the original string is key)
    - is_root: The decimal part is 0 and is regarded as True; non-digits are regarded as True
    """
    s = str(factor_label).strip()
    try:
        v = float(s)
        tree_key = str(int(v))
        # Determine whether it is an integer (the decimal part is 0)
        is_root = float(v).is_integer()
        return tree_key, is_root
    except Exception:
        # Unresolvable tag encountered: treat itself as a tree and treat itself as root
        return s, True


def _select_by_mode_dft(run_dir: str,
                        scores: pd.DataFrame,
                        top_k: int) -> pd.DataFrame:
    """
    According to the global overall_rank, take the previous top_k, merge factor_list.csv and output.
    """
    bdir = _backtest_dir(run_dir)
    select_dir = os.path.join("analysis", run_dir, "select")
    factor_list_path = os.path.join(bdir, "factor_list.csv")
    if not os.path.exists(factor_list_path):
        raise FileNotFoundError(f"factor_list.csv does not exist: {factor_list_path}")

    fl = pd.read_csv(factor_list_path)
    if "factor" not in fl.columns:
        raise ValueError("factor_list.csv missing column ’factor’")
    fl["factor"] = fl["factor"].astype(str)

    merged = fl.merge(scores.reset_index(), on="factor", how="inner")
    merged = merged.sort_values("overall_rank", ascending=True).head(top_k)
    # Selective retention of decimal places before output:
    # - alphaforge_score reserves the 4 bit
    # - Other numerical columns reserve the 2 bit
    num_cols = list(merged.select_dtypes(include="number").columns)
    if "alphaforge_score" in merged.columns and "alphaforge_score" in num_cols:
        num_cols_except_af = [c for c in num_cols if c != "alphaforge_score"]
        if num_cols_except_af:
            merged[num_cols_except_af] = merged[num_cols_except_af].round(2)
        merged["alphaforge_score"] = merged["alphaforge_score"].round(4)
    else:
        merged[num_cols] = merged[num_cols].round(2)

    save_path = os.path.join(select_dir, "factor_select.csv")
    os.makedirs(select_dir, exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"[SELECT] Saved CSV: {save_path}")
    return merged


def _select_by_mode_opt(run_dir: str, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Group by tree and filter according to the root threshold, then take up to 3 optimal factors and merge the output.
    """
    if scores is None or scores.empty:
        return pd.DataFrame(columns=["factor"])  # Empty return

    # Normalize index with required columns
    work = scores.copy()
    if work.index.name != "factor":
        work.index.name = "factor"
    work = work.reset_index()
    work["factor"] = work["factor"].astype(str)
    if "overall_rank" not in work.columns:
        raise ValueError("scores missing column ’overall_rank’")

    # Parse tree and root tags
    tree_list: List[str] = []
    is_root_list: List[bool] = []
    for f in work["factor"].tolist():
        tree_key, is_root = _parse_tree_and_root_flag(f)
        tree_list.append(tree_key)
        is_root_list.append(is_root)
    work["_tree"] = tree_list
    work["_is_root"] = is_root_list

    # Grouping processing
    selected_rows: List[pd.DataFrame] = []
    for tree_key, g in work.groupby("_tree", sort=False):
        g_sorted = g.sort_values("overall_rank", ascending=True)
        roots = g_sorted[g_sorted["_is_root"]]
        # Select a root (if it exists), giving priority to factor with the smallest value
        root_pick = None
        if len(roots) > 0:
            def _safe_factor_float_root(x):
                try:
                    return float(str(x).strip())
                except Exception:
                    return float('inf')
            roots_tmp = roots.copy()
            roots_tmp["_factor_num_root"] = roots_tmp["factor"].map(_safe_factor_float_root)
            roots_tmp = roots_tmp.sort_values(["_factor_num_root"], ascending=[True])
            root_pick = roots_tmp.iloc[[0]].drop(columns=["_factor_num_root"], errors="ignore")
            root_rank = float(root_pick.iloc[0]["overall_rank"])  # Taking the score of root as the threshold
        else:
            # When there is no root, fall back to the minimum score of the group as the threshold
            root_rank = float(g_sorted.iloc[0]["overall_rank"]) if len(g_sorted) > 0 else float("inf")

        # In the same tree, deduplicate according to the same overall_rank: only the one with the smallest factor value is retained
        def _safe_factor_float_local(x):
            try:
                return float(str(x).strip())
            except Exception:
                return float('inf')

        g_dedup = g_sorted.copy()
        g_dedup["_factor_num"] = g_dedup["factor"].map(_safe_factor_float_local)
        g_dedup = g_dedup.sort_values(["overall_rank", "_factor_num"], ascending=[True, True])
        g_dedup = g_dedup.drop_duplicates(subset=["overall_rank"], keep="first")
        g_dedup = g_dedup.drop(columns=["_factor_num"], errors="ignore")

        eligible = g_dedup[g_dedup["overall_rank"] <= root_rank].copy()
        # Create a numerical sorting auxiliary column for factor to avoid using the unavailable key callback for multi-column sorting When
        eligible["_factor_num"] = eligible["factor"].map(_safe_factor_float_local)

        if KEEP_ROOT_ALWAYS and root_pick is not None and len(root_pick) > 0:
            root_factor_val = str(root_pick.iloc[0]["factor"])
            extras = eligible[eligible["factor"] != root_factor_val]
            extras = extras.sort_values(["overall_rank", "_factor_num"], ascending=[True, True]).head(3)
            extras = extras.drop(columns=["_factor_num"], errors="ignore")
            picked = pd.concat([root_pick, extras], axis=0, ignore_index=True)
        else:
            picked = eligible.sort_values(["overall_rank", "_factor_num"], ascending=[True, True]).head(3)
            picked = picked.drop(columns=["_factor_num"], errors="ignore")
        selected_rows.append(picked)

    if not selected_rows:
        return pd.DataFrame(columns=["factor"])  # empty

    picked_all = pd.concat(selected_rows, axis=0, ignore_index=True)

    # Merge factor_list.csv to bring out information such as expressions
    bdir = _backtest_dir(run_dir)
    select_dir = os.path.join("analysis", run_dir, "select")
    factor_list_path = os.path.join(bdir, "factor_list.csv")
    if not os.path.exists(factor_list_path):
        raise FileNotFoundError(f"factor_list.csv does not exist: {factor_list_path}")

    fl = pd.read_csv(factor_list_path)
    if "factor" not in fl.columns:
        raise ValueError("factor_list.csv missing column ’factor’")
    fl["factor"] = fl["factor"].astype(str)

    merged = fl.merge(picked_all[["factor", "overall_rank"]], on="factor", how="inner")
    # The output table is sorted in ascending order by factor (numeric comparison takes priority, and falls back to string comparison when it cannot be parsed into a numerical value)
    def _safe_factor_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return float('inf')
    merged = merged.sort_values(
        by="factor",
        key=lambda s: s.map(_safe_factor_float),
        ascending=True
    )

    # Output pre-processing numerical accuracy
    num_cols = list(merged.select_dtypes(include="number").columns)
    if "alphaforge_score" in merged.columns and "alphaforge_score" in num_cols:
        num_cols_except_af = [c for c in num_cols if c != "alphaforge_score"]
        if num_cols_except_af:
            merged[num_cols_except_af] = merged[num_cols_except_af].round(2)
        merged["alphaforge_score"] = merged["alphaforge_score"].round(4)
    else:
        merged[num_cols] = merged[num_cols].round(2)

    # Save
    save_path = os.path.join(select_dir, "factor_select.csv")
    os.makedirs(select_dir, exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"[SELECT] Saved CSV: {save_path}")
    return merged


def select_top_factors(run_dir: str,
                       scores: pd.DataFrame,
                       top_k: int,
                       mode: Optional[str] = None,
                       export_stats: bool = True) -> pd.DataFrame:
    """
    Select factors according to mode and output factor_select.csv.
    - dft: Global sorting takes the first top_k;
    - opt: Filter by tree and root thresholds, and get up to 3 for each tree.
    """
    use_mode = (mode or MODE or "dft").strip().lower()
    if use_mode == "opt":
        print("[SELECT] Mode: opt (filter by tree group, each group can take up to 3)")
        selected = _select_by_mode_opt(run_dir, scores)
    else:
        print("[SELECT] Mode: dft (global sorting takes TOP_K)")
        selected = _select_by_mode_dft(run_dir, scores, top_k)

    # Automatically export factor_stats.csv after selection is completed (can be switched on and off)
    if export_stats and isinstance(selected, pd.DataFrame) and not selected.empty:
        try:
            export_factor_stats(run_dir, selected)
        except Exception as e:
            print(f"❌ Export factor_stats.csv failed: {e}")
    return selected


def _load_period_table_prefixed(bdir: str, filename: str, prefix: str) -> Optional[pd.DataFrame]:
    """
    Read the backtest table of a certain period and eliminate [’factor’, ’expression’, ’Expression’, ’alphaforge_score’],
    Prefix the remaining columns with the given prefix and return (retaining ’factor’ for subsequent merging).
    If the file does not exist or the reading fails, None is returned.
    """
    fpath = os.path.join(bdir, filename)
    if not os.path.exists(fpath):
        return None
    try:
        df = normalize_metric_columns(pd.read_csv(fpath))
        if "factor" not in df.columns:
            # Skip directly when it cannot be aligned with the selected factor by row
            print(f"[SELECT] Skip tables without ’factor’ columns: {fpath}")
            return None
        df["factor"] = df["factor"].astype(str)

        drop_cols = {"factor", "expression", "Expression", "alphaforge_score"}
        keep_cols = [c for c in df.columns if c not in drop_cols]

        out = df[["factor"] + keep_cols].copy()
        # Prefix the remaining columns
        rename_map = {c: f"{prefix}{c}" for c in keep_cols}
        out = out.rename(columns=rename_map)
        return out
    except Exception as e:
        print(f"❌ Failed to read or process: {fpath} -> {e}")
        return None


def _generate_factor_summary(factor_stats_df: pd.DataFrame, select_dir: str) -> str:
    """
    Generate summary statistics table factor_sum.csv based on factor_stats data.

    Statistical rules:
    - For IC, RankIC, excess IC related columns: use absolute values for statistics (mean, std, med, min, max)
    - For other numerical columns: use the original value for statistics
    - Automatically exclude non-numeric columns (factor, expression, Expression, alphaforge_score)

    Parameters:
        factor_stats_df: Data of factor_stats.csv
        select_dir: Output directory

    Return:
        Save path
    """
    if factor_stats_df is None or factor_stats_df.empty:
        print("[SELECT] factor_stats data is empty, skip summary statistics")
        return ""

    # Exclude non-numeric columns (factor, expression, Expression, etc.)
    exclude_cols = {"factor", "expression", "Expression", "alphaforge_score"}
    numeric_cols = [col for col in factor_stats_df.columns
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(factor_stats_df[col])]

    if not numeric_cols:
        print("[SELECT] No numerical column found, skip summary statistics")
        return ""

    # Calculate statistical indicators
    summary_data = {}
    for col in numeric_cols:
        series = factor_stats_df[col].dropna()  # Exclude NaN value
        if len(series) == 0:
            continue

        # For columns related to IC, RankIC, and excess IC, use absolute values for statistics
        if any(ic_keyword in col.lower() for ic_keyword in ["ic", "rankic", "Excess ic"]):
            series_for_stats = series.abs()
        else:
            series_for_stats = series

        summary_data[col] = {
            "mean": series_for_stats.mean(),
            "std": series_for_stats.std(),
            "med": series_for_stats.median(),
            "min": series_for_stats.min(),
            "max": series_for_stats.max()
        }

    if not summary_data:
        print("[SELECT] No valid numerical data, skip summary statistics")
        return ""

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data).T
    summary_df = summary_df.round(4)  # Keep 4 decimal places

    # Save file
    save_path = os.path.join(select_dir, "factor_sum.csv")
    summary_df.to_csv(save_path)
    print(f"[SELECT] Saved CSV: {save_path}")
    return save_path


def export_factor_stats(run_dir: str, selected: pd.DataFrame) -> str:
    """
    Based on the selected factor selected (including ’factor’ and the expression column),
    Only read the corresponding backtest_{train,valid,test}.csv according to the table enabled in TABLE_WEIGHTS (weight>0),
    Non-existing files will be automatically ignored; factor/expression/alphaforge_score will be eliminated when merging,
    And add the corresponding prefix train_/valid_/test_ to the remaining columns, and output select/factor_stats.csv.
    Return to save path.
    """
    bdir = _backtest_dir(run_dir)
    select_dir = os.path.join("analysis", run_dir, "select")
    os.makedirs(select_dir, exist_ok=True)

    # Key column type for normalized selected
    work = selected.copy()
    if "factor" not in work.columns:
        raise ValueError("selected missing column ’factor’")
    work["factor"] = work["factor"].astype(str)

    # Only retain factor and the expression column (if it exists) as the left table base
    expr_col = "expression" if "expression" in work.columns else ("Expression" if "Expression" in work.columns else None)
    base_cols = ["factor"] + ([expr_col] if expr_col else [])
    base = work[base_cols].copy()

    # Read and prefix according to the table enabled in TABLE_WEIGHTS
    enabled_tables = {k for k, w in TABLE_WEIGHTS.items() if w}
    period_to_file_prefix = {
        "TRAIN": ("backtest_train.csv", "train_"),
        "VALID": ("backtest_valid.csv", "valid_"),
        "TEST": ("backtest_test.csv", "test_"),
    }

    period_tables: List[pd.DataFrame] = []
    for period in ["TRAIN", "VALID", "TEST"]:
        if period not in enabled_tables:
            continue
        fname, pfx = period_to_file_prefix[period]
        dfp = _load_period_table_prefixed(bdir, fname, pfx)
        if dfp is not None:
            period_tables.append(dfp)

    # Merge table by table (press factor left join)
    out = base
    for t in period_tables:
        out = out.merge(t, on="factor", how="left")

    save_path = os.path.join(select_dir, "factor_stats.csv")
    out.to_csv(save_path, index=False)
    print(f"[SELECT] Saved CSV: {save_path}")

    # If the summary statistics function is enabled, factor_sum.csv is generated
    if EXPORT_FACTOR_SUMMARY:
        try:
            _generate_factor_summary(out, select_dir)
        except Exception as e:
            print(f"❌ Failed to generate summary statistics: {e}")

    return save_path

def main():
    print("🎯 Factor selection module")
    print("=" * 40)
    print(f"RUN_DIR: {RUN_DIR}")
    print(f"METRICS: {list(METRIC_WEIGHTS.keys())}")
    print(f"TABLE_WEIGHTS: {TABLE_WEIGHTS}")
    print(f"TOP_K: {TOP_K}")
    print(f"MODE: {MODE}")
    print(f"EXPORT_FACTOR_SUMMARY: {EXPORT_FACTOR_SUMMARY}")

    try:
        scores = compute_overall_scores(
            run_dir=RUN_DIR,
            metric_weights=METRIC_WEIGHTS,
            table_weights=TABLE_WEIGHTS,
        )
    except Exception as e:
        print(f"❌ Failed to calculate ranking: {e}")
        sys.exit(1)

    try:
        selected = select_top_factors(RUN_DIR, scores, TOP_K, MODE)
    except Exception as e:
        print(f"❌ Failed to output selection result: {e}")
        sys.exit(1)

    # Terminal print result summary: serial number, factor number, expression, rank score
    try:
        print("\n=== Factor selection details ===")
        for i, row in enumerate(selected.itertuples(index=False), start=1):
            factor = getattr(row, 'factor', '')
            expr = getattr(row, 'expression', getattr(row, 'Expression', ''))
            rank_score = getattr(row, 'overall_rank', None)
            # Expression appropriately truncated for readability
            expr_disp = (expr if isinstance(expr, str) else '')
            if len(expr_disp) > 120:
                expr_disp = expr_disp[:117] + '...'
            if isinstance(rank_score, (int, float)):
                print(f"{i:2d}. {factor} | {rank_score:.2f}\n    {expr_disp}")
            else:
                print(f"{i:2d}. {factor} | N/A\n    {expr_disp}")
    except Exception as e:
        print(f"❌ Failed to print details: {e}")

    # has been exported by select_top_factors internally according to export_stats control

    print("✅ Factor selection completed")


if __name__ == "__main__":
    main()
