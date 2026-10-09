# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, List
import sys
import os
import pandas as pd
from backtest.metric_schema import normalize_metric_columns
from pandas.core.indexes.accessors import NoNewAttributesMixin


@dataclass
class DataConfig:
    max_days: int = 120
    max_stocks: Optional[int] = None
    max_backtrack_days: int = 60
    max_future_days: int = 0


@dataclass
class ModeConfig:
    # Operation mode: sgl single factor, mpl multi-factor
    mode: str = ""


@dataclass
class FactorConfigSgl:
    # sgl: Processing a single factor at a time
    factor_dir: str = ""
    factor_name: str = ""
    sp_name: str = ""
    pool_base_dir: str = "pool"
    csv_filename: str = "csv_zoo_final.csv"
    pkl_filename: str = "z_bld_zoo_final.pkl"


@dataclass
class FactorConfigMpl:
    # mpl: batch processing of multiple factors at one time
    factor_dir: str = ""
    factor_name: str = ""
    sp_name_rule: str = ""
    sp_name: str = ""
    pool_base_dir: str = "pool"
    csv_filename: str = "csv_zoo_final.csv"
    pkl_filename: str = "z_bld_zoo_final.pkl"
    selected_factors: List[str] = field(default_factory=list)

    stats_sort_col: str = ""
    descending: bool = True
    stats_limit: Optional[int] = None


@dataclass
class PoolConfig:
    top_percentage: float = 0.2
    use_direction: bool = True
    direction_metric: str = 'rankic'
    direction_lookback_days: int = 60


def parse_cmd_args():
    """Parse command line parameters: -d directory name -n factor name -s stock pool name"""
    args = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-d' and i + 1 < len(sys.argv):
            args['filedir'] = sys.argv[i + 1]
            i += 2
        elif arg == '-n' and i + 1 < len(sys.argv):
            args['factorname'] = sys.argv[i + 1]
            i += 2
        elif arg == '-s' and i + 1 < len(sys.argv):
            args['spname'] = sys.argv[i + 1]
            i += 2
        elif arg == '-m' and i + 1 < len(sys.argv):
            args['mode'] = sys.argv[i + 1]
            i += 2
        elif arg == '--sort' and i + 1 < len(sys.argv):
            args['sort'] = sys.argv[i + 1]
            i += 2
        elif arg == '--desc' and i + 1 < len(sys.argv):
            args['desc'] = sys.argv[i + 1]
            i += 2
        elif arg == '--limit' and i + 1 < len(sys.argv):
            # Only parsing, no rollback; if subsequent conversion fails, an error will be reported and exited
            args['limit'] = sys.argv[i + 1]
            i += 2

        else:
            i += 1
    return args


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    factor_sgl: FactorConfigSgl = field(default_factory=FactorConfigSgl)
    factor_mpl: FactorConfigMpl = field(default_factory=FactorConfigMpl)
    pool: PoolConfig = field(default_factory=PoolConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)

    @property
    def factor(self):
        # Keep the config.factor interface unchanged externally: return to the corresponding configuration according to the mode
        return self.factor_sgl if self.mode.mode == "sgl" else self.factor_mpl

    def __post_init__(self):
        """Post-initialization processing, reading command line parameters"""
        cmd_args = parse_cmd_args()

        # If parameters are specified on the command line, override the default value
        if 'filedir' in cmd_args:
            # Both modes will use the factor directory
            self.factor_sgl.factor_dir = cmd_args['filedir']
            self.factor_mpl.factor_dir = cmd_args['filedir']
        if 'factorname' in cmd_args:
            self.factor_sgl.factor_name = cmd_args['factorname']
        if 'spname' in cmd_args:
            self.factor_sgl.sp_name = cmd_args['spname']

        if 'mode' in cmd_args:
            self.mode.mode = cmd_args['mode']
        if 'sort' in cmd_args:
            self.factor_mpl.stats_sort_col = cmd_args['sort']
        if 'desc' in cmd_args:
            val = str(cmd_args['desc']).strip().lower()
            if val in ('1', 'true', 'yes', 'y', 't'):
                self.factor_mpl.descending = True
            elif val in ('0', 'false', 'no', 'n', 'f'):
                self.factor_mpl.descending = False
            else:
                print("\n[ERROR] parameter --desc only supports true/false")
                sys.exit(1)
        if 'limit' in cmd_args:
            try:
                self.factor_mpl.stats_limit = int(cmd_args['limit']) if cmd_args['limit'] is not None else None
            except Exception:
                print("\n[ERROR] parameter --limit requires integer")
                sys.exit(1)

        # Parameter verification and batch mode preprocessing
        if self.mode.mode not in ("sgl", "mpl"):
            print("\n[ERROR] must pass -m to specify the mode (sgl or mpl)")
            sys.exit(1)

        if self.mode.mode == "sgl":
            # sgl: dir, name, sp are required
            if not self.factor_sgl.factor_dir or not self.factor_sgl.factor_name or not self.factor_sgl.sp_name:
                print("\n[ERROR] sgl mode needs to specify the factor directory (-d), factor name (-n), and stock pool name (-s)")
                sys.exit(1)
            # In order to be compatible with subsequent processes, fill in selected_factors
            self.factor_mpl.selected_factors = []
            # Provide a single factor list to sgl when used outside the context (such as the unified interface)
            # But config.factor will not use this field under sgl
        else:
            # mpl: Requires dir, sp naming rules (supports {factor} placeholder)
            if not self.factor_mpl.factor_dir:
                print("\n[ERROR] mpl mode needs to specify the factor directory (-d)")
                sys.exit(1)
            # If passed in -s, it will be treated as a naming rule in mpl mode
            if 'spname' in cmd_args and cmd_args['spname']:
                self.factor_mpl.sp_name_rule = cmd_args['spname']
            # Read <project_root>/pool/<factor_dir>/factor_stats.csv
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stats_path = os.path.join(project_root, self.factor_mpl.pool_base_dir, self.factor_mpl.factor_dir, "factor_stats.csv")
            if not os.path.exists(stats_path):
                print(f"\n[ERROR] Factor statistics file not found: {stats_path}")
                sys.exit(1)
            try:
                df_stats = normalize_metric_columns(pd.read_csv(stats_path))
            except Exception as e:
                print(f"\n[ERROR] Failed to read factor statistics file: {e}")
                sys.exit(1)

            if "factor" not in df_stats.columns:
                print("\n[ERROR] factor_stats.csv missing column ’factor’")
                sys.exit(1)

            sort_col = self.factor_mpl.stats_sort_col
            if sort_col not in df_stats.columns:
                print(f"\n[ERROR] Specifies the sorting column ’{sort_col}’ does not exist in factor_stats.csv")
                sys.exit(1)

            df_sorted = df_stats.sort_values(by=sort_col, ascending=not self.factor_mpl.descending)
            if self.factor_mpl.stats_limit is not None:
                try:
                    limit = int(self.factor_mpl.stats_limit)
                    df_sorted = df_sorted.head(limit)
                except Exception:
                    pass
            selected = df_sorted["factor"].dropna().astype(str).tolist()
            if not selected:
                print("\n[ERROR] No factors were selected from factor_stats.csv")
                sys.exit(1)
            self.factor_mpl.selected_factors = selected

            # Verify naming rules (from -s)
            # sgl uses sp_name; mpl uses sp_name_rule (passed in through -s)
            if not self.factor_sgl.sp_name and not self.factor_mpl.sp_name_rule:
                print("\n[ERROR] mpl mode needs to specify the stock pool naming rule through -s, such as ’sp_{factor}’")
                sys.exit(1)

        # Print configuration information
        print(f"\n=== Paper Trading CONFIG ===")
        print(f"Factor directory: {self.factor.factor_dir}")
        if self.mode.mode == "sgl":
            print(f"Operating mode: sgl (single factor)")
            print(f"Factor name: {self.factor_sgl.factor_name}")
            print(f"Stock pool name: {self.factor_sgl.sp_name}")
        else:
            print(f"Operation mode: mpl (multi-factor batch)")
            print(f"Sorting column: {self.factor_mpl.stats_sort_col}, descending order: {self.factor_mpl.descending}")
            print(f"Quantity limit: {self.factor_mpl.stats_limit}")
            print(f"Naming rules: {self.factor_mpl.sp_name_rule}")
            print(f"Select the number of factors: {len(self.factor_mpl.selected_factors)}")
            # Preview of several items before printing
            preview = self.factor_mpl.selected_factors[:5]
            print(f"Example factors: {preview} ...")
        print("=" * 30)
