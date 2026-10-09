"""
Factor Backtest Module - Main process module
Factor backtesting module - integrates all backtesting functions

Extract and rewrite from high_freq_factor_anal.ipynb:
- factor_backtest: Single factor backtest function
- backtest_all_factors: Batch factor backtest function

Main functions:
1. Set global time range configuration
2. Set backtest directory
3. Integrate functions such as data import, factor analysis, and drawing
4. Provide a complete backtest process
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom module
from backtest.factor_data import FactorData
from backtest.factor_utils import FactorUtils
from backtest.factor_analysis import FactorAnalysis

warnings.filterwarnings('ignore')

# ============================== [CONFIG] ==============================

# Backtest directory configuration
DEFAULT_BACKTEST_DIR = "test"  # Default backtest directory

# Time range configuration
DEFAULT_PLOT_START_DATE = "2022-01-01"  # Plot start date (data import also uses this range)
DEFAULT_PLOT_END_DATE = "2025-06-30"    # Drawing end date

# Training period configuration
DEFAULT_TRAIN_START_DATE = "2022-01-01"  # Training period start date
DEFAULT_TRAIN_END_DATE = "2024-12-31"    # Training period end date

# Validation period configuration (optional)
DEFAULT_VAL_START_DATE = None  # Validation period start date, None indicates that the validation period is not used
DEFAULT_VAL_END_DATE = None    # Validation period end date

# Test period configuration
DEFAULT_TEST_START_DATE = "2025-01-01"  # Test period start date
DEFAULT_TEST_END_DATE = "2025-06-30"    # Test period end date

# Factor quantity configuration
DEFAULT_TOP_N_FACTORS = None    # None represents backtesting all factors
DEFAULT_GROUP_N = 5         # Default group number

# Operating mode: ’dft’ | ’opt’
MODE = 'dft'

# Mask configuration
USE_VALID_MASK = True       # Whether to enable data_valid mask processing

# Drawing switch
DEFAULT_ENABLE_PLOTS = True # Whether to save the drawing

# ======================================================================

def get_latest_backtest_dir(specified_dir: str = None):
    """
    Get backtest directory (strict mode)
    - No longer automatically falls back to the latest directory; it must exist explicitly, otherwise an error will be reported.
    """
    out_root = 'out'

    if specified_dir is None:
        raise FileNotFoundError("The backtest directory (backtest_dir) is not specified.")

    specified_path = os.path.join(out_root, specified_dir)
    if os.path.exists(specified_path) and os.path.isdir(specified_path):
        print(f"✅ Use the specified backtest directory: {specified_dir}")
        return specified_dir

    raise FileNotFoundError(f"The specified backtest directory does not exist: {specified_dir}")


class FactorBacktest:
    """
    Factor backtest main process class
    """

    def __init__(self,
                 backtest_dir: str = DEFAULT_BACKTEST_DIR,
                 plot_start_date: str = DEFAULT_PLOT_START_DATE,
                 plot_end_date: str = DEFAULT_PLOT_END_DATE,
                 train_start_date: str = DEFAULT_TRAIN_START_DATE,
                 train_end_date: str = DEFAULT_TRAIN_END_DATE,
                 val_start_date: str = DEFAULT_VAL_START_DATE,
                 val_end_date: str = DEFAULT_VAL_END_DATE,
                 test_start_date: str = DEFAULT_TEST_START_DATE,
                 test_end_date: str = DEFAULT_TEST_END_DATE,
                 top_n_factors: int = DEFAULT_TOP_N_FACTORS,
                 group_n: int = DEFAULT_GROUP_N,
                 use_valid_mask: bool = USE_VALID_MASK,
                 enable_plots: bool = DEFAULT_ENABLE_PLOTS):
        """
        Initialization factor backtester

        Parameters:
        - backtest_dir: str, backtest directory name
        - plot_start_date: str, drawing start date
        - plot_end_date: str, drawing end date
        - train_start_date: str, training period start date
        - train_end_date: str, training period end date
        - val_start_date: str, validation period start date
        - val_end_date: str, validation period end date
        - test_start_date: str, test period start date
        - test_end_date: str, test period end date
        - top_n_factors: int, the number of backtest factors
        - group_n: int, number of groups
        - enable_plots: bool, whether to draw
        """
        print("🔧 Initialization factor backtest module")

        # Get backtest directory
        self.backtest_dir = get_latest_backtest_dir(backtest_dir)
        self.analysis_dir = f"analysis/{self.backtest_dir}"

        # Set time range
        self.plot_start_date = plot_start_date
        self.plot_end_date = plot_end_date
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.top_n_factors = top_n_factors
        self.group_n = group_n
        self.use_valid_mask = use_valid_mask
        self.enable_plots = enable_plots

        # Initialize each module
        self.factor_data = None
        self.factor_utils = None
        self.factor_analysis = None

        # Data storage
        self.market_data = None
        self.top_factors = None
        self.data_valid = None

        print(f"✅ Factor backtest module initialization completed")
        print(f"📊 Backtest directory: {self.backtest_dir}")
        print(f"📅 Data range: {self.plot_start_date} to {self.plot_end_date}")
        print(f"🎯 Number of backtest factors: {self.top_n_factors}")
        print(f"📈 Number of groups: {self.group_n}")

    def load_market_data(self, qlib_path: str = None, max_instruments: int = 1000):
        """
        Load market data

        Parameters:
        - qlib_path: str, QLib data path
        - max_instruments: int, maximum number of stocks

        Returns:
        - bool: Whether loaded successfully
        """
        print(f"\n🚀 Step 1: Load market data")
        print("=" * 60)

        try:
            # Use the plot time range as the data import range
            self.factor_data = FactorData(qlib_path=qlib_path, load_ohlcv=False)  # OHLCV is not loaded during initialization
            self.market_data = self.factor_data.load_all_data(
                start_time=self.plot_start_date,
                end_time=self.plot_end_date,
                max_instruments=max_instruments,
                include_filters=False,  # Filter data is not loaded by default
                load_ohlcv=True  # Load close and open data for constructing the mask
            )

            # Initialize FactorUtils (only use close data)
            self.factor_utils = FactorUtils(
                data_close=self.factor_data.data_close,
                data_open=self.factor_data.data_open,
                data_volume=self.factor_data.data_volume,
                data_high=self.factor_data.data_high,
                data_low=self.factor_data.data_low,
                ST=self.factor_data.ST,
                new=self.factor_data.new,
                yjl=self.factor_data.yjl,
                marketcap=self.factor_data.marketcap
            )

            # Initialize FactorAnalysis
            self.factor_analysis = FactorAnalysis(self.factor_utils)

            # Construct data_valid mask (if enabled)
            if self.use_valid_mask:
                print(f"\n🔍 Construct data_valid mask...")
                self.data_valid = self.factor_data.load_valid_mask(
                    start_time=self.plot_start_date,
                    end_time=self.plot_end_date
                )
                if self.data_valid is not None:
                    print("✅ data_valid mask construction completed")
                else:
                    print("⚠️ data_valid mask construction failed, the mask will not be used")

            print("✅ Market data loading completed")
            return True

        except Exception as e:
            print(f"❌ Market data loading failed: {e}")
            return False

    def load_top_factors(self):
        """
        Load the first N factors

        Returns:
        - bool: Whether loaded successfully
        """
        print(f"\n📊 Step 2: Batch backtest factor")
        print("=" * 60)

        try:
            factor_file = f"out/{self.backtest_dir}/csv_zoo_final.csv"

            if not os.path.exists(factor_file):
                print(f"❌ The factor file does not exist: {factor_file}")
                return False

            # Read factor data
            factor_df = pd.read_csv(factor_file, dtype=str, keep_default_na=False)
            print(f"✅ Successfully read factor file: {len(factor_df)} factors")

            # Standardized column names
            if 'exprs' in factor_df.columns:
                factor_df = factor_df.rename(columns={'exprs': 'expression'})
            if 'scores' in factor_df.columns:
                factor_df = factor_df.rename(columns={'scores': 'alphaforge_score'})

            # Check the actual number of factor sequence files
            factor_series_dir = os.path.join(self.analysis_dir, "factor_series")
            if os.path.exists(factor_series_dir):
                existing_factor_files = [f for f in os.listdir(factor_series_dir) if f.endswith('.csv')]
                existing_factor_count = len(existing_factor_files)
                print(f"📁 Discover {existing_factor_count} factor sequence files")

                # Adjust the number of factors to match the actual existing files (only when top_n_factors is a positive integer)
                if isinstance(self.top_n_factors, int) and self.top_n_factors > 0:
                    adjusted_factor_count = min(self.top_n_factors, existing_factor_count)
                    if adjusted_factor_count < self.top_n_factors:
                        print(f"⚠️ Request backtest {self.top_n_factors} factors, but only {existing_factor_count} factor sequence files")
                        print(f"🔄 Automatically adjust to before backtesting {adjusted_factor_count} factors")
                        self.top_n_factors = adjusted_factor_count
            else:
                print(f"⚠️ The factor sequence directory does not exist: {factor_series_dir}")
                return False

            # csv_zoo_final.csv format is fixed, directly rename the column
            factor_df = factor_df.rename(columns={'Unnamed: 0': 'index', 'exprs': 'expression', 'scores': 'alphaforge_score'})

            # Numericalize columns that may be used for sorting
            if 'alphaforge_score' in factor_df.columns:
                factor_df['alphaforge_score'] = pd.to_numeric(factor_df['alphaforge_score'], errors='coerce').fillna(0.0)
            if 'index' in factor_df.columns:
                # Try to digitize index, which is used for numerical sorting. The original string that cannot be converted is kept and placed at the end.
                index_numeric = pd.to_numeric(factor_df['index'], errors='coerce')
                factor_df['index_numeric'] = index_numeric

            # According to MODE selection sorting logic: dft->alphaforge_score descending order; opt->index ascending order
            # Construct index numerical column
            if 'index' in factor_df.columns and 'index_numeric' not in factor_df.columns:
                factor_df['index_numeric'] = pd.to_numeric(factor_df['index'], errors='coerce')

            mode = (MODE or 'dft').strip().lower()
            if mode == 'opt':
                sort_col = 'index_numeric' if 'index_numeric' in factor_df.columns else 'index'
                asc = True
                print("🔧 Sorting mode: opt (in ascending order by index)")
            else:
                sort_col = 'alphaforge_score' if 'alphaforge_score' in factor_df.columns else 'index_numeric'
                asc = False if sort_col == 'alphaforge_score' else True
                print("🔧 Sorting mode: dft (descending order by alphaforge_score)")

            if sort_col is None:
                raise ValueError("Unable to determine the sorting column, please check whether the csv_zoo_final.csv column name is complete")

            factor_df = factor_df.sort_values(sort_col, ascending=asc, na_position='last')

            if self.top_n_factors is None or (isinstance(self.top_n_factors, int) and self.top_n_factors <= 0):
                self.top_factors = factor_df
            else:
                self.top_factors = factor_df.head(self.top_n_factors)

            print(f"🎯 Number of successfully screened factors: {len(self.top_factors)}")
            return True

        except Exception as e:
            print(f"❌ Factor loading failed: {e}")
            return False

    def factor_backtest(self, factor_name: str, expression: str, alphaforge_score: float = None):
        """
        Single factor backtest

        Parameters:
        - factor_name: str, factor name
        - expression: str, factor expression
        - alphaforge_score: float, AlphaForge score

        Returns:
        - DataFrame: Backtest results
        """

        try:
            # Read factor data
            factor_path = os.path.join(self.analysis_dir, "factor_series", f"{factor_name}.csv")
            if not os.path.exists(factor_path):
                print(f"❌ The factor file does not exist: {factor_path}")
                return None

            factor_series = pd.read_csv(factor_path, index_col=0)
            factor_series.index = pd.to_datetime(factor_series.index, errors="ignore")

            # Aligned with market data
            factor_series = factor_series.reindex(index=self.factor_data.data_close.index,
                                                columns=self.factor_data.data_close.columns)

            # Apply data_valid mask (if enabled)
            if self.use_valid_mask and self.data_valid is not None:
                factor_series = self.factor_utils.apply_valid_mask(factor_series, self.data_valid)

            # Set time range
            train_period = (self.train_start_date, self.train_end_date) if self.train_start_date and self.train_end_date else None
            val_period = (self.val_start_date, self.val_end_date) if self.val_start_date and self.val_end_date else None
            test_period = (self.test_start_date, self.test_end_date) if self.test_start_date and self.test_end_date else None

            # Perform factor analysis
            result_df = self.factor_analysis.factor_analyse_simple(
                factor=factor_series,
                train_period=train_period,
                val_period=val_period,
                test_period=test_period,
                N=self.group_n,
                factor_name=factor_name,
                expression=expression,
                alphaforge_score=alphaforge_score
            )

            # Drawing (optional)
            if self.enable_plots:
                factor_dict = self.factor_utils.factor_dict(factor_series.loc[pd.to_datetime(self.plot_start_date):pd.to_datetime(self.plot_end_date)])
                factor_dict = self.factor_utils.factor_std(self.factor_utils.winsorize(factor_dict))
                self.factor_analysis.plot_return(
                    factor=factor_dict,
                    N=self.group_n,
                    save_dir=self.analysis_dir,
                    factor_name=factor_name
                )

            return result_df

        except Exception as e:
            print(f"❌ Factor backtest failed: {e}")
            return None

    def backtest_all_factors(self, align: bool = False):
        """
        Batch factor backtest

        Parameters:
        - align: bool, whether to align the data

        Returns:
        - Dict: Dictionary of backtest results
        """


        if self.top_factors is None:
            print("❌ Please load factor data first")
            return None

        results = {}

        for i, (_, factor_row) in enumerate(self.top_factors.iterrows(), 1):
            # Use the index in the original csv_zoo_final.csv as the factor name
            factor_name = str(factor_row['index'])  # Use original index column
            expression = factor_row.get('expression', factor_row.get('exprs', 'N/A'))
            alphaforge_score = factor_row.get('alphaforge_score', factor_row.get('scores', None))

            print(f"\n📊 [{i:2d}/{len(self.top_factors)}] {factor_name}")

            result = self.factor_backtest(factor_name, expression, alphaforge_score)
            if result is not None:
                results[factor_name] = result

        return results

    def run_complete_backtest(self, qlib_path: str = None, max_instruments: int = 1000, align: bool = False):
        """
        Run the complete backtest process

        Parameters:
        - qlib_path: str, QLib data path
        - max_instruments: int, maximum number of stocks
        - align: bool, whether to align the data

        Returns:
        - Dict: Backtest results
        """
        print("🎯 Start the complete factor backtesting process")
        print("=" * 60)

        # Step 1: Load data
        if not self.load_market_data(qlib_path, max_instruments):
            print("❌ Process termination: Data loading failed")
            return None

        # Step 2: Batch backtesting
        if not self.load_top_factors():
            print("❌ Process termination: Factor data loading failed")
            return None

        results = self.backtest_all_factors(align)

        # Step 3: Save and analyze the results
        if results:
            print(f"\n💾 Step 3: Save analysis results")
            print("=" * 60)
            # Unify the factor number of factor_list.csv as the standard and repair the inconsistency of 117 vs 117.0
            try:
                factor_list_path = os.path.join(self.analysis_dir, "backtest", "factor_list.csv")
                if os.path.exists(factor_list_path):
                    factor_list_df = pd.read_csv(factor_list_path)
                    factor_list_df['factor'] = factor_list_df['factor'].astype(str)

                    def normalize_id(s: str) -> str:
                        try:
                            t = str(s).strip()
                            v = float(t)
                            if v.is_integer():
                                return str(int(v))
                            return t
                        except Exception:
                            return str(s)

                    # Mapping: Normalized value -> factor_list original string
                    norm_to_label = {}
                    for lbl in factor_list_df['factor'].tolist():
                        nk = normalize_id(lbl)
                        if nk not in norm_to_label:
                            norm_to_label[nk] = lbl

                    # Convert the key of results to a label consistent with factor_list
                    rebased_results = {}
                    for k, v in results.items():
                        nk = normalize_id(str(k))
                        label = norm_to_label.get(nk, str(k))
                        # If there are duplicate names, the first one will be retained first. Subsequent overwriting of the same name will not affect the preservation.
                        rebased_results[label] = v

                    self.factor_analysis.save_backtest_results_by_period(rebased_results, self.analysis_dir, self.backtest_dir)
                else:
                    # Fallback: Save according to the original result when there is no factor_list.csv
                    self.factor_analysis.save_backtest_results_by_period(results, self.analysis_dir, self.backtest_dir)
            except Exception as e:
                print(f"⚠️ Result remapping failed, use the original result to save: {e}")
                self.factor_analysis.save_backtest_results_by_period(results, self.analysis_dir, self.backtest_dir)
            self.factor_analysis.analyze_backtest_statistics(self.analysis_dir, self.backtest_dir)
            print("✅ Result saving and analysis completed")

        if results:
            print(f"\n🎉 Backtest completed! Processed in total {len(results)} factors")
        else:
            print("\n❌ Backtest failed")

        return results


# Convenience function
def run_backtest(backtest_dir: str = DEFAULT_BACKTEST_DIR,
                plot_start_date: str = DEFAULT_PLOT_START_DATE,
                plot_end_date: str = DEFAULT_PLOT_END_DATE,
                train_start_date: str = DEFAULT_TRAIN_START_DATE,
                train_end_date: str = DEFAULT_TRAIN_END_DATE,
                val_start_date: str = DEFAULT_VAL_START_DATE,
                val_end_date: str = DEFAULT_VAL_END_DATE,
                test_start_date: str = DEFAULT_TEST_START_DATE,
                test_end_date: str = DEFAULT_TEST_END_DATE,
                top_n_factors: int = DEFAULT_TOP_N_FACTORS,
                group_n: int = DEFAULT_GROUP_N,
                qlib_path: str = None,
                max_instruments: int = 1000,
                align: bool = False,
                use_valid_mask: bool = USE_VALID_MASK,
                enable_plots: bool = DEFAULT_ENABLE_PLOTS):
    """
    Convenient backtest function

    Parameters:
    - backtest_dir: str, backtest directory
    - plot_start_date: str, drawing start date
    - plot_end_date: str, drawing end date
    - train_start_date: str, training period start date
    - train_end_date: str, training period end date
    - val_start_date: str, validation period start date
    - val_end_date: str, validation period end date
    - test_start_date: str, test period start date
    - test_end_date: str, test period end date
    - top_n_factors: int, the number of backtest factors
    - group_n: int, number of groups
    - qlib_path: str, QLib data path
    - max_instruments: int, maximum number of stocks
    - align: bool, whether to align the data
    - enable_plots: bool, whether to draw

    Returns:
    - Dict: Backtest results
    """
    backtest = FactorBacktest(
        backtest_dir=backtest_dir,
        plot_start_date=plot_start_date,
        plot_end_date=plot_end_date,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        val_start_date=val_start_date,
        val_end_date=val_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        top_n_factors=top_n_factors,
        group_n=group_n,
        use_valid_mask=use_valid_mask,
        enable_plots=enable_plots
    )

    return backtest.run_complete_backtest(qlib_path, max_instruments, align)


# Test function
if __name__ == "__main__":
    print("🧪 Test factor backtest module...")

    # Run the backtest using the default configuration at the top of the file
    results = run_backtest()

    if results:
        print(f"\n✅ Test completed! Successful backtest {len(results)} factors")
    else:
        print("\n❌ test failed")
