"""
Factor Processor Module
Factor processing module (only generates and saves factor values, no evaluation/plotting).

Process:
1. Import the factors converted into QLib format, and filter the N names before
2. Import market data (prices/returns)
3. Calculate factor values one by one and save to analysis/<run_dir>/factor_series/
"""

import pandas as pd
import numpy as np
import warnings
import os
from typing import Dict, List, Optional
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import module
from backtest.qlib_factor_calculator import QLibFactorCalculator

warnings.filterwarnings('ignore')

# ============================== [CONFIG] ==============================

# Specify folder configuration
DEFAULT_SPECIFIED_DIR = "test"  # Use the latest folder by default, None=automatically select the latest folder

# Operating mode: ’dft’ | ’opt’
MODE = 'dft'

# Factor quantity configuration
DEFAULT_TOP_N_FACTORS = None    # None represents backtesting all factors

# Backtest time configuration
DEFAULT_START_DATE = "2022-01-01"  # Default backtest start date
DEFAULT_END_DATE = "2025-06-30"    # Default backtest end date

# ======================================================================

def get_latest_out_dir(specified_dir: str = None):
    """
    Get output directory (strict mode)
    - The specified folder must exist; raise an error if it is missing.
    """
    out_root = 'out'

    if specified_dir is None:
        raise FileNotFoundError("The output folder (specified_dir) is not specified, and automatic selection of the latest folder is prohibited.")

    specified_path = os.path.join(out_root, specified_dir)
    if os.path.exists(specified_path) and os.path.isdir(specified_path):
        print(f"✅ Use the specified folder: {specified_dir}")
        return specified_dir

    raise FileNotFoundError(f"The specified folder does not exist: {specified_dir}")

class FactorProcessor:
    """
    Factor Processor - Process Coordinator
    is only responsible for process coordination and does not repeatedly implement specific functions.
    """

    def __init__(self, specified_dir: str = DEFAULT_SPECIFIED_DIR):
        """
        Initialize the factor processor

        Parameters:
        specified_dir: str, the specified folder name, DEFAULT_SPECIFIED_DIR is used by default
        """
        print("🔧 Initializing factor processing module")

        run_dir = get_latest_out_dir(specified_dir)
        analysis_dir = f"analysis/{run_dir}"
        input_csv = f"out/{run_dir}/csv_zoo_final.csv"

        # Check whether the original factor file exists
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"The original factor file does not exist: {input_csv}")

        print(f"✅ Use original factor file: {input_csv}")

        # All subsequent paths are spliced with analysis_dir
        self.analysis_dir = analysis_dir
        self.run_dir = run_dir

        # Initialize each module
        self.factor_calculator = QLibFactorCalculator()

        # Data storage
        self.top_factors = None
        self.market_data = None

        print(f"✅ Factor processing module initialization completed")

    def load_top_factors(self, factor_file: str = None,
                        top_n: Optional[int] = DEFAULT_TOP_N_FACTORS) -> Optional[pd.DataFrame]:
        """
        Step 1: Import the original factor file and filter the top N names

        Parameters:
        factor_file: str, factor file path
        top_n: int, select the first N factors

        Returns:
        DataFrame: the first N factors after screening
        """
        print(f"\n📋 Step 1: Read the original factor file and filter it {top_n}")
        print("=" * 50)

        # If no file is specified, the original factor file is used
        if factor_file is None:
            factor_file = f"out/{self.run_dir}/csv_zoo_final.csv"

        try:
            # Check file existence
            if not os.path.exists(factor_file):
                print(f"❌ The factor file does not exist: {factor_file}")
                return None

            # Read factor data (all read as strings to avoid format loss such as 10.10 -> 10.1)
            factor_df = pd.read_csv(factor_file, dtype=str, keep_default_na=False)
            print(f"✅ Successfully read factor file: {len(factor_df)} factors")

            # Standardized column names
            if 'exprs' in factor_df.columns:
                factor_df = factor_df.rename(columns={'exprs': 'Expression'})
            if 'scores' in factor_df.columns:
                factor_df = factor_df.rename(columns={'scores': 'AlphaForge_Score'})

            # csv_zoo_final.csv format is fixed, directly rename the column
            factor_df = factor_df.rename(columns={'Unnamed: 0': 'index', 'exprs': 'Expression', 'scores': 'AlphaForge_Score'})

            # Convert the rating column to a numerical value for sorting (other columns remain strings to ensure that the factor name format remains unchanged)
            if 'AlphaForge_Score' in factor_df.columns:
                factor_df['AlphaForge_Score'] = pd.to_numeric(factor_df['AlphaForge_Score'], errors='coerce').fillna(0.0)
            # Construct a numerical column for index for sorting
            if 'index' in factor_df.columns:
                factor_df['index_numeric'] = pd.to_numeric(factor_df['index'], errors='coerce')

            # Select sorting logic according to MODE
            if (MODE or 'dft').strip().lower() == 'opt':
                # opt: in ascending order by index
                sort_col = 'index_numeric' if 'index_numeric' in factor_df.columns else 'index'
                factor_df = factor_df.sort_values(sort_col, ascending=True, na_position='last')
                print("🔧 Sorting mode: opt (in ascending order by index)")
            else:
                # dft: According to AlphaForge_Score descending order
                factor_df = factor_df.sort_values('AlphaForge_Score', ascending=False)
                print("🔧 Sorting mode: dft (descending order by AlphaForge_Score)")
            self.top_factors = factor_df if (top_n is None or top_n <= 0) else factor_df.head(top_n)

            print(f"🎯 Number of successfully screened factors: {len(self.top_factors)}")
            for i, (_, row) in enumerate(self.top_factors.iterrows(), 1):
                expr = row.get('Expression', 'N/A')
                score = row.get('AlphaForge_Score', 0)
                print(f"   {i:2d}. {expr[:60]}... (score: {score:.6f})")

            return self.top_factors

        except Exception as e:
            print(f"❌ Failed to read factor file: {e}")
            return None

    def load_bond_data(self, start_date: str = DEFAULT_START_DATE,
                      end_date: str = DEFAULT_END_DATE,
                      max_bonds: int = 1000) -> Optional[Dict]:
        """
        Step 2: Import all convertible bond data 2020-2024

        Parameters:
        start_date: str, start date
        end_date: str, end date
        max_bonds: int, the maximum number of convertible bonds

        Returns:
        dict: Market data dictionary
        """
        print(f"\n📊 Step 2: Read convertible bond data")
        print("=" * 50)

        try:
            print(f"📍 Data period: {start_date} to {end_date}")
            print(f"📍 Maximum number of convertible bonds: {max_bonds}")

            # Load data using QLib mode
            print("🔄 StockData creation method using training process...")

            import os
            import numpy as np
            import pandas as pd
            import torch
            from alphagen_qlib.stock_data import StockData, FeatureType

            qlib_path = os.environ.get('QLIB_PATH', os.path.expanduser('~/.qlib/qlib_data/day1_data_qlib'))

            # Read the instruments list of qlib, limit the quantity
            instruments_file = os.path.join(qlib_path, 'instruments', 'all.txt')
            if not os.path.exists(instruments_file):
                raise FileNotFoundError(f"instruments file not found: {instruments_file}")

            with open(instruments_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            instruments = [line.split('\t')[0] for line in lines]
            if max_bonds:
                instruments = instruments[:max_bonds]

            # Construct the real StockData according to the training process
            stock_data = StockData(
                instrument=instruments,
                start_time=start_date,
                end_time=end_date,
                max_backtrack_days=300,
                max_future_days=5,
                features=list(FeatureType),
                device=torch.device('cpu'),
                raw=True,
                qlib_path=qlib_path,
                freq='day',
            )

            # Derive price_data and return_data from StockData, strictly aligning the effective trading days
            all_dates = stock_data._dates
            if stock_data.max_future_days == 0:
                effective_dates = all_dates[stock_data.max_backtrack_days:]
            else:
                effective_dates = all_dates[stock_data.max_backtrack_days:-stock_data.max_future_days]
            stock_ids = list(stock_data._stock_ids)

            # Take the CLOSE feature and convert it to DataFrame
            close_tensor = stock_data.data[:, FeatureType.CLOSE, :]
            close_effective = close_tensor[stock_data.max_backtrack_days: close_tensor.shape[0]-stock_data.max_future_days]
            close_np = close_effective.detach().cpu().numpy()
            price_df = pd.DataFrame(close_np, index=effective_dates, columns=stock_ids)

            # Future rate of return (consistent with training): t to t+1
            return_df = price_df.shift(-1) / price_df - 1
            return_df = return_df.iloc[:-1]
            price_df = price_df.iloc[:-1]

            # Organize market_data for use in subsequent processes
            market_data = {
                'price_data': price_df,
                'return_data': return_df,
                'stock_data': stock_data,
                'pickle_dir': f"out/{self.run_dir}",
            }

            print("✅ Successfully created real StockData and derived price and yield data")

            if market_data and 'price_data' in market_data and 'return_data' in market_data:
                self.market_data = market_data
                price_shape = market_data['price_data'].shape
                return_shape = market_data['return_data'].shape

                print(f"✅ Convertible bond data successfully loaded:")
                print(f"   📈 Price data: {price_shape[0]} days × {price_shape[1]} convertible bonds")
                print(f"   📊 Yield data: {return_shape[0]} days × {return_shape[1]} convertible bonds")

                return market_data
            else:
                print("❌ Failed to load convertible bond data")
                return None

        except Exception as e:
            print(f"❌ Abnormal loading of convertible bond data: {e}")
            return None

    def save_factors(self) -> Optional[List[Dict]]:
        """
        Step 3: Calculate and save factor values (without any evaluation/plotting)

        Returns:
        List[Dict]: Meta information list of saved factors
        """
        print(f"\n🚀 Step 3: Batch calculation and saving of factor values")
        print("=" * 50)

        if self.top_factors is None:
            print("❌ Please load factor data first")
            return None

        if self.market_data is None:
            print("❌ Please load market data first")
            return None

        print(f"🔄 Start processing {len(self.top_factors)} factors...")

        factor_list = []

        for i, (_, factor_row) in enumerate(self.top_factors.iterrows(), 1):
            # Use the index in the original csv_zoo_final.csv as the factor name
            factor_name = str(factor_row['index'])  # Use original index column
            expression = factor_row.get('Expression', '')
            alphaforge_score = factor_row.get('AlphaForge_Score', 0)

            print(f"\n📊 [{i:2d}/{len(self.top_factors)} Processing factor: {factor_name}")
            print(f"   Expression: {expression[:60] if expression else 'N/A'}...")

            # Check whether the expression is valid
            if not expression:
                print(f"   ❌ The expression is empty, skip")
                continue

            try:
                # Generate factor value
                factor_data = self._generate_factor_values(expression)

                if factor_data is not None and self.market_data is not None:
                    # Save the factor value sequence to CSV
                    try:
                        saved_path = self._save_factor_series(factor_name, factor_data)
                        if saved_path:
                            print(f"   💾 Factor value has been saved: {saved_path}")
                            # Record factor information (excluding savedpath)
                            factor_list.append({
                                'factor': factor_name,
                                'expression': expression,
                                'alphaforge_score': alphaforge_score
                            })
                    except Exception as e:
                        print(f"   ⚠️ Failed to save factor value: {e}")
                else:
                    print(f"   ❌ Factor calculation failed")

            except Exception as e:
                print(f"   ❌ Backtest failed: {e}")
                continue

        print(f"\n✅ Factor saving completed! Processed {len(factor_list)} factors")
        return factor_list

    def run_complete_pipeline(self,
                             factor_file: str = None,
                             top_n: int = DEFAULT_TOP_N_FACTORS,
                             start_date: str = DEFAULT_START_DATE,
                             end_date: str = DEFAULT_END_DATE,
                             max_bonds: int = 1000) -> Optional[pd.DataFrame]:
        """
        Run the complete factor processing process

        Parameters:
        factor_file: str, factor file path
        top_n: int, the first N factors
        start_date: str, start date
        end_date: str, end date
        max_bonds: int, the maximum number of convertible bonds

        Returns:
        DataFrame: Final result table
        """
        print("🎯 Start the complete factor processing process")
        print("=" * 60)

        # Step 1: Read and filter factors
        top_factors = self.load_top_factors(factor_file, top_n)
        if top_factors is None or len(top_factors) == 0:
            print("❌ Process termination: Factor loading failed")
            return None

        # Step 2: Read convertible bond data
        market_data = self.load_bond_data(start_date, end_date, max_bonds)
        if market_data is None:
            print("❌ Process termination: Market data loading failed")
            return None

        # Step 3: Only save factor values
        factor_list = self.save_factors()
        if not factor_list:
            print("❌ Process termination: Failed to save factors")
            return None

        # Generate factor_list.csv and save it to the backtest folder
        results_table = pd.DataFrame(factor_list)
        backtest_dir = os.path.join(self.analysis_dir, "backtest")
        save_path = os.path.join(backtest_dir, "factor_list.csv")
        try:
            os.makedirs(backtest_dir, exist_ok=True)
            results_table.to_csv(save_path, index=False)
            print(f"💾 The factor list has been written: {save_path}")
        except Exception as e:
            print(f"⚠️ Failed to write factor list: {e}")

        return results_table

    def _generate_factor_values(self, expression: str) -> Optional[pd.DataFrame]:
        """
        Generate factor values (based on calculation mode)

        Parameters:
        expression: str, factor expression

        Returns:
        DataFrame: factor value data
        """
        if self.market_data is None:
            return None

        # Use FactorCalculator to calculate factor value
        factor_data = self.factor_calculator.calculate_factor(expression, self.market_data)
        return factor_data

    def _save_factor_series(self, factor_name: str, factor_data: pd.DataFrame) -> Optional[str]:
        """
        Save the factor value time series of a single factor as a CSV file.
        Save path: analysis/<run_dir>/factor_series/<factor_name>.csv
        """
        # Build save directory
        save_dir = os.path.join(self.analysis_dir, 'factor_series')
        os.makedirs(save_dir, exist_ok=True)

        # Build file path and save
        filename = f"{factor_name}.csv"
        save_path = os.path.join(save_dir, filename)
        factor_data.to_csv(save_path)
        return save_path


# Convenience function
def run_factor_analysis_pipeline(factor_file: str = None,
                                top_n: int = DEFAULT_TOP_N_FACTORS,
                                start_date: str = DEFAULT_START_DATE,
                                end_date: str = DEFAULT_END_DATE,
                                max_bonds: int = 1000,
                                specified_dir: str = None) -> tuple:
    """
    Convenient factor analysis process function

    Parameters:
    factor_file: str, factor file path
    top_n: int, the first N factors
    start_date: str, start date
    end_date: str, end date
    max_bonds: int, the maximum number of convertible bonds
    specified_dir: str, the specified folder name, if it is None, use the latest folder

    Returns:
    tuple: (results_table, processor)
    """
    processor = FactorProcessor(specified_dir=specified_dir)
    results_table = processor.run_complete_pipeline(
        factor_file=factor_file,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
        max_bonds=max_bonds
    )

    return results_table, processor

if __name__ == "__main__":
    """Main program entry"""
    print("🎯 Factor processing module")
    print("=" * 40)

    # Run the complete process (only factor values are generated and saved)
    results, processor = run_factor_analysis_pipeline(
        max_bonds=1000,
        specified_dir=DEFAULT_SPECIFIED_DIR  # Use default folder configuration
    )

    if results is not None:
        print(f"\n🎉 Factor processing completed!")

    else:
        print("❌ Analysis failed!")
