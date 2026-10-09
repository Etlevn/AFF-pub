#!/usr/bin/env python3
"""
QLib factor calculator module - calculate factor values using the same method as the training phase
"""

import pandas as pd
import numpy as np
import warnings
import os
import pickle
from typing import Optional, Dict, Any, List
import torch

# Import AlphaForge related modules
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData

warnings.filterwarnings('ignore')

class QLibFactorCalculator:
    """QLib Factor Calculator - Calculates factor values using the same method as the training phase"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.expressions_cache = {}  # Caching parsed expressions

        print(f"🔧 QLib factor calculator initialization completed")
        print(f"   Calculation mode: training phase recurrence (using AlphaForge Expression.evaluate)")

    def load_expressions_from_pickle(self, pickle_dir: str) -> Dict[str, Expression]:
        """Load Expression object from pickle file"""
        expressions = {}

        if not os.path.exists(pickle_dir):
            print(f"❌ The Pickle directory does not exist: {pickle_dir}")
            return expressions

        # Find all pickle files
        pickle_files = []
        for root, dirs, files in os.walk(pickle_dir):
            for file in files:
                if file.endswith('.pkl'):
                    pickle_files.append(os.path.join(root, file))

        if not pickle_files:
            print(f"❌ No pickle files found in {pickle_dir}")
            return expressions

        print(f"📁 Found {len(pickle_files)} pickle files")

        for pickle_file in pickle_files:
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)

                # Check whether it is a Builders object
                if hasattr(data, 'exprs') and hasattr(data, 'exprs_str'):
                    for expr, expr_str in zip(data.exprs, data.exprs_str):
                        if expr is not None and expr_str is not None:
                            expressions[expr_str] = expr

            except Exception as e:
                print(f"⚠️ Failed to load pickle file {pickle_file}: {e}")
                continue

        print(f"✅ Loading successfully {len(expressions)} Expression objects")
        return expressions

    def calculate_factor(self, expression: str, market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Calculate factor value time series (using the same method as the training phase)"""

        try:
            # Get StockData object
            stock_data = market_data.get('stock_data')
            if stock_data is None:
                print("❌ Missing StockData object")
                return None

            # Find the corresponding Expression object
            expr_obj = None

            # First search from cache
            if expression in self.expressions_cache:
                expr_obj = self.expressions_cache[expression]
            else:
                # Try to load from pickle file
                pickle_dir = market_data.get('pickle_dir')
                if pickle_dir:
                    expressions = self.load_expressions_from_pickle(pickle_dir)
                    self.expressions_cache.update(expressions)
                    expr_obj = expressions.get(expression)

            if expr_obj is None:
                print(f"❌ Unable to find the Expression object corresponding to the expression: {expression[:60]}...")
                return None

            # Use the same evaluate method as in the training phase
            try:
                factor_tensor = expr_obj.evaluate(stock_data)

                # Apply the same normalization as in the training phase
                factor_tensor = normalize_by_day(factor_tensor)
            except Exception as e:
                print(f"❌ Expression evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Convert to DataFrame format
            factor_df = self._tensor_to_dataframe(factor_tensor, stock_data)

            if self.verbose:
                # Single line printing: shape, mean, standard deviation, range
                try:
                    arr = factor_df.to_numpy().ravel()
                    valid = arr[np.isfinite(arr)]
                    if valid.size > 0:
                        mean_v = float(valid.mean())
                        std_v = float(valid.std())
                        min_v = float(valid.min())
                        max_v = float(valid.max())
                        print(
                            f"   ✅ Factor value shape: {factor_df.shape}, statistics: mean {mean_v:.4f}, standard deviation {std_v:.4f}, range[{min_v:.4f}, {max_v:.4f}]"
                        )
                    else:
                        print(f"   ✅ Factor value shape: {factor_df.shape}, statistics: no valid value")
                    # Missing value and zero value statistics (overall and latest date)
                    total_cells = factor_df.size
                    nan_total = int(factor_df.isna().sum().sum())
                    miss_ratio_total = (nan_total / total_cells) if total_cells > 0 else 0.0
                    zero_total = int((factor_df == 0).sum().sum())
                    zero_ratio_total = (zero_total / total_cells) if total_cells > 0 else 0.0
                    latest_dt = factor_df.index.max() if not factor_df.empty else None
                    if latest_dt is not None:
                        latest_row = factor_df.loc[latest_dt]
                        nan_latest = int(latest_row.isna().sum())
                        miss_ratio_latest = nan_latest / latest_row.shape[0] if latest_row.shape[0] > 0 else 0.0
                        zero_latest = int((latest_row == 0).sum())
                        zero_ratio_latest = zero_latest / latest_row.shape[0] if latest_row.shape[0] > 0 else 0.0
                        print(f"   📊 Missing value proportion: global {miss_ratio_total:.4%}({nan_total}/{total_cells}), latest date {miss_ratio_latest:.4%}({nan_latest}/{latest_row.shape[0]})")
                        print(f"   📊 Zero value scale: global {zero_ratio_total:.4%}({zero_total}/{total_cells}), latest date {zero_ratio_latest:.4%}({zero_latest}/{latest_row.shape[0]})")
                    else:
                        print(f"   📊 Missing value proportion: global {miss_ratio_total:.4%}({nan_total}/{total_cells})")
                        print(f"   📊 Zero value scale: global {zero_ratio_total:.4%}({zero_total}/{total_cells})")
                except Exception:
                    print(f"   ✅ Factor value shape: {factor_df.shape}")

            return factor_df

        except Exception as e:
            print(f"❌ QLib calculation failed: {e}")
            return None

    def _tensor_to_dataframe(self, factor_tensor: torch.Tensor, stock_data: StockData) -> pd.DataFrame:
        """Convert PyTorch Tensor to DataFrame (align the valid date window of the training process)"""
        factor_np = factor_tensor.detach().cpu().numpy()

        # In the training process, the effective date is [max_backtrack_days: -max_future_days]
        all_dates = getattr(stock_data, '_dates', [])
        max_backtrack = getattr(stock_data, 'max_backtrack_days', 0)
        max_future = getattr(stock_data, 'max_future_days', 0)
        if isinstance(all_dates, list):
            dates_series = all_dates
        else:
            dates_series = list(all_dates)

        if max_future == 0:
            effective_dates = dates_series[max_backtrack:]
        else:
            effective_dates = dates_series[max_backtrack:-max_future]

        stock_codes = getattr(stock_data, 'stock_codes', getattr(stock_data, '_stock_ids', []))

        # Defensive alignment (evaluate results will not be cropped when they are consistent with the valid date length)
        if factor_np.shape[0] != len(effective_dates):
            min_days = min(factor_np.shape[0], len(effective_dates))
            factor_np = factor_np[:min_days, :]
            effective_dates = effective_dates[:min_days]

        if factor_np.shape[1] != len(stock_codes):
            min_stocks = min(factor_np.shape[1], len(stock_codes))
            factor_np = factor_np[:, :min_stocks]
            stock_codes = stock_codes[:min_stocks]

        return pd.DataFrame(factor_np, index=effective_dates, columns=stock_codes)

    def preload_expressions(self, pickle_dir: str) -> bool:
        """Preload all Expression objects into cache"""
        try:
            expressions = self.load_expressions_from_pickle(pickle_dir)
            self.expressions_cache.update(expressions)
            print(f"✅ Preloading completed, already in cache {len(self.expressions_cache)} expressions")
            return True
        except Exception as e:
            print(f"❌ Preloading failed: {e}")
            return False
