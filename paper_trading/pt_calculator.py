# -*- coding: utf-8 -*-
'''
Step 4: Factor calculation module (pt_calculator)
'''

import os
import sys
import pandas as pd
from typing import Optional, Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtest.qlib_factor_calculator import QLibFactorCalculator


class PTCalculator:
    def __init__(self):
        self.qlib_calculator = QLibFactorCalculator(verbose=True)
        self.factor_df = None

    def calculate_factor_values(self, factor_info: Dict[str, Any], stock_data, factor_dir: str, config):
        try:
            expression = factor_info.get('expression')
            if isinstance(expression, list):
                expression = expression[0]
            if not expression:
                print("[ERROR] Unable to obtain factor expression")
                return None
            print(f"Factor expression: {expression[:60]}...")
            pickle_dir = os.path.join(project_root, config.factor.pool_base_dir, factor_dir)
            print(f"Pickle directory: {pickle_dir}")
            market_data = { 'stock_data': stock_data, 'pickle_dir': pickle_dir }
            print("Preload Expression object...")
            self.qlib_calculator.preload_expressions(pickle_dir)
            print("Calculate factor value...")
            factor_df = self.qlib_calculator.calculate_factor(expression, market_data)
            if factor_df is None:
                print("[ERROR] Factor calculation failed")
                return None
            self.factor_df = factor_df
            print(f"[INFO] Successfully calculated factor value")
            print(f"Factor value data shape: {factor_df.shape}")
            return factor_df
        except Exception as e:
            print(f"[ERROR] Factor calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
