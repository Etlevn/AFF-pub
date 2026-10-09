# -*- coding: utf-8 -*-
"""
Unified pipeline module (pt_pipeline): steps 1-5
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any

from paper_trading.pt_fetcher import PTFetcher
from paper_trading.pt_converter import PTConverter
from paper_trading.pt_factor import PTFactorReader
from paper_trading.pt_calculator import PTCalculator
from paper_trading.pt_stockpool import PTStockPool
from paper_trading.pt_mask import PTMask
from paper_trading.pt_config import PipelineConfig


class TradingPipeline:
    def __init__(self):
        self.fetcher = PTFetcher()
        self.data_converter = PTConverter()
        self.factor_reader = PTFactorReader()
        self.factor_calculator = PTCalculator()
        self.stock_pool_generator = PTStockPool()
        self.mask_processor = PTMask()

    def run(
        self,
        df_info: pd.DataFrame,
        config: PipelineConfig,
        tp_info: pd.DataFrame = None,
        on_pool=None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
        print("\n" + "=" * 80)
        print("Step 1: Obtain daily data")
        print("=" * 80)
        df_day01_all = self.fetcher.get_all_day01_data(df_info=df_info, config=config)

        print("\n" + "=" * 80)
        print("Step 2: Data format conversion")
        print("=" * 80)
        stock_data = self.data_converter.convert_day01_to_qlib_format(df_day01_all, config=config)

        mode = config.mode.mode
        if mode == 'sgl':
            print("\n" + "=" * 80)
            print("Step 3: Factor reading")
            print("=" * 80)
            factor_info = self.factor_reader.load_top_factor(
                config=config,
            )
            if factor_info is None:
                print("[ERROR] Factor reading failed and the process terminated")
                return None, None, None, None

            print("\n" + "=" * 80)
            print("Step 4: Factor calculation")
            print("=" * 80)
            factor_df = self.factor_calculator.calculate_factor_values(
                factor_info=factor_info,
                stock_data=stock_data,
                factor_dir=config.factor.factor_dir,
                config=config,
            )
            if factor_df is None:
                print("[ERROR] Factor calculation failed and the process terminated")
                return None, None, None, None

            print("\n" + "=" * 80)
            print("Step 5: Mask processing")
            print("=" * 80)
            factor_df = self.mask_processor.process_factor_with_mask(factor_df, tp_info, df_info, stock_data)

            print("\n" + "=" * 80)
            print("Step 6: Stock pool generation")
            print("=" * 80)
            df_sp_new = self.stock_pool_generator.generate_stock_pool(
                factor_df=factor_df,
                config=config,
                stock_data=stock_data,
            )

            if config.mode.mode == 'sgl':
                sp_name = config.factor_sgl.sp_name
            else:
                sp_name = config.factor_mpl.sp_name

            if on_pool is not None and df_sp_new is not None:
                on_pool(sp_name, df_sp_new.copy())

            try:
                latest_date = factor_df.index.max()
                latest_row = factor_df.loc[latest_date]
                codes = list(df_sp_new['tr_code']) if df_sp_new is not None else []
                selected_vals = latest_row[codes].dropna() if len(codes) > 0 else latest_row.iloc[0:0]
                n = len(codes)
                valid_ratio = (len(selected_vals) / n) if n > 0 else 0.0
                if len(selected_vals) > 0:
                    mean_v = float(selected_vals.mean())
                    std_v = float(selected_vals.std())
                    min_v = float(selected_vals.min())
                    max_v = float(selected_vals.max())
                else:
                    mean_v = float('nan'); std_v = float('nan'); min_v = float('nan'); max_v = float('nan')
                line = (
                    f"{sp_name} | size={n} | mean={mean_v:.6f} | std={std_v:.6f} | "
                    f"range=[{min_v:.6f}, {max_v:.6f}] | valid={valid_ratio:.2%}"
                )
                print("\n single generation completed: only 1 individual stock pool")
                print("=" * 100)
                print(line)
                print("=" * 100 + "\n")
            except Exception:
                pass

            return df_sp_new, factor_df, factor_info, sp_name
        else:
            df_sp_new_last: Optional[pd.DataFrame] = None
            factor_df_last: Optional[pd.DataFrame] = None
            factor_info_last: Optional[Dict[str, Any]] = None
            sp_name_last: Optional[str] = None
            pool_summaries: list[str] = []

            selected = list(getattr(config.factor, 'selected_factors', []))
            print("\n" + "=" * 80)
            print(f"Steps 3-6: Batch processing {len(selected)} factors")
            print("=" * 80)

            for idx, factor_name in enumerate(selected, start=1):
                print("\n\n" + "=" * 100+"\n")
                print(f"[Batch {idx}/{len(selected)}] FACTOR: {factor_name}")
                print("\n"+"=" * 100+"\n\n")

                # Only set the factor name of the current batch to reuse the reading/calculation logic
                # Synchronously write the current batch factor name to the mpl configuration
                config.factor_mpl.factor_name = factor_name
                setattr(config.factor, 'factor_name', factor_name)

                # 3. Read
                print("Step 3: Factor reading")
                factor_info = self.factor_reader.load_top_factor(
                    config=config,
                )
                if factor_info is None:
                    print(f"[WARN] skip factor {factor_name}: Failed to read")
                    continue

                # 4. Calculation
                print("Step 4: Factor calculation")
                factor_df = self.factor_calculator.calculate_factor_values(
                    factor_info=factor_info,
                    stock_data=stock_data,
                    factor_dir=config.factor.factor_dir,
                    config=config,
                )
                if factor_df is None:
                    print(f"[WARN] skip factor {factor_name}: Calculation failed")
                    continue

                # 5. Mask
                print("Step 5: Mask processing")
                factor_df = self.mask_processor.process_factor_with_mask(factor_df, tp_info, df_info, stock_data)

                # 6. Stock pool
                print("Step 6: Stock pool generation")
                df_sp_new = self.stock_pool_generator.generate_stock_pool(
                    factor_df=factor_df,
                    config=config,
                    stock_data=stock_data,
                )

                df_sp_new_last = df_sp_new
                factor_df_last = factor_df
                factor_info_last = factor_info
                sp_name_last = config.factor_mpl.sp_name

                if on_pool is not None and df_sp_new is not None:
                    on_pool(sp_name_last, df_sp_new.copy())

                try:
                    latest_date = factor_df.index.max()
                    latest_row = factor_df.loc[latest_date]
                    codes = list(df_sp_new['tr_code']) if df_sp_new is not None else []
                    selected_vals = latest_row[codes].dropna() if len(codes) > 0 else latest_row.iloc[0:0]
                    n = len(codes)
                    valid_ratio = (len(selected_vals) / n) if n > 0 else 0.0
                    if len(selected_vals) > 0:
                        mean_v = float(selected_vals.mean())
                        std_v = float(selected_vals.std())
                        min_v = float(selected_vals.min())
                        max_v = float(selected_vals.max())
                    else:
                        mean_v = float('nan'); std_v = float('nan'); min_v = float('nan'); max_v = float('nan')
                    line = (
                        f"{sp_name_last} | size={n} | mean={mean_v:.6f} | std={std_v:.6f} | "
                        f"range=[{min_v:.6f}, {max_v:.6f}] | valid={valid_ratio:.2%}"
                    )
                    pool_summaries.append(line)
                except Exception:
                    pass

            try:
                print(f"\n batch generation completed: total {len(pool_summaries)} Individual stock pool")
                print("=" * 100)
                for s in pool_summaries:
                    print(s)
                print("=" * 100 + "\n")
            except Exception:
                pass

            return df_sp_new_last, factor_df_last, factor_info_last, sp_name_last
