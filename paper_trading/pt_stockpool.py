# -*- coding: utf-8 -*-
'''
Step 5: Stock pool generation module (pt_stockpool)
'''

import pandas as pd
from typing import Optional
from paper_trading.pt_config import PipelineConfig
import numpy as np


class PTStockPool:
    def __init__(self):
        self.stock_pool = None

    def _rolling_rankic_direction(self, factor_df: pd.DataFrame, stock_data, lookback: int) -> tuple[bool, float, int, float, float, float]:
        # Use the cross-sectional rank correlation mean of the historical window [T-K..T-1] factor and the known return [T-K+1..T] to determine the direction
        try:
            if stock_data is None or not hasattr(stock_data, 'close'):
                return True, float('nan'), 0, float('nan'), float('nan'), float('nan')

            # Align index and column
            dates = stock_data.dates
            if isinstance(dates[0], str):
                dates = [pd.to_datetime(d) for d in dates]
            close_df = pd.DataFrame(stock_data.close.numpy(), index=dates, columns=stock_data.stock_codes)

            # Next day’s return (use shift (-1) to align with factors)
            ret_df = close_df.shift(-1) / close_df - 1.0

            # Get history window
            end_date = factor_df.index.max()
            try:
                end_pos = factor_df.index.get_loc(end_date)
            except Exception:
                return True, float('nan'), 0, float('nan'), float('nan'), float('nan')

            start_pos = max(0, end_pos - lookback)
            hist_idx = factor_df.index[start_pos:end_pos]  # [T-K..T-1]

            if len(hist_idx) < 2:
                return True, float('nan'), 0, float('nan'), float('nan'), float('nan')

            f_hist = factor_df.loc[hist_idx]
            r_hist = ret_df.loc[hist_idx]

            # Align columns
            common_cols = f_hist.columns.intersection(r_hist.columns)
            if len(common_cols) == 0:
                return True, float('nan'), 0, float('nan'), float('nan'), float('nan')

            f_hist = f_hist[common_cols]
            r_hist = r_hist[common_cols]

            # Daily Spearman cross-sectional correlation (robust version): if the variance is too small or the effective sample is insufficient, it will be recorded as 0
            vals = []
            for d in hist_idx:
                f_row = f_hist.loc[d]
                r_row = r_hist.loc[d]
                s = pd.concat([f_row, r_row], axis=1).dropna()
                if s.shape[0] < 5:
                    vals.append(0.0)
                    continue
                x = s.iloc[:, 0]
                y = s.iloc[:, 1]
                if np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
                    vals.append(0.0)
                    continue
                corr_val = float(x.corr(y, method='spearman'))
                vals.append(corr_val)

            if len(vals) == 0:
                return True, float('nan'), 0, float('nan'), float('nan'), float('nan')

            mean_rankic = float(np.nanmean(vals))
            std_rankic = float(np.nanstd(vals))
            min_rankic = float(np.nanmin(vals))
            max_rankic = float(np.nanmax(vals))
            n_days_used = int(len(vals))
            return (
                (mean_rankic >= 0 if np.isfinite(mean_rankic) else True),
                mean_rankic,
                n_days_used,
                std_rankic,
                min_rankic,
                max_rankic,
            )
        except Exception:
            return True, float('nan'), 0, float('nan'), float('nan'), float('nan')

    def generate_stock_pool(self, factor_df: pd.DataFrame, config: PipelineConfig, stock_data=None) -> pd.DataFrame | None:
        try:
            if factor_df is None or factor_df.empty:
                print("[ERROR] Factor value data is empty")
                return None
            latest_date = factor_df.index.max()
            latest_values = factor_df.loc[latest_date]
            print(f"[INFO] Generate stock pool - Date: {latest_date}")
            print(f"[INFO] Total number of convertible bonds: {len(latest_values)}")
            valid_values = latest_values.dropna()
            if len(valid_values) == 0:
                print("[ERROR] No valid factor value")
                return None
            print(f"[INFO] Effective number of convertible bonds: {len(valid_values)}")

            # Direction judgment
            if getattr(config.pool, 'use_direction', False):
                metric = getattr(config.pool, 'direction_metric', 'rankic')
                lookback = int(getattr(config.pool, 'direction_lookback_days', 60))
                if metric == 'rankic':
                    is_positive, mean_rankic, n_days_used, std_rankic, min_rankic, max_rankic = self._rolling_rankic_direction(factor_df, stock_data, lookback)
                    mean_str = f"{mean_rankic:.4f}" if np.isfinite(mean_rankic) else 'nan'
                    std_str = f"{std_rankic:.4f}" if np.isfinite(std_rankic) else 'nan'
                    range_str = f"[{min_rankic:.4f}, {max_rankic:.4f}]" if np.isfinite(min_rankic) and np.isfinite(max_rankic) else '[nan, nan]'
                    print(f"[INFO] RankIC_mean={mean_str} | RankIC_std={std_str} | RankIC_range={range_str}")
                    print(f"[INFO] Direction judgment: Standard ={metric} | Look back ={lookback} days | number of days used ={n_days_used} | direction={'Forward (larger value first)' if is_positive else 'Reverse (lower value first)'}")
                else:
                    is_positive = True
                sorted_values = valid_values.sort_values(ascending=not is_positive)
            else:
                sorted_values = valid_values.sort_values(ascending=False)

            top_percentage = float(config.pool.top_percentage)
            select_count = max(1, int(len(sorted_values) * top_percentage))
            print(f"[INFO] Before selection {top_percentage*100}% of convertible bonds, a total of {select_count} only")
            selected_stocks = sorted_values.head(select_count)
            n = len(selected_stocks)
            equal_weight = 100.0 / n
            weights = [round(equal_weight, 2) for _ in range(n)]
            df_sp_new = pd.DataFrame({'tr_code': list(selected_stocks.index), 'mark': weights})
            self.stock_pool = df_sp_new
            self._print_stock_pool_info(df_sp_new, selected_stocks, latest_date)
            print("Stock pool format example:")
            print(df_sp_new)

            # Derive and write the stock pool name when generating the stock pool (without using getattr fallback logic)
            if config.mode.mode == 'sgl':
                sp_name = config.factor_sgl.sp_name
            else:
                # Each batch is generated based on the current factor_name according to rules and covers sp_name to ensure uniqueness
                sp_name = config.factor_mpl.sp_name_rule.format(factor=str(config.factor_mpl.factor_name))
                config.factor_mpl.sp_name = sp_name

            print(f"\n[DONE] Stock pool generated successfully: {sp_name}")
            return df_sp_new
        except Exception as e:
            print(f"[ERROR] Stock pool generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _print_stock_pool_info(self, df_sp_new: pd.DataFrame, selected_stocks: pd.Series, latest_date):
        print("\n" + "="*80)
        print("Stock pool information")
        print("="*80)
        print(f"Generation date: {latest_date}")
        print(f"Stock pool size: {len(df_sp_new)} convertible bonds")
        print(f"Total weight: {df_sp_new['mark'].sum():.2f}")
        print(f"\n stock pool convertible bond list:")
        for i, (_, row) in enumerate(df_sp_new.iterrows(), 1):
            tr_code = row['tr_code']
            mark = row['mark']
            factor_value = selected_stocks[tr_code]
            print(f"{i:2d}. {tr_code}: weight ={mark:.2f}%, factor value ={factor_value:.6f}")
        factor_values = selected_stocks.values
        print(f"\n stock pool statistical information:")
        # Convert to Series for statistics NaN/ zero value, etc.
        vals_s = selected_stocks
        total_cnt = int(vals_s.shape[0])
        nan_cnt = int(vals_s.isna().sum())
        zero_cnt = int((vals_s == 0).sum())
        valid_cnt = total_cnt - nan_cnt
        nan_ratio = (nan_cnt / total_cnt) if total_cnt > 0 else 0.0
        zero_ratio = (zero_cnt / total_cnt) if total_cnt > 0 else 0.0

        print(f"  - Basic statistics:")
        print(f"    Range: [{vals_s.min(skipna=True):.6f}, {vals_s.max(skipna=True):.6f}]")
        print(f"    Mean: {vals_s.mean(skipna=True):.6f}  Standard deviation: {vals_s.std(skipna=True):.6f}")
        print(f"  - Missing/zero value:")
        print(f"    Missing value: {nan_cnt:>5d} / {total_cnt:<5d}  ({nan_ratio:.2%})")
        print(f"    Zero value: {zero_cnt:>5d} / {total_cnt:<5d}  ({zero_ratio:.2%})")
        print(f"    Valid values: {valid_cnt:>5d} / {total_cnt:<5d}  ({(valid_cnt/total_cnt if total_cnt>0 else 0.0):.2%})")
        print(f"  - Symbol distribution:")
        print(f"    Positive value: {(vals_s > 0).sum()}  Negative value: {(vals_s < 0).sum()}")
        print("="*80)
