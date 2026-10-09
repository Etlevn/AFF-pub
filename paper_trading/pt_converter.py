# -*- coding: utf-8 -*-
'''
Step 2: Data format conversion module (pt_converter)
'''

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from paper_trading.pt_config import PipelineConfig


class PTConverter:
    def __init__(self):
        self.stock_data = None

    def convert_day01_to_qlib_format(self, df_day01_all: pd.DataFrame, config: PipelineConfig):
        if df_day01_all is None or df_day01_all.empty:
            print("[ERROR] The input daily data is empty")
            return None

        try:
            print("Step 1: Data verification and preprocessing")
            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'tr_code']
            missing_columns = [col for col in required_columns if col not in df_day01_all.columns]
            if missing_columns:
                print(f"[ERROR] Required columns are missing: {missing_columns}")
                return None

            df_day01_all['time'] = pd.to_datetime(df_day01_all['time'])
            stock_codes = df_day01_all['tr_code'].unique().tolist()

            all_dates = df_day01_all['time'].dt.strftime('%Y-%m-%d').unique()
            all_dates = sorted(all_dates)
            if len(all_dates) > config.data.max_days:
                all_dates = all_dates[-config.data.max_days:]

            print("Step 2: Create data matrix")
            n_days = len(all_dates)
            n_stocks = len(stock_codes)

            close_data = np.full((n_days, n_stocks), np.nan)
            open_data = np.full((n_days, n_stocks), np.nan)
            high_data = np.full((n_days, n_stocks), np.nan)
            low_data = np.full((n_days, n_stocks), np.nan)
            volume_data = np.full((n_days, n_stocks), np.nan)

            print("Step 3: Fill in the data matrix")
            with tqdm(total=len(stock_codes), desc="Processing convertible bond data", unit="bond", leave=False) as pbar:
                for i, tr_code in enumerate(stock_codes):
                    stock_df = df_day01_all[df_day01_all['tr_code'] == tr_code].copy()
                    stock_df['date_str'] = stock_df['time'].dt.strftime('%Y-%m-%d')
                    stock_df = stock_df[stock_df['date_str'].isin(all_dates)]

                    for j, date in enumerate(all_dates):
                        date_data = stock_df[stock_df['date_str'] == date]
                        if not date_data.empty:
                            latest_data = date_data.iloc[-1]
                            close_data[j, i] = latest_data['close']
                            open_data[j, i] = latest_data['open']
                            high_data[j, i] = latest_data['high']
                            low_data[j, i] = latest_data['low']
                            volume_data[j, i] = latest_data['volume']

                    pbar.update(1)
                    pbar.set_postfix({"Current convertible bonds": tr_code, "Progress": f"{i+1}/{len(stock_codes)}"})

            class CompatibleStockData:
                def __init__(self, stock_codes, dates, close, open_data, high, low, volume, *, max_backtrack_days: int, max_future_days: int):
                    self.stock_codes = stock_codes
                    self.dates = dates
                    self.close = close
                    self.open = open_data
                    self.high = high
                    self.low = low
                    self.volume = volume
                    self.max_backtrack_days = max_backtrack_days
                    self.max_future_days = max_future_days
                    self._dates = dates
                    self._stock_ids = stock_codes
                    features_data = torch.stack([
                        open_data,
                        close,
                        high,
                        low,
                        volume
                    ], dim=1)
                    self.data = features_data
                    self.n_stocks = close.shape[1]
                    self.n_features = 5
                    effective_days = close.shape[0] - self.max_backtrack_days - self.max_future_days
                    self.n_days = max(effective_days, 1)
                    self.device = torch.device('cpu')

            close_tensor = torch.tensor(close_data, dtype=torch.float32)
            open_tensor = torch.tensor(open_data, dtype=torch.float32)
            high_tensor = torch.tensor(high_data, dtype=torch.float32)
            low_tensor = torch.tensor(low_data, dtype=torch.float32)
            volume_tensor = torch.tensor(volume_data, dtype=torch.float32)

            stock_data = CompatibleStockData(
                stock_codes=stock_codes,
                dates=all_dates,
                close=close_tensor,
                open_data=open_tensor,
                high=high_tensor,
                low=low_tensor,
                volume=volume_tensor,
                max_backtrack_days=config.data.max_backtrack_days,
                max_future_days=config.data.max_future_days,
            )

            print(f"Data shape: {stock_data.data.shape}")
            return stock_data

        except Exception as e:
            print(f"\n[ERROR] Data format conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
