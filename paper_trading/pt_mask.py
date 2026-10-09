# -*- coding: utf-8 -*-
'''
Paper-trading mask module (pt_mask): processing suspended convertible bond mask
'''

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class PTMask:
    """
    Paper-trading mask processing class
    Used to process masking functions such as suspended convertible bonds and listing intervals.
    """

    def __init__(self):
        self.tp_info = None  # Trading suspension information
        self.cb_info = None  # Basic information on convertible bonds
        self.mask_data = None  # Mask data

    def create_trading_mask(self,
                          stock_codes: List[str],
                          dates: List[str],
                          current_date: str = None,
                          stock_data=None) -> Optional[pd.DataFrame]:
        """
        Create transaction mask

        Parameters:
        - stock_codes: List[str], convertible bond code list
        - dates: List[str], date list
        - current_date: str, current date, if it is None, use the latest date

        Returns:
        - DataFrame: Mask matrix (1 = tradable, 0 = not tradable)
        """
        if not stock_codes or not dates:
            print("[ERROR] The convertible bond code or date list is empty")
            return None

        try:
            # Convert date format
            dates = [pd.to_datetime(date) for date in dates]
            if current_date:
                current_date = pd.to_datetime(current_date)
            else:
                current_date = max(dates)

            print(f"[INFO] Create transaction mask - number of convertible bonds: {len(stock_codes)}, date quantity: {len(dates)}")
            print(f"[INFO] Current date: {current_date.strftime('%Y-%m-%d')}")

            # Create mask matrix
            mask_df = pd.DataFrame(1, index=dates, columns=stock_codes, dtype='int8')

            # 1. Processing suspension mask
            if self.tp_info is not None:
                mask_df = self._apply_tp_mask(mask_df, current_date)

            # 2. Process listing interval mask
            if self.cb_info is not None:
                mask_df = self._apply_listing_mask(mask_df)

            # 3. Processing data validity mask (based on price data)
            mask_df = self._apply_data_validity_mask(mask_df, stock_data)

            self.mask_data = mask_df
            valid_ratio = mask_df.values.mean()
            print(f"[INFO] Mask creation completed - effective sample proportion: {valid_ratio:.4%}")

            return mask_df

        except Exception as e:
            print(f"[ERROR] Failed to create transaction mask: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_tp_mask(self, mask_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
        """
        Apply suspension mask (based on the underlying stock suspension information)

        Parameters:
        - mask_df: DataFrame, original mask matrix
        - current_date: Timestamp, current date

        Returns:
        - DataFrame: Matrix after applying stop mask
        """
        if self.tp_info is None or self.cb_info is None or not {"bond_code", "stock_code"}.issubset(self.cb_info.columns):
            print("[WARNING] The trading suspension information or basic convertible bond information is empty, skip the trading suspension mask")
            return mask_df

        try:
            # Get the latest date of suspension information (use the latest date instead of the specified date)
            latest_tp = self.tp_info.iloc[-1]  # Get the last row (latest date)
            tp_date = latest_tp['time']
            tp_codes = latest_tp['tp_codes']

            print(f"[INFO] Use suspension information date: {tp_date.strftime('%Y-%m-%d')}")
            print(f"[INFO] Number of suspended stocks: {len(tp_codes)}")

            if not tp_codes:
                print("[INFO] There is no suspension of the underlying stocks on the latest date")
                return mask_df

            # Create a mapping from the underlying stock code to the convertible bond code
            # Extract the corresponding relationship between the underlying stock code and the convertible bond code from cb_info
            stock_to_bond_map = {}
            for _, row in self.cb_info.iterrows():
                bond_code = row['bond_code']
                stock_code = row['stock_code']

                # Convert to instrument code format
                if len(bond_code) == 6:
                    if bond_code.startswith('1'):  # Shenzhen convertible bonds
                        bond_tr_code = f"{bond_code}.SZ"
                    else:  # Shanghai Stock Exchange Convertible Bonds
                        bond_tr_code = f"{bond_code}.SH"

                    # Stock code format conversion (assuming the format is 000001 or 600000)
                    if len(stock_code) == 6:
                        if stock_code.startswith('6'):  # Shanghai stock market
                            stock_tr_code = f"{stock_code}.SH"
                        else:  # Shenzhen stock market
                            stock_tr_code = f"{stock_code}.SZ"

                        stock_to_bond_map[stock_tr_code] = bond_tr_code

            print(f"[INFO] Create mapping from underlying stocks to convertible bonds - mapping quantity: {len(stock_to_bond_map)}")

            # Find the convertible bonds corresponding to the suspended stocks
            suspended_bonds = []
            for tp_stock in tp_codes:
                if tp_stock in stock_to_bond_map:
                    bond_code = stock_to_bond_map[tp_stock]
                    suspended_bonds.append(bond_code)

            print(f"[INFO] The number of convertible bonds corresponding to the suspended stocks: {len(suspended_bonds)}")
            if suspended_bonds:
                print(f"[INFO] Example of suspended convertible bonds: {suspended_bonds[:5]}")

            # Apply suspension mask to convertible bonds
            for bond_code in suspended_bonds:
                if bond_code in mask_df.columns:
                    mask_df.loc[:, bond_code] = 0

            # Statistics of the actual number of masked convertible bonds
            actual_suspended = len([code for code in suspended_bonds if code in mask_df.columns])
            print(f"[INFO] Apply suspension mask - actual number of suspended convertible bonds: {actual_suspended}")

            return mask_df

        except Exception as e:
            print(f"[ERROR] Failed to apply suspension mask: {e}")
            import traceback
            traceback.print_exc()
            return mask_df

    def _apply_listing_mask(self, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply listing interval mask

        Parameters:
        - mask_df: DataFrame, original mask matrix

        Returns:
        - DataFrame: Matrix after applying listing interval mask
        """
        if self.cb_info is None or not {"bond_code", "subscription_date"}.issubset(self.cb_info.columns):
            return mask_df

        try:
            # Create convertible bond listing information mapping
            cb_map = {}
            for _, row in self.cb_info.iterrows():
                code = row['bond_code']
                # Convert to instrument code format (assuming the format is 123456.SH or 123456.SZ)
                if len(code) == 6:
                    # Determine the exchange according to the bond code (this needs to be adjusted according to the actual situation)
                    if code.startswith('1'):  # Shenzhen convertible bonds
                        tr_code = f"{code}.SZ"
                    else:  # Shanghai Stock Exchange Convertible Bonds
                        tr_code = f"{code}.SH"
                    cb_map[tr_code] = {
                        'start_date': pd.to_datetime(row['subscription_date'], unit='ms') if pd.notna(row['subscription_date']) else None,
                        'end_date': None  # Assuming that convertible bonds will not be delisted
                    }

            # Apply listing interval mask
            for code in mask_df.columns:
                if code in cb_map:
                    start_date = cb_map[code]['start_date']
                    if start_date is not None:
                        # Dates before the listing date are set as non-tradable
                        mask_df.loc[mask_df.index < start_date, code] = 0

            print(f"[INFO] Apply listing interval mask - process convertible bond quantity: {len(cb_map)}")

            return mask_df

        except Exception as e:
            print(f"[ERROR] Failed to apply listing interval mask: {e}")
            return mask_df

    def _apply_data_validity_mask(self, mask_df: pd.DataFrame, stock_data=None) -> pd.DataFrame:
        """
        Apply data validity mask (based on price data)

        Parameters:
        - mask_df: DataFrame, original mask matrix
        - stock_data: Stock data object, containing OHLCV data

        Returns:
        - DataFrame: Matrix after applying data validity mask
        """
        if stock_data is None:
            print("[WARNING] Stock data not provided, skip data validity mask")
            return mask_df

        try:
            # Check whether there is necessary OHLCV data
            if not hasattr(stock_data, 'open') or not hasattr(stock_data, 'close') or not hasattr(stock_data, 'volume'):
                print("[WARNING] The stock data lacks the necessary OHLCV field, skipping the data validity mask")
                return mask_df

            # Create a tradable day mask (not suspended on the day: open/close is not empty and volume>0)
            # Use direct attribute access, clearer and more reliable
            open_data = stock_data.open  # Opening price
            close_data = stock_data.close  # Closing price
            volume_data = stock_data.volume  # Trading volume

            # Convert to DataFrame format for processing
            dates = stock_data.dates
            stock_codes = stock_data.stock_codes

            # Ensure the date format is consistent
            if isinstance(dates[0], str):
                dates = [pd.to_datetime(date) for date in dates]

            # Convert to numpy array and create DataFrame
            open_df = pd.DataFrame(open_data.numpy(), index=dates, columns=stock_codes)
            close_df = pd.DataFrame(close_data.numpy(), index=dates, columns=stock_codes)
            volume_df = pd.DataFrame(volume_data.numpy(), index=dates, columns=stock_codes)

            # Create tradable day mask
            data_tradable = (
                (~open_df.isna()) &
                (~close_df.isna()) &
                (volume_df > 0)
            ).astype('int8')

            # Ensure index and column alignment
            common_dates = mask_df.index.intersection(data_tradable.index)
            common_codes = mask_df.columns.intersection(data_tradable.columns)

            if len(common_dates) > 0 and len(common_codes) > 0:
                # Apply tradable day mask
                mask_df.loc[common_dates, common_codes] = (
                    mask_df.loc[common_dates, common_codes] &
                    data_tradable.loc[common_dates, common_codes]
                ).astype('int8')

            print(f"[INFO] Apply Data Validity Mask - Process Date Quantity: {len(common_dates)}")
            print(f"[INFO] Valid sample proportion of tradable day masks: {data_tradable.loc[common_dates, common_codes].values.mean():.4%}")

            return mask_df

        except Exception as e:
            print(f"[ERROR] Failed to apply data validity mask: {e}")
            import traceback
            traceback.print_exc()
            return mask_df

    def filter_factor_data(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Use mask to filter factor data

        Parameters:
        - factor_df: DataFrame, factor data matrix

        Returns:
        - DataFrame: filtered factor data
        """
        if self.mask_data is None:
            print("[WARNING] Mask data is not created, return original factor data")
            return factor_df

        try:
            # Ensure index and column alignment
            common_dates = factor_df.index.intersection(self.mask_data.index)
            common_codes = factor_df.columns.intersection(self.mask_data.columns)

            if len(common_dates) == 0 or len(common_codes) == 0:
                print("[WARNING] There is no intersection between factor data and mask data")
                return factor_df

            # Apply mask
            filtered_df = factor_df.loc[common_dates, common_codes].copy()
            mask_subset = self.mask_data.loc[common_dates, common_codes]

            # Set the location with mask 0 to NaN
            filtered_df = filtered_df.where(mask_subset == 1)

            print(f"[INFO] Factor data filtering completed")
            print(f"  Original data: {factor_df.shape}")
            print(f"  Filtered data: {filtered_df.shape}")
            print(f"  Effective sample proportion: {filtered_df.notna().values.mean():.4%}")

            return filtered_df

        except Exception as e:
            print(f"[ERROR] Filter factor data failed: {e}")
            return factor_df

    def process_factor_with_mask(self, factor_df: pd.DataFrame, tp_info: pd.DataFrame = None, cb_info: pd.DataFrame = None, stock_data=None) -> pd.DataFrame:
        """
        Unified mask processing interface: create masks and filter factors

        Parameters:
        - factor_df: DataFrame, factor data matrix
        - tp_info: DataFrame, trading suspension information (optional)
        - cb_info: DataFrame, convertible bond information (optional)

        Returns:
        - DataFrame: processed factor data
        """
        if factor_df is None or factor_df.empty:
            print("[WARNING] Factor data is empty, skip mask processing")
            return factor_df

        try:
            print("[INFO] Start mask processing...")

            # 1. Load data (if provided)
            if tp_info is not None:
                if not tp_info.empty:
                    self.tp_info = tp_info.copy()
                    # Convert time format
                    self.tp_info['time'] = pd.to_datetime(self.tp_info['time'])
                    print(f"[INFO] Load suspension information: {len(self.tp_info)} records")

            if cb_info is not None:
                if not cb_info.empty:
                    self.cb_info = cb_info.copy()
                    print(f"[INFO] Load basic information of convertible bonds: {len(self.cb_info)} convertible bonds")

            # 2. Create transaction mask
            stock_codes = factor_df.columns.tolist()
            dates = factor_df.index.tolist()

            # Ensure that the index is of type DatetimeIndex
            if not isinstance(factor_df.index, pd.DatetimeIndex):
                factor_df.index = pd.to_datetime(factor_df.index)
                dates = factor_df.index.tolist()

            current_date = factor_df.index.max()

            mask_df = self.create_trading_mask(
                stock_codes=stock_codes,
                dates=dates,
                current_date=current_date,
                stock_data=stock_data
            )

            # 3. Apply mask filter factor data
            if mask_df is not None:
                factor_df = self.filter_factor_data(factor_df)

            print("[INFO] Mask processing completed")
            return factor_df

        except Exception as e:
            print(f"[ERROR] Mask processing failed: {e}")
            import traceback
            traceback.print_exc()
            return factor_df
