"""
Factor Data Module
Factor data module - use the same data import method as notebook

Provides the same data import function as high_freq_factor_anal.ipynb:
- Support for external incoming data paths
- Support custom time range
- Supports multiple data types (closing price, opening price, turnover, highest price, lowest price, etc.)
- Supports filtering data of ST shares, new shares, market value, premium rate, etc.
"""

import os
import qlib
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple
from qlib.data import D
from qlib.config import REG_CN
from qlib.data.filter import NameDFilter, ExpressionDFilter

warnings.filterwarnings('ignore')


class FactorData:
    """
    Factor data loader - uses the same data import method as notebook
    """

    def __init__(self, qlib_path: Optional[str] = None, region: str = REG_CN, load_ohlcv: bool = False):
        """
        Initialize factor data loader

        Parameters:
        - qlib_path: str, QLib data path, if it is None, use the default path
        - region: str, data area, default REG_CN
        - load_ohlcv: bool, whether to load OHLCV data (open, volume, high, low), default False
        """
        self.qlib_path = qlib_path or os.environ.get('QLIB_PATH')
        if not self.qlib_path:
            raise ValueError('Set QLIB_PATH or pass an explicit qlib_path')
        self.region = region
        self.load_ohlcv = load_ohlcv
        self.qlib_initialized = False

        # Data storage
        self.instruments = None
        self.data_close = None
        self.data_open = None
        self.data_volume = None
        self.data_high = None
        self.data_low = None
        self.ST = None
        self.new = None
        self.marketcap = None
        self.yjl = None

        # Initialize QLib
        self._init_qlib()

    def _init_qlib(self):
        """Initialize QLib"""
        try:
            qlib.init(provider_uri=self.qlib_path, region=self.region)
            self.qlib_initialized = True
            print(f"✅ QLib was initialized successfully, using the data path: {self.qlib_path}")
        except Exception as e:
            print(f"❌ QLib initialization failed: {e}")
            self.qlib_initialized = False

    def load_instruments(self, max_instruments: Optional[int] = None) -> List[str]:
        """
        Load stock pool

        Parameters:
        - max_instruments: int, the maximum number of stocks, None means loading all

        Returns:
        - List[str]: Stock code list
        """
        if not self.qlib_initialized:
            raise RuntimeError("QLib is not initialized, please check the data path")

        try:
            all_txt = os.path.join(self.qlib_path, 'instruments', 'all.txt')
            if not os.path.exists(all_txt):
                raise FileNotFoundError(f"instruments file does not exist: {all_txt}")

            with open(all_txt, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            self.instruments = [line.split('\t')[0] for line in lines]

            if max_instruments:
                self.instruments = self.instruments[:max_instruments]

            print(f"✅ Successfully loaded stock pool: {len(self.instruments)} only stocks")
            return self.instruments

        except Exception as e:
            print(f"❌ Failed to load stock pool: {e}")
            return []

    def load_basic_data(self, start_time: str, end_time: str,
                       instruments: Optional[List[str]] = None,
                       features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load basic price data

        Parameters:
        - start_time: str, start time
        - end_time: str, end time
        - instruments: List[str], stock list, None means using the loaded stock pool
        - features: List[str], feature list, None means using the default feature

        Returns:
        - Dict[str, pd.DataFrame]: data dictionary
        """
        if not self.qlib_initialized:
            raise RuntimeError("QLib is not initialized, please check the data path")

        if instruments is None:
            if self.instruments is None:
                self.load_instruments()
            instruments = self.instruments

        if features is None:
            if self.load_ohlcv:
                features = ['$close', '$open', '$volume']  # '$high', '$low'
            else:
                features = ['$close']

        print(f"📊 Start loading basic data...")
        print(f"   Time range: {start_time} to {end_time}")
        print(f"   Number of shares: {len(instruments)}")
        print(f"   Feature list: {features}")

        data_dict = {}

        try:
            for feature in features:
                print(f"🔄 Loading {feature}...")

                # Determine the storage variable name according to the feature name
                if feature == '$close':
                    var_name = 'data_close'
                elif feature == '$open':
                    var_name = 'data_open'
                elif feature == '$volume':
                    var_name = 'data_volume'
                elif feature == '$high':
                    var_name = 'data_high'
                elif feature == '$low':
                    var_name = 'data_low'
                else:
                    var_name = f'data_{feature.replace("$", "")}'

                # Load data
                data = D.features(instruments, [feature],
                                start_time=start_time,
                                end_time=end_time,
                                freq='day').unstack().T.droplevel(0)

                # Store data
                setattr(self, var_name, data)
                data_dict[var_name] = data

                # Calculate the percentage of non-null values
                total_values = data.shape[0] * data.shape[1]
                non_null_count = data.count().sum()
                non_null_percentage = (non_null_count / total_values * 100) if total_values > 0 else 0

                print(f"   ✅ Successfully obtained {feature} Data: {data.shape} | Non-null value: {non_null_percentage:.2f}%")

            return data_dict

        except Exception as e:
            print(f"❌ Failed to load basic data: {e}")
            return {}

    def load_filter_data(self, start_time: str, end_time: str,
                        instruments: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load filtered data (ST shares, new shares, market value, premium rate, etc.)

        Parameters:
        - start_time: str, start time
        - end_time: str, end time
        - instruments: List[str], stock list, None means using the loaded stock pool

        Returns:
        - Dict[str, pd.DataFrame]: filter data dictionary
        """
        if not self.qlib_initialized:
            raise RuntimeError("QLib is not initialized, please check the data path")

        if instruments is None:
            if self.instruments is None:
                self.load_instruments()
            instruments = self.instruments

        print(f"🔍 Start loading filtered data...")

        filter_data = {}

        try:
            # ST stock data
            print("🔄 Loading ST stock data...")
            self.ST = D.features(instruments, ['$ST'],
                               start_time=start_time,
                               end_time=end_time,
                               freq='day').unstack().T.droplevel(0)
            filter_data['ST'] = self.ST
            print("   ✅ Successfully obtained ST stock data")

            # New stock data
            print("🔄 Loading new stock data...")
            self.new = D.features(instruments, ['$new'],
                                start_time=start_time,
                                end_time=end_time,
                                freq='day').unstack().T.droplevel(0)
            filter_data['new'] = self.new
            print("   ✅ Successfully obtained new stock data")

            # Market capitalization data
            print("🔄 Loading market capitalization data...")
            self.marketcap = D.features(instruments, ['$marketcap'],
                                      start_time=start_time,
                                      end_time=end_time,
                                      freq='day').unstack().T.droplevel(0)
            filter_data['marketcap'] = self.marketcap
            print("   ✅ Successfully obtained market value data")

            # Premium rate data
            print("🔄 Loading premium rate data...")
            self.yjl = D.features(instruments, ['$CONVERTIBLEPREMIUMRATE'],
                                start_time=start_time,
                                end_time=end_time,
                                freq='day').unstack().T.droplevel(0)
            filter_data['yjl'] = self.yjl
            print("   ✅ Successfully obtained premium rate data")

            print("✅ All filtered data loading is completed")
            return filter_data

        except Exception as e:
            print(f"❌ Failed to load filtered data: {e}")
            return {}

    def load_all_data(self, start_time: str, end_time: str,
                     instruments: Optional[List[str]] = None,
                     max_instruments: Optional[int] = None,
                     include_filters: bool = False,
                     load_ohlcv: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all data (basic data + filtered data)

        Parameters:
        - start_time: str, start time
        - end_time: str, end time
        - instruments: List[str], stock list, None means automatic loading
        - max_instruments: int, maximum number of stocks
        - include_filters: bool, whether it contains filtered data
        - load_ohlcv: bool, whether to load OHLCV data, None means using the instance default value

        Returns:
        - Dict[str, pd.DataFrame]: All data dictionaries
        """
        print("🚀 Start loading all data...")

        # Load stock pool
        if instruments is None:
            self.load_instruments(max_instruments)
            instruments = self.instruments

        # Temporarily set load_ohlcv parameters
        if load_ohlcv is not None:
            original_load_ohlcv = self.load_ohlcv
            self.load_ohlcv = load_ohlcv

        # Load basic data
        basic_data = self.load_basic_data(start_time, end_time, instruments)

        # Restore original settings
        if load_ohlcv is not None:
            self.load_ohlcv = original_load_ohlcv

        # Load filter data
        filter_data = {}
        if include_filters:
            filter_data = self.load_filter_data(start_time, end_time, instruments)

        # Merge all data
        all_data = {**basic_data, **filter_data}

        return all_data

    def load_valid_mask(self, start_time: str, end_time: str,
                       instruments: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Construct data_valid mask (listing interval ∩ tradable days)

        Parameters:
        - start_time: str, start time
        - end_time: str, end time
        - instruments: List[str], stock list, None means using the loaded stock pool

        Returns:
        - DataFrame: data_valid mask (1=valid, 0=invalid)
        """
        if not self.qlib_initialized:
            raise RuntimeError("QLib is not initialized, please check the data path")

        if instruments is None:
            if self.instruments is None:
                self.load_instruments()
            instruments = self.instruments

        print(f"🔍 Start constructing data_valid mask...")
        print(f"   Time range: {start_time} to {end_time}")
        print(f"   Number of shares: {len(instruments)}")

        try:
            # 1) Read listing interval information (all.txt)
            all_txt_path = os.path.join(self.qlib_path, 'instruments', 'all.txt')
            if not os.path.exists(all_txt_path):
                print(f"❌ all.txt file does not exist: {all_txt_path}")
                return None

            inst_df = pd.read_csv(
                all_txt_path,
                sep='\t',
                header=None,
                names=['instrument', 'start_date', 'end_date'],
                dtype={'instrument': str, 'start_date': str, 'end_date': str}
            )
            inst_df['start_date'] = pd.to_datetime(inst_df['start_date'], format='%Y-%m-%d', errors='coerce')
            inst_df['end_date'] = pd.to_datetime(inst_df['end_date'], format='%Y-%m-%d', errors='coerce')

            # 2) listing interval table (same dimension as data_close)
            dates = self.data_close.index
            stocks = self.data_close.columns
            data_listed = pd.DataFrame(0, index=dates, columns=stocks, dtype='int8')

            inst_map = {r.instrument: (r.start_date, r.end_date) for r in inst_df.itertuples(index=False)}
            for code in stocks:
                if code not in inst_map:
                    continue
                s, e = inst_map[code]
                if pd.isna(s):
                    continue
                if pd.isna(e):
                    e = dates.max()
                mask = (dates >= s) & (dates <= e)
                data_listed.loc[mask, code] = 1

            # 3) tradable daily table (non-trading suspension on the day: open/close is not empty and volume>0)
            data_tradable = (
                (~self.data_open.isna()) &
                (~self.data_close.isna()) &
                (self.data_volume > 0)
            ).astype('int8')

            # 4) Final effective mask: Listing range ∩ Tradable on the same day
            data_valid = (data_listed & data_tradable).astype('int8')

            print(f"✅ data_valid mask construction completed: {data_valid.shape}")
            print(f"   Effective sample proportion: {data_valid.values.mean():.4%}")

            return data_valid

        except Exception as e:
            print(f"❌ Failed to construct data_valid mask: {e}")
            return None

    def get_data_summary(self) -> Dict[str, any]:
        """
        Obtain data summary information

        Returns:
        - Dict[str, any]: Data summary
        """
        summary = {
            'instruments_count': len(self.instruments) if self.instruments else 0,
            'data_loaded': {}
        }

        # Check whether various data have been loaded
        data_types = {
            'data_close': self.data_close,
            'data_open': self.data_open,
            'data_volume': self.data_volume,
            'data_high': self.data_high,
            'data_low': self.data_low,
            'ST': self.ST,
            'new': self.new,
            'marketcap': self.marketcap,
            'yjl': self.yjl
        }

        for name, data in data_types.items():
            if data is not None:
                # Calculate the percentage of non-null values
                total_values = data.shape[0] * data.shape[1]
                non_null_count = data.count().sum()
                non_null_percentage = (non_null_count / total_values * 100) if total_values > 0 else 0

                # Formatted date display (only displays the date, not the time)
                if len(data.index) > 0:
                    start_date = data.index[0].strftime('%Y-%m-%d') if hasattr(data.index[0], 'strftime') else str(data.index[0])
                    end_date = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
                    date_range = f"{start_date} to {end_date}"
                else:
                    date_range = "No data"

                summary['data_loaded'][name] = {
                    'shape': data.shape,
                    'date_range': date_range,
                    'non_null_percentage': non_null_percentage
                }
            else:
                summary['data_loaded'][name] = None

        return summary

    def print_data_summary(self):
        """Print data summary"""
        summary = self.get_data_summary()

        print("\n" + "="*50)
        print("📊 Data summary")
        print("="*50)
        print(f"Number of shares: {summary['instruments_count']}")
        print("\n loaded data:")

        for name, info in summary['data_loaded'].items():
            if info is not None:
                print(f"  ✅ {name}: {info['shape']} | {info['date_range']} | Non-null value: {info['non_null_percentage']:.2f}%")
            else:
                print(f"  ❌ {name}: not loaded")
        print("="*50)


# Convenience function
def load_factor_data(qlib_path: Optional[str] = None,
                    start_time: str = '2020-01-01',
                    end_time: str = '2025-06-30',
                    max_instruments: Optional[int] = None,
                    include_filters: bool = False,
                    load_ohlcv: bool = False) -> FactorData:
    """
    Convenient data loading function

    Parameters:
    - qlib_path: str, QLib data path
    - start_time: str, start time
    - end_time: str, end time
    - max_instruments: int, maximum number of stocks
    - include_filters: bool, whether to include filtered data, default False
    - load_ohlcv: bool, whether to load OHLCV data, default False

    Returns:
    - FactorData: factor data object
    """
    factor_data = FactorData(qlib_path=qlib_path, load_ohlcv=load_ohlcv)
    factor_data.load_all_data(
        start_time=start_time,
        end_time=end_time,
        max_instruments=max_instruments,
        include_filters=include_filters,
        load_ohlcv=load_ohlcv
    )
    return factor_data


# Test function
if __name__ == "__main__":
    # Test data loading
    print("🧪 Test factor data loading...")

    try:
        # Load data using default parameters
        factor_data = load_factor_data(
            start_time='2023-01-01',
            end_time='2023-12-31',
            max_instruments=None,  # Load all stocks
            include_filters=True
        )

        # Print data summary
        factor_data.print_data_summary()

        print("\n✅ Test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
