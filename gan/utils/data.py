from alphagen_generic.features import open_
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *

import os
QLIB_PATH = os.environ.get("QLIB_PATH")
if QLIB_PATH is None:
    raise ValueError(
        "Environment variable QLIB_PATH is not set"
    )

def get_data_by_year(
    train_start = 2010,train_end=2019,valid_year=2020,test_year =2021,
    instruments=None, target=None,freq=None,
                    ):

    from gan.utils import load_pickle,save_pickle
    # from gan.utils.qlib import get_data_my
    get_data_my = StockData

    train_dates=(f"{train_start}-01-01", f"{train_end}-12-31")
    val_dates=(f"{valid_year}-01-01", f"{valid_year}-12-31")
    test_dates=(f"{test_year}-01-01", f"{test_year}-12-31")

    train_start,train_end = train_dates
    valid_start,valid_end = val_dates
    valid_head_start = f"{valid_year-2}-01-01"
    test_start,test_end = test_dates
    test_head_start = f"{test_year-2}-01-01"

    name = instruments + '_pkl_' + str(target).replace('/','_').replace(' ','') + '_' + freq
    name = f"{name}_{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"

    # First load the complete data set and obtain a unified stock list
    try:
        data_all = load_pickle(f'pkl/{name}/data_all.pkl')
    except:
        print('Complete data does not exist, load complete data first')
        data_all = get_data_my(instruments, train_start, test_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        os.makedirs(f"pkl/{name}",exist_ok=True)
        save_pickle(data_all,f'pkl/{name}/data_all.pkl')

    # Get a unified stock list
    unified_stocks = data_all._stock_ids

    try:
        data = load_pickle(f'pkl/{name}/data.pkl')
        data_valid = load_pickle(f'pkl/{name}/data_valid.pkl')
        data_valid_withhead = load_pickle(f'pkl/{name}/data_valid_withhead.pkl')
        data_test = load_pickle(f'pkl/{name}/data_test.pkl')
        data_test_withhead = load_pickle(f'pkl/{name}/data_test_withhead.pkl')
    except:
        print('Data not exist, load from qlib')
        # Reload each data set using a unified stock list
        data = get_data_my(unified_stocks, train_start, train_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_valid = get_data_my(unified_stocks, valid_start, valid_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_valid_withhead = get_data_my(unified_stocks,valid_head_start, valid_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_test = get_data_my(unified_stocks, test_start, test_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_test_withhead = get_data_my(unified_stocks, test_head_start, test_end,raw = True,qlib_path = QLIB_PATH,freq=freq)

        save_pickle(data,f'pkl/{name}/data.pkl')
        save_pickle(data_valid,f'pkl/{name}/data_valid.pkl')
        save_pickle(data_valid_withhead,f'pkl/{name}/data_valid_withhead.pkl')
        save_pickle(data_test,f'pkl/{name}/data_test.pkl')
        save_pickle(data_test_withhead,f'pkl/{name}/data_test_withhead.pkl')

    return data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name

def get_data_by_year_with_dates(
    train_start: str, train_end: str,
    valid_start: str, valid_end: str,
    test_start: str, test_end: str,
    instruments=None, target=None, freq=None,
):
    """
    Use date-specific data loading functions to avoid sliding window cache confusion

    Args:
        train_start: Training start date (YYYY-MM-DD)
        train_end: Training end date (YYYY-MM-DD)
        valid_start: Verification start date (YYYY-MM-DD)
        valid_end: Verification end date (YYYY-MM-DD)
        test_start: Test start date (YYYY-MM-DD)
        test_end: Test end date (YYYY-MM-DD)
        instruments: Stock pool
        target: target variable
        freq: frequency
    """
    from gan.utils import load_pickle, save_pickle
    import pandas as pd
    get_data_my = StockData

    # Calculate head date (historical data for validation and testing)
    valid_head_start = (pd.Timestamp(valid_start) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    test_head_start = (pd.Timestamp(test_start) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    # Construct a unique cache name containing all specific dates
    name = instruments + '_pkl_' + str(target).replace('/','_').replace(' ','') + '_' + freq
    name = f"{name}_{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"

    # First load the complete data set and obtain a unified stock list
    try:
        data_all = load_pickle(f'pkl/{name}/data_all.pkl')
    except:
        print(f'The complete data does not exist, load the complete data first: {train_start} to {test_end}')
        data_all = get_data_my(instruments, train_start, test_end, raw=True, qlib_path=QLIB_PATH, freq=freq)
        os.makedirs(f"pkl/{name}", exist_ok=True)
        save_pickle(data_all, f'pkl/{name}/data_all.pkl')

    # Get a unified stock list
    unified_stocks = data_all._stock_ids

    try:
        data = load_pickle(f'pkl/{name}/data.pkl')
        data_valid = load_pickle(f'pkl/{name}/data_valid.pkl')
        data_valid_withhead = load_pickle(f'pkl/{name}/data_valid_withhead.pkl')
        data_test = load_pickle(f'pkl/{name}/data_test.pkl')
        data_test_withhead = load_pickle(f'pkl/{name}/data_test_withhead.pkl')
    except:
        print(f'Segmented data does not exist, reload: training ({train_start}~{train_end}), verify({valid_start}~{valid_end}), test({test_start}~{test_end})')
        # Reload each data set using a unified stock list
        data = get_data_my(unified_stocks, train_start, train_end, raw=True, qlib_path=QLIB_PATH, freq=freq)
        data_valid = get_data_my(unified_stocks, valid_start, valid_end, raw=True, qlib_path=QLIB_PATH, freq=freq)
        data_valid_withhead = get_data_my(unified_stocks, valid_head_start, valid_end, raw=True, qlib_path=QLIB_PATH, freq=freq)
        data_test = get_data_my(unified_stocks, test_start, test_end, raw=True, qlib_path=QLIB_PATH, freq=freq)
        data_test_withhead = get_data_my(unified_stocks, test_head_start, test_end, raw=True, qlib_path=QLIB_PATH, freq=freq)

        save_pickle(data, f'pkl/{name}/data.pkl')
        save_pickle(data_valid, f'pkl/{name}/data_valid.pkl')
        save_pickle(data_valid_withhead, f'pkl/{name}/data_valid_withhead.pkl')
        save_pickle(data_test, f'pkl/{name}/data_test.pkl')
        save_pickle(data_test_withhead, f'pkl/{name}/data_test_withhead.pkl')

    return data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, name
