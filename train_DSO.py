import torch
import sklearn
import tensorflow as tf
import numpy as np
import os,json
import pandas as pd
from dateutil.relativedelta import relativedelta

from alphagen.data.expression import *
# from alphagen_qlib.calculator import QLibStockDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year

from alphagen_generic.features import target

# ======================== DSO CONFIG ========================

DEFAULT_INSTRUMENTS = "all"
DEFAULT_FREQ = 'day'
DEFAULT_SEEDS = [0]

DEFAULT_DATA_START = "2022-01-01"
DEFAULT_DATA_END = "2024-12-31"

DEFAULT_DEVICE = 'cpu'

DEFAULT_NAME_SUFFIX = "test"

# DSO training parameters
DEFAULT_CAPACITY = 100                # AlphaPool capacity
DEFAULT_N_SAMPLES = 10000             # DSO sampling quantity
DEFAULT_BATCH_SIZE = 200              # Batch size
DEFAULT_EPSILON = 0.05                # DSO epsilon parameters
DEFAULT_ENTROPY_WEIGHT = 0.03         # Entropy weight
DEFAULT_LEARNING_RATE = 0.001         # Learning rate
DEFAULT_EARLY_STOPPING = True         # Early stop

# Log and output control
DEFAULT_DISABLE_LOGGING = False

# ============================================================

def get_data_by_time_range(instruments, data_start, data_end, target, device='cuda:0', freq: str = DEFAULT_FREQ):
    """
    Load data according to time range, refer to the implementation of get_data_by_year_with_dates
    DSO only requires one data set for calculating the reward function and supports caching mechanism
    """
    from alphagen_qlib.stock_data import StockData

    print(f"[DATA] Start loading {instruments} Data")
    print(f"[DATA] Time range: {data_start} to {data_end}")

    # Reference to the implementation of get_data_by_year_with_dates
    from gan.utils import load_pickle, save_pickle
    import os

    # Get QLIB_PATH environment variable
    qlib_path = os.environ.get('QLIB_PATH', '')
    if not qlib_path:
        raise ValueError("QLIB_PATH environment variable is not set")

    # Construct cache name
    name = f"{instruments}_dso_{str(target).replace('/','_').replace(' ','')}_{freq}_{data_start}_{data_end}"

    try:
        # Attempt to load cached data
        data = load_pickle(f'pkl/{name}/data.pkl')
        print(f"[DATA] Successfully loaded from cache: {data.data.shape}")
    except:
        print(f"[DATA] The cache does not exist, loading data from Qlib")
        # Directly use StockData to load data
        data = StockData(
            instrument=instruments,
            start_time=data_start,
            end_time=data_end,
            device=device,
            raw=True,  # Use original data
            qlib_path=qlib_path,
            freq=freq  # Explicitly specify frequency
        )

        # Save cache
        os.makedirs(f"pkl/{name}", exist_ok=True)
        save_pickle(data, f'pkl/{name}/data.pkl')
        print(f"[DATA] Data is loaded and cached successfully: {data.data.shape}")

    return data


import re
from typing import Dict

# Convert operators.funcs (GenericOperator list) to Token mapping of DSO
# Rules:
# - Ordinary one yuan/two yuan: continue to use the name and yuan number
# - Scrolling/binary scrolling: The name in operators is in the shape of ts_mean10, ts_corr20, normalizes the name to the class name ts_mean/ts_corr, and changes the element number to 2/3
#   The number of days parameter is provided by the constant Token (we add the 10/20/30/40/50 constant below)
dso_funcs: Dict[str, Token] = {}
rolling_name_pattern = re.compile(r"^(ts_\w+?)(\d+)$")

for op in generic_funcs:
    name = op.name
    arity = op.arity
    m = rolling_name_pattern.match(name)
    if m:
        base = m.group(1)
        # Determine whether it is binary scrolling: judge by the base name in operators
        if base in ("ts_cov", "ts_corr"):
            arity = 3
        else:
            arity = 2
        name = base

    # Avoid repeated coverage (multiple days mapped to the same base name)
    if name not in dso_funcs:
        dso_funcs[name] = Token(name=name, arity=arity, complexity=1, function=None)

funcs = dso_funcs.copy()
for i, feature in enumerate(['open', 'close', 'high', 'low', 'volume', 'vwap']):
    funcs[f'x {i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 20., 30., 40., 50.]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)

def main(
        # Data configuration
        instruments: str = DEFAULT_INSTRUMENTS,
        data_start: str = DEFAULT_DATA_START,
        data_end: str = DEFAULT_DATA_END,
        freq: str = DEFAULT_FREQ,

        # Training configuration
        seeds: list = DEFAULT_SEEDS,
        capacity: int = DEFAULT_CAPACITY,
        name_suffix: str = DEFAULT_NAME_SUFFIX,
        device: str = DEFAULT_DEVICE,

        # DSO training parameters
        n_samples: int = DEFAULT_N_SAMPLES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epsilon: float = DEFAULT_EPSILON,
        entropy_weight: float = DEFAULT_ENTROPY_WEIGHT,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        early_stopping: bool = DEFAULT_EARLY_STOPPING,

        # Log configuration
        disable_logging: bool = DEFAULT_DISABLE_LOGGING,
    ):
    import os
    # Do not use CUDA visibility control, only run as device
    if isinstance(seeds,str):
        seeds = eval(seeds)
    print("\n====== DSO Configuration Overview =====")
    print(f"Data set: {instruments} | Frequency: {freq} | Qlib path: {os.environ.get('QLIB_PATH', 'Not set')}")
    print(f"Data time: {data_start} to {data_end}")
    print(f"Random seed: {seeds} | Capacity: {capacity} | Custom suffix: {name_suffix}")
    print(f"DSO parameters: n_samples={n_samples} | batch_size={batch_size} | epsilon={epsilon}")
    print(f"Optimization parameters: entropy_weight={entropy_weight} | lr={learning_rate} | early_stopping={early_stopping}")
    print(f"Device: {device}")
    print("========================\n")

    # Check the QLIB_PATH environment variable
    if 'QLIB_PATH' not in os.environ:
        print("[WARNING]Warning: QLIB_PATH environment variable is not set!")

    for seed in seeds:
        tf.random.set_seed(seed)
        reseed_everything(seed)
        data = get_data_by_time_range(
            instruments=instruments,
            data_start=data_start,
            data_end=data_end,
            target=target,
            device=device,
            freq=freq
        )

        cache = {}
        # Consistent with the overall situation, construct torch equipment
        device = torch.device(device)

        X = np.array([['open_', 'close', 'high', 'low', 'volume', 'vwap']])
        y = np.array([[1]])
        functions.function_map = funcs

        pool = AlphaPool(capacity=capacity,
                        stock_data=data,
                        target=target,
                        ic_lower_bound=None)
        # Create save path: dso_{instruments}_{startYear}-{endYear}_{seed}[_suffix]
        year_start = str(data_start)[:4]
        year_end = str(data_end)[:4]
        dir_name = f"dso_{instruments}_{year_start}-{year_end}_{seed}"
        if name_suffix:
            dir_name = f"{dir_name}_{name_suffix}"
        save_path = f'out_dso/{dir_name}'
        os.makedirs(save_path, exist_ok=True)
        print(f"[SAVE] The results will be saved to: {save_path}")

        class Ev:
            def __init__(self, pool):
                self.cnt = 0
                self.pool = pool
                self.results = {}

            def alpha_ev_fn(self, key):
                expr = eval(key)
                try:
                    ret = self.pool.try_new_expr(expr)
                except OutOfDataRangeError:
                    ret = 0.0
                self.cnt += 1
                if self.cnt % 100 == 0:
                    # Evaluate the performance of the current factor pool on the same data set (used to monitor training progress)
                    current_ic = pool.test_ensemble(data, target)[0]
                    self.results[self.cnt] = current_ic
                    print(f"[INFO] No.{self.cnt} times evaluation, current IC: {current_ic:.4f}")
                return ret

        ev = Ev(pool)


        config = dict(
            task=dict(
                task_type='regression',
                function_set=list(funcs.keys()),
                metric='alphagen',
                metric_params=[lambda key: ev.alpha_ev_fn(key)],
            ),
            training={
                'n_samples': int(n_samples),
                'batch_size': int(batch_size),
                'epsilon': float(epsilon),
                'early_stopping': bool(early_stopping),
                'verbose': True,
            },
            policy={
                'policy_type': 'rnn',
                'max_length': 20,
                'cell': 'lstm',
                'num_layers': 1,
                'num_units': 32,
                'initializer': 'zeros'
            },
            policy_optimizer={
                'learning_rate': float(learning_rate),
                'entropy_weight': float(entropy_weight)
            },
            prior={'length': {'min_': 2, 'max_': 20, 'on': True}},
            experiment={'seed': seed},
            logging={'save_summary': False}
        )

        # Allow all file writing to be turned off via switch
        if disable_logging:
            config['experiment'].update({'disable_logging': True, 'logdir': None})

        # Create the model: Set logdir of DSO directly as the experimental directory (log subdirectory is no longer used)
        if not disable_logging:
            cfg_exp = config.setdefault('experiment', {})
            cfg_exp['logdir'] = save_path
            os.makedirs(cfg_exp['logdir'], exist_ok=True)
        model = DeepSymbolicRegressor(config=config)
        model.fit(X, y)
        # The final result is uniformly named dso.json
        with open(f'{save_path}/dso.json', 'w') as f:
            json.dump(pool.to_dict(), f)
        print(ev.results)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
