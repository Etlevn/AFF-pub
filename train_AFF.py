# Part of the of this repository refers to the following code:
# Shuo Yu et.al (2023).https://github.com/RL-MLDM/alphagen/tree/master
# Petersen et.al (2023)[https://github.com/dso-org/deep-symbolic-optimization]
# Microsoft (2024) [https://github.com/microsoft/qlib/tree/main/qlib]

import pandas as pd
from dateutil.relativedelta import relativedelta
from aff_module import aff_train_window
from alphagen.utils.random import reseed_everything

# ======================== AFF CONFIG ========================

# Device configuration
DEFAULT_CUDA = 0                   # GPU device number

# Data configuration
DEFAULT_INSTRUMENTS = "all"        # Stock pool: ”all”, ”csi500”, ”csi300”
DEFAULT_FREQ = "day"               # Data frequency: ”day”, ”1min”, ”5min”
DEFAULT_SEEDS = "[0]"              # Random seed: ”[0, 1, 2, 3, 4]”

# Sliding window configuration
DEFAULT_START_DATE = "2022-01-01"  # Sliding window start date
DEFAULT_END_DATE = "2024-12-31"    # Sliding window end date
DEFAULT_WINDOW_MONTHS = 24         # Total window months
DEFAULT_TRAIN_MONTHS = 18          # Training months
DEFAULT_VALID_MONTHS = 3           # Validation months
DEFAULT_TEST_MONTHS = 3            # Test months
DEFAULT_STEP_MONTHS = 3            # Sliding step number of months

# File name configuration
DEFAULT_NAME_SUFFIX = "cnn_cnn_2"       # Custom file name suffix

# ============================================================

def generate_windows(start_date, end_date, window_months=24, train_months=18, valid_months=3, test_months=3, step_months=3):
    windows = []
    current_start = pd.Timestamp(start_date)
    last_possible_start = pd.Timestamp(end_date) - pd.DateOffset(months=window_months-1)
    while current_start <= last_possible_start:
        # Global window coverage DEFAULT_WINDOW_MONTHS (including training + verification + testing)
        global_start = current_start
        global_end = global_start + pd.DateOffset(months=window_months-1)

        # Sub-interval: training/validation/testing
        train_start = global_start
        train_end = train_start + pd.DateOffset(months=train_months-1)
        valid_start = train_end + pd.DateOffset(months=1)
        valid_end = valid_start + pd.DateOffset(months=valid_months-1)
        test_start = valid_end + pd.DateOffset(months=1)
        test_end = test_start + pd.DateOffset(months=test_months-1)
        windows.append({
            'global_start': global_start.strftime('%Y-%m-%d'),
            'global_end': global_end.strftime('%Y-%m-%d'),
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'valid_start': valid_start.strftime('%Y-%m-%d'),
            'valid_end': valid_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        current_start += pd.DateOffset(months=step_months)
    return windows

def run_aff_sliding_window(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    window_months=DEFAULT_WINDOW_MONTHS,
    train_months=DEFAULT_TRAIN_MONTHS,
    valid_months=DEFAULT_VALID_MONTHS,
    test_months=DEFAULT_TEST_MONTHS,
    step_months=DEFAULT_STEP_MONTHS,
    seeds: str = DEFAULT_SEEDS,
    name_suffix: str = DEFAULT_NAME_SUFFIX,
    **aff_kwargs
):
    print("\n" + "="*100)
    print("="*100)
    print("[Sliding Window Master] Start sliding window experiment")
    print("="*100)
    print("="*100)

    # Print configuration information
    print(f"[Configuration information] Sliding window: {start_date} to {end_date}")
    print(f"[Configuration information] Window settings: {window_months} months total, {train_months} months training, {valid_months} months validation, {test_months} months testing")
    print(f"[Configuration information] Sliding step size: {step_months} months")
    print(f"[Configuration information] Random seed: {seeds}")
    print(f"[Configuration information] GPU device: {aff_kwargs.get('cuda', 0)}")
    print(f"[Configuration information] File name suffix: ’{name_suffix}'" if name_suffix else "[Configuration Information] File name suffix: None")

    # Original random seed processing logic
    if isinstance(seeds, str):
        seeds = eval(seeds)
    assert isinstance(seeds, list)
    print(f"[INFO] Random seed setting: {seeds}")
    print(f"[INFO] Training with {len(seeds)} seeds")

    import torch
    cuda_val = DEFAULT_CUDA
    if cuda_val > 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_val}')
    else:
        device = torch.device('cpu')
    print(f"[INFO] Device: {device}")
    aff_kwargs['device'] = device

    instruments = DEFAULT_INSTRUMENTS
    freq = DEFAULT_FREQ

    # Create a total training folder with a fixed prefix of aff
    start_year = start_date[:4]
    end_year = end_date[:4]

    # Create an independent training process for each seed
    for seed in seeds:
        print(f"\n" + "="*80)
        print(f"[SEED {seed} Start training")
        print("="*80)

        # Set random seed
        reseed_everything(seed)
        print(f"[INFO] Random seed has been set: {seed}")

        # Create a seed-specific save name: aff_{instruments}_{startYear}-{endYear}_{seed}[_suffix]
        base_name = f"aff_{instruments}_{start_year}-{end_year}_{seed}"
        if name_suffix:
            global_save_name = f"{base_name}_{name_suffix}"
        else:
            global_save_name = base_name
        print(f"[INFO] Seed {seed} training folder: out/{global_save_name}")

        # Generate time window
        windows = generate_windows(
            start_date=start_date,
            end_date=end_date,
            window_months=window_months,
            train_months=train_months,
            valid_months=valid_months,
            test_months=test_months,
            step_months=step_months
        )
        total_windows = len(windows)
        print(f"[INFO] Seed {seed} Will train {total_windows} time windows")

        # Pass parameters for the current seed
        current_aff_kwargs = aff_kwargs.copy()
        current_aff_kwargs['seeds'] = f'[{seed}]'  # The current seed only uses one seed

        factor_pool = None
        for idx, win in enumerate(windows):
            # Remove redundant seed information printing and only transfer seed information to aff_train_window
            factor_pool = aff_train_window(
                instruments=instruments,
                global_start=win['global_start'],
                global_end=win['global_end'],
                train_start=win['train_start'],
                train_end=win['train_end'],
                valid_start=win['valid_start'],
                valid_end=win['valid_end'],
                test_start=win['test_start'],
                test_end=win['test_end'],
                freq=freq,
                cuda=cuda_val,
                save_name=global_save_name,
                factor_pool_in=factor_pool,
                window_idx=idx,
                total_windows=total_windows,
                current_seed=seed,
                **current_aff_kwargs
            )

        print(f"\n[SEED {seed}] Training is completed, and the final factor pool is saved at: out/{global_save_name}")
        print("="*80)

    print(f"\n" + "="*100)
    print("="*100)
    print("[Sliding Window Master] All seed training completed")
    print(f"[INFO] The results are saved in the following directory:")
    for seed in seeds:
        base_dir = f"aff_{instruments}_{start_year}-{end_year}_{seed}"
        result_dir = f"out/{base_dir}{('_' + name_suffix) if name_suffix else ''}"
        print(f"  - seed {seed}: {result_dir}")
    print("="*100)
    print("="*100)

if __name__ == '__main__':
    import fire
    fire.Fire(run_aff_sliding_window)
