# aff_module.py
"""
Reusable Alpha Factor Factory training for sliding windows
aff_train_window collects expressions, evaluates factors, trains networks, and
rebalances the factor pool for a caller-supplied data window.
"""
import torch
import os
import warnings
import time
import types
import pandas as pd
import qlib
from qlib.config import REG_CN
import alphagen.config as config
from gan.dataset import Collector
from gan.network.masker import NetM
from gan.network.predictor import train_regression_model_with_weight, adaptive_weighted_mse_loss, adaptive_weighted_mse_loss_debug, weighted_mse_loss
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_ret,batch_pearsonr
import numpy as np
from alphagen.utils.random import reseed_everything
from gan.utils import filter_valid_blds,save_blds
from gan.network.generater import train_network_generator
import gc
from gan.utils.data import get_data_by_year, get_data_by_year_with_dates
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from gan.utils.builder import filter_valid_blds
from backtest.factor_data import FactorData
import importlib

# ======================== AFF CONFIG ========================

# Training configuration
DEFAULT_ROUND_NUM = 20  # Training rounds
DEFAULT_ZOO_SIZE = 100000  # Factor pool size

# Threshold configuration
DEFAULT_CORR_THRESH = 0.85  # Correlation threshold
DEFAULT_IC_THRESH = 0.03    # IC threshold
DEFAULT_ICIR_THRESH = 0.065 # ICIR threshold
DEFAULT_RIC_THRESH = 0.02   # RankIC threshold
DEFAULT_RICIR_THRESH = 0.05 # RankICIR threshold

# Function switch configuration
DEFAULT_PLOT_LOSS = True                  # Whether to draw epoch-loss diagram
DEFAULT_USE_EXCESS_IC = True              # Whether to use excess IC as the prediction target
DEFAULT_USE_ADVANCED_OPERATORS = True     # Whether to use advanced operator library
DEFAULT_USE_PRECISE_DATES = True          # Whether to use precise date data loading function
DEFAULT_USE_Y_PREPROCESSING = True        # Whether to enable y preprocessing
DEFAULT_ENABLE_RANKIC = False             # Whether to enable RankIC calculation
DEFAULT_CLEAN_TARGET_NAN = False          # Whether to enable target variable NaN cleanup
DEFAULT_USE_ADAPTIVE_LOSS = False         # Whether to use adaptive loss function
DEFAULT_FILTER_ZERO_TARGETS = False       # Whether to filter zero value target data
DEFAULT_USE_ADAPTIVE_COLLECTION = False   # Whether to use round-by-round expression collection strategy
DEFAULT_ONLY_CORR_REBALANCE = False       # Whether to perform factor pool rebalancing only by correlation
DEFAULT_ENABLE_MASK = True                # Whether to enable effective data masking
DEFAULT_SAVE_MASK = False                 # Whether to save the effective data mask of each window
DEFAULT_SAVE_REBALANCE_CORR = True        # Whether to save the factor pool correlation coefficient matrix after rebalancing
DEFAULT_SAVE_CORR_MEAN_TXT = False        # Whether to save the statistical text of correlation after rebalancing

# Operator library tag
DEFAULT_OPERATORS_TAGS = "std,cstm,qlib,talib"       # "std, cstm, qlib, talib"

# Model configuration
DEFAULT_MODEL_CONFIG = "cnn_cnn"       # "cnn_cnn", "dcgan_netp", "lstm_netp"

# ============================================================

def numpy2onehot(integer_matrix,max_num_categories=None,min_num_categories=None):
    if max_num_categories is None:
        max_num_categories = np.max(integer_matrix) + 1
    if min_num_categories is None:
        min_num_categories = np.min(integer_matrix)

    # Ensure that the value in integer_matrix does not exceed max_num_categories
    if max_num_categories is not None:
        integer_matrix = np.clip(integer_matrix, min_num_categories, max_num_categories - 1)

    integer_matrix = integer_matrix - min_num_categories
    num_categories = max_num_categories - min_num_categories
    return np.eye(num_categories)[integer_matrix]

def blds_list_to_tensor(blds_list,weights_list:List[int]):
    # Dynamically obtain SIZE_ACTION consistent with NetP/NetG
    SIZE_ACTION = importlib.import_module('alphagen.rl.env.wrapper').SIZE_ACTION
    assert len(blds_list) == len(weights_list)
    x_numpy_list = []
    y_numpy_list = []
    weights_numpy_list = []
    for blds,weight_int in zip(blds_list,weights_list):
        x_numpy = numpy2onehot(np.array(blds.builders_tokens),SIZE_ACTION,0).astype('float32')
        y_numpy = np.array(blds.scores).astype('float32')[:,None]
        weights_numpy = np.ones(x_numpy.shape[0]).astype('float32')[:,None] * weight_int
        x_numpy_list.append(x_numpy)
        y_numpy_list.append(y_numpy)
        weights_numpy_list.append(weights_numpy)
    x_numpy = np.concatenate(x_numpy_list,axis=0)
    y_numpy = np.concatenate(y_numpy_list,axis=0)
    weights_numpy = np.concatenate(weights_numpy_list,axis=0)
    x = torch.from_numpy(x_numpy)
    y = torch.from_numpy(y_numpy)
    weights = torch.from_numpy(weights_numpy)
    return x,y,weights

def train_net_p_with_weight(cfg, net, x, y, weights, device, lr=0.001, use_adaptive_loss=True, filter_zero_targets=True):

    # Filter zero value data (optional)
    if filter_zero_targets:
        # Use threshold to determine zero value instead of strict y == 0
        threshold = 1e-6  # Very small threshold
        non_zero_mask = (torch.abs(y) > threshold).flatten()
        zero_count = (torch.abs(y) <= threshold).sum().item()
        total_count = y.size(0)
        non_zero_count = non_zero_mask.sum().item()

        print(f"[INFO] Zero value proportion: {zero_count}/{total_count}={zero_count/total_count:.1%}")
        print(f"[INFO] Zero value judgment threshold: {threshold}")

        if non_zero_count > 100:  # Ensure there is enough non-zero data for training
            print(f"[INFO] Filter zero value data and retain {non_zero_count} valid samples")
            x = x[non_zero_mask]
            y = y[non_zero_mask]
            weights = weights[non_zero_mask]
        else:
            print(f"[WARNING] Too few non-zero data ({non_zero_count}), retain all data")
    else:
        print("[INFO] Zero value data filtering disabled")

    x_train, x_valid, y_train, y_valid,weights_train,weights_valid = train_test_split(x, y,weights, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(x_train, y_train,weights_train), batch_size=cfg.batch_size_p, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid,weights_valid), batch_size=cfg.batch_size_p, shuffle=False)

    # Select loss function according to parameters
    if use_adaptive_loss:
        loss_fn = adaptive_weighted_mse_loss  # Adaptive normalized loss function
        print("[INFO] Use adaptive loss function (adaptive_weighted_mse_loss)")
    else:
        loss_fn = weighted_mse_loss  # Original loss function
        print("[INFO] Use the original loss function (weighted_mse_loss)")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Check whether loss plot drawing is enabled
    plot_loss = getattr(cfg, 'plot_loss', False)  # Default is False

    train_regression_model_with_weight(train_loader, valid_loader, net, loss_fn, optimizer, device=device, num_epochs=cfg.num_epochs_p, use_tensorboard=False, tensorboard_path='logs', early_stopping_patience=cfg.es_p, plot_loss=plot_loss)

def safe_batch_spearmanr(x, y, chunk_size=100):
    n_days = x.shape[0]
    spearmanr_list = []
    for i in range(0, n_days, chunk_size):
        end_idx = min(i + chunk_size, n_days)
        x_chunk = x[i:end_idx]
        y_chunk = y[i:end_idx]
        chunk_results = []
        for day_idx in range(len(x_chunk)):
            x_day = x_chunk[day_idx]
            y_day = y_chunk[day_idx]
            valid_mask = torch.isfinite(x_day) & torch.isfinite(y_day)
            if valid_mask.sum() < 3:
                chunk_results.append(torch.tensor(0.0))
                continue
            x_valid = x_day[valid_mask]
            y_valid = y_day[valid_mask]
            try:
                x_ranked = torch.argsort(torch.argsort(x_valid)).float()
                y_ranked = torch.argsort(torch.argsort(y_valid)).float()
                x_centered = x_ranked - x_ranked.mean()
                y_centered = y_ranked - y_ranked.mean()
                numerator = (x_centered * y_centered).sum()
                denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
                if denominator == 0:
                    corr = torch.tensor(0.0)
                else:
                    corr = numerator / denominator
                chunk_results.append(corr)
            except:
                chunk_results.append(torch.tensor(0.0))
        spearmanr_list.extend(chunk_results)
    return torch.stack(spearmanr_list)

def get_metric(zoo_blds,device,corr_thresh=0.5,metric_target='ic',enable_rankic=False,use_excess_ic=False,benchmark_returns=None):
    n_blds = len(zoo_blds)
    if n_blds >0:
        n_days = len(zoo_blds.ret_list[0])
        existed = zoo_blds.ret_list
        existed = np.vstack(existed)
        existed = torch.from_numpy(existed).to(device)
        assert existed.shape == (n_blds,n_days)
    else:
        existed = None
        n_days = None
    def get_score(fct,tgt):
        metric_target = 'ic'
        ret = batch_ret(fct,tgt)
        ic = batch_pearsonr(fct,tgt)
        if enable_rankic:
            ric = safe_batch_spearmanr(fct, tgt, chunk_size=50)
            ric_mean = ric.mean().abs().item()
            ricir = (ric_mean/ric.std()).item()
        else:
            ric_mean = 0.0
            ricir = 0.0
        ic_mean = ic.mean().abs().item()
        icir = (ic_mean/ic.std()).item()
        ret_mean = ret.mean().abs().item()
        ret_ir = (ret_mean/ret.std()).item()
        sharpe = ((ret_mean- 0.03/252)/ret.std() * np.sqrt(252)).item()

        # Calculate excess IC related indicators (if enabled)
        if use_excess_ic:
            excess_ic, benchmark_ic, individual_ic = calculate_excess_ic(fct, tgt, benchmark_returns)
        else:
            excess_ic = 0.0
            benchmark_ic = 0.0
            individual_ic = ic_mean

        def invalid_to_zero(x):
            if not np.isfinite(x):
                return 0.
            else:
                return max(x,0.)

        multi_score = {'ic':ic_mean,'icir':icir,'ret':ret_mean,'sharpe':sharpe,'retir':ret_ir}
        if enable_rankic:
            multi_score.update({'ric':ric_mean,'ricir':ricir})
        if use_excess_ic:
            multi_score.update({'excess_ic':excess_ic,'benchmark_ic':benchmark_ic,'individual_ic':individual_ic})
        multi_score = {k:invalid_to_zero(v) for k,v in multi_score.items()}
        score = multi_score[metric_target]
        if torch.isfinite(fct[0]).sum()/torch.isfinite(tgt[0]).sum() <0.8:
            score = 0.
        elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) <0.01:
            score = 0.
        if n_blds > 0 and score > 0. and existed is not None:
            assert len(ret.shape) == 1 , f"{ret.shape},{n_days}"
            assert len(ret) == n_days , f"{ret.shape},{n_days}"
            all_matrix = torch.concatenate([existed,ret[None]],dim=0)
            assert all_matrix.shape == (n_blds+1,n_days) , f"{all_matrix.shape}"
            corr_score = torch.corrcoef(all_matrix)[-1,:-1].abs().max().item()
            if corr_score > corr_thresh:
                score = 0.
        return {'score':score,'ret':ret.detach().cpu().numpy(),'multi_score':multi_score}
    return get_score

def calculate_benchmark_returns(data):
    """
    Calculate benchmark return - using the market average return of the current window
    Refer to the benchmark income calculation method in factor_analysis_functions.py

    Args:
        data: Stock data object (data of the current training window)

    Returns:
        benchmark_returns: Benchmark rate of return (time x stock)
    """
    # Obtain data dimensions
    n_days, n_features, n_stocks = data.data.shape

    # Get closing price data (FeatureType.CLOSE = 1)
    close_prices = data.data[:, 1, :]  # (time x stock)

    # Calculate the rate of return: (today’s closing price - yesterday’s closing price) / yesterday’s closing price
    # Note: There is no yesterday’s data on the first day, so the rate of return starts from the next day
    if n_days < 2:
        # If the data is less than 2 days, return zero rate of return
        benchmark_returns = torch.zeros_like(close_prices)
    else:
        # Calculate the return rate of all stocks
        returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]  # (Time-1 x Stock)

        # Calculate the market average return (average return of all stocks, ignore NaN)
        market_avg_returns = torch.nanmean(returns, dim=1, keepdim=True)  # (Time-1 x 1)

        # Expand to all stocks
        market_avg_returns = market_avg_returns.expand(-1, n_stocks)  # (Time-1 x Stock)

        # Add zero yield for the first day
        first_day_returns = torch.zeros(1, n_stocks, device=close_prices.device)
        benchmark_returns = torch.cat([first_day_returns, market_avg_returns], dim=0)  # (time x stock)

    print(f"[INFO] Calculate the current window benchmark return (market average return)")
    print(f"[INFO] Data range: {n_days} day x {n_stocks} only stocks")
    print(f"[INFO] Baseline income shape: {benchmark_returns.shape}")
    print(f"[INFO] Benchmark income statistics: mean ={benchmark_returns.mean().item():.6f}, standard deviation ={benchmark_returns.std().item():.6f}")

    return benchmark_returns

def calculate_excess_ic(fct, tgt, benchmark_returns=None):
    """
    Calculate excess IC - the predictive ability of factors on excess returns
    Data calculation based on current training window

    Args:
        fct: factor value (time x stock) - current window
        tgt: Target return (time x stock) - current window
        benchmark_returns: Baseline return (time x stock) - current window, if it is None, use the original return

    Returns:
        excess_ic: Excess IC value
        benchmark_ic: Baseline IC value
        individual_ic: Original IC value
    """
    # Calculate original IC
    ic = batch_pearsonr(fct, tgt)
    individual_ic = ic.mean().item()

    if benchmark_returns is None:
        # If no baseline income is provided, use the original income
        excess_ic = individual_ic
        benchmark_ic = 0.0
    else:
        # Ensure that the benchmark income and target income dimensions match
        tgt_shape = tgt.shape
        benchmark_shape = benchmark_returns.shape

        if benchmark_shape[0] != tgt_shape[0]:
            # If the time dimension does not match, alignment is required
            # Assume that the target income is a subset of the benchmark income, and take the corresponding time range
            time_offset = benchmark_shape[0] - tgt_shape[0]
            if time_offset > 0:
                # Take the back part of the benchmark income and align it with the target income
                benchmark_returns_aligned = benchmark_returns[time_offset:, :]
            else:
                # If the benchmark profit time is shorter, use it directly
                benchmark_returns_aligned = benchmark_returns
        else:
            benchmark_returns_aligned = benchmark_returns

        # Calculate excess returns
        excess_returns = tgt - benchmark_returns_aligned

        # Calculate excess IC
        excess_ic_tensor = batch_pearsonr(fct, excess_returns)
        excess_ic = excess_ic_tensor.mean().item()

        # Calculate benchmark IC (correlation between factors and benchmark returns)
        benchmark_ic_tensor = batch_pearsonr(fct, benchmark_returns_aligned)
        benchmark_ic = benchmark_ic_tensor.mean().item()

    return excess_ic, benchmark_ic, individual_ic

def get_training_metric(device='cpu', corr_thresh=0.5, use_excess_ic=True, benchmark_returns=None):
    """
    Obtain training index function
    Calculating indicators based on the data of the current training window

    Args:
        device: Compute device
        corr_thresh: correlation threshold
        use_excess_ic: Whether to use excess IC as the prediction target
        benchmark_returns: Benchmark return (time x stock) - current window
    """
    def get_score(fct, tgt):
        # Calculate basic indicators
        ret = batch_ret(fct, tgt)
        ic = batch_pearsonr(fct, tgt)

        if use_excess_ic:
            # Use excess IC as prediction target
            excess_ic, benchmark_ic, individual_ic = calculate_excess_ic(fct, tgt, benchmark_returns)

            # Other indicators remain unchanged
            icir = (ic.mean().abs()/ic.std()).item()
            ret_mean = ret.mean().abs().item()
            ret_ir = (ret_mean/ret.std()).item()
            sharpe = ((ret_mean- 0.03/252)/ret.std() * np.sqrt(252)).item()

            def invalid_to_zero(x):
                if not np.isfinite(x):
                    return 0.
                else:
                    return x  # Reserve negative values, not forced to 0

            multi_score = {
                'excess_ic': excess_ic,  # Excess IC
                'benchmark_ic': benchmark_ic,  # Benchmark IC
                'individual_ic': individual_ic,  # Original IC
                'ic': ic.mean().abs().item(),  # Keep the original IC absolute value
                'icir': icir,
                'ret': ret_mean,
                'sharpe': sharpe,
                'retir': ret_ir
            }
            multi_score = {k: invalid_to_zero(v) for k,v in multi_score.items()}
            # Consistent with the original version: absolute values are used for scoring
            score = abs(multi_score['excess_ic'])  # Use excess IC intensity as prediction target

        else:
            # Original implementation: use the absolute value of IC
            metric_target = 'ic'
            ic_mean = ic.mean().abs().item()
            icir = (ic_mean/ic.std()).item()
            ret_mean = ret.mean().abs().item()
            ret_ir = (ret_mean/ret.std()).item()
            sharpe = ((ret_mean- 0.03/252)/ret.std() * np.sqrt(252)).item()

            def invalid_to_zero(x):
                if not np.isfinite(x):
                    return 0.
                else:
                    return max(x,0.)

            multi_score = {'ic':ic_mean,'icir':icir,'ret':ret_mean,'sharpe':sharpe,'retir':ret_ir}
            multi_score = {k:invalid_to_zero(v) for k,v in multi_score.items()}
            score = multi_score[metric_target]

        # Validity checks remain unchanged
        if torch.isfinite(fct[0]).sum()/torch.isfinite(tgt[0]).sum() < 0.8:
            score = 0.
        elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) < 0.01:
            score = 0.

        return {
            'score': score,
            'ret': ret.detach().cpu().numpy(),
            'multi_score': multi_score
        }
    return get_score

def pre_process_y(y):
    min_y = 0
    max_y = y.flatten().max()
    y = (y - min_y) / (max_y - min_y) * 100
    return y

def clean_target_nan_samples(data, target, verbose=True):
    """
    Strategy B: Completely remove samples whose target variable is NaN
    By reconstructing the data tensor, only the (time, stock) pairs that are valid for the target variable are retained.

    Args:
        data: Stock data object
        target: target variable expression
        verbose: Whether to display detailed output

    Returns:
        tuple: (reconstructed data object, statistical information dictionary)
    """
    if not verbose:
        return data, {"status": "skipped", "message": "NaN cleanup function disabled"}

    print("\n" + "="*50)
    print("Target variable NaN sample cleaning:")
    print("="*50)

    try:
        if not DEFAULT_ENABLE_MASK:
            print("[INFO] The global mask switch is turned off, skipping mask generation and application")
            raise RuntimeError("MASK_DISABLED")
        # Re-evaluate the target variable to obtain the latest status
        target_sample = target.evaluate(data)

        # Calculate NaN mask (time x stock)
        target_nan_mask = torch.isnan(target_sample)
        nan_count_before = target_nan_mask.sum().item()
        total_samples = target_sample.numel()

        print(f"[INFO] NaN sample before cleaning: {nan_count_before:,}/{total_samples:,} ({nan_count_before/total_samples:.2%})")

        if nan_count_before > 0:
            # Obtain original data dimensions
            original_data_shape = data.data.shape  # (time x feature x stock)
            n_days, n_features, n_stocks = original_data_shape
            target_days, target_stocks = target_sample.shape

            # Calculate time offset
            time_offset = n_days - target_days

            # Find valid (time, stock) pairs
            valid_mask = ~target_nan_mask  # (time x stock)

            # Create a copy of the data and delete the sample whose target variable is NaN
            import copy
            cleaned_data = copy.deepcopy(data)

            removed_count = 0
            for target_day_idx in range(target_days):
                data_day_idx = target_day_idx + time_offset
                if 0 <= data_day_idx < n_days:
                    for stock_idx in range(min(target_stocks, n_stocks)):
                        if target_nan_mask[target_day_idx, stock_idx]:
                            cleaned_data.data[data_day_idx, :, stock_idx] = float('nan')
                            removed_count += 1

            print(f"[INFO] Deleted {removed_count:,} target variable NaN samples")
            data = cleaned_data

        else:
            print(f"[INFO] The target variable has no NaN value and does not need to be cleaned.")

        # Verification after cleaning
        target_sample_final = target.evaluate(data)
        final_valid_count = torch.isfinite(target_sample_final).sum().item()
        final_total = target_sample_final.numel()
        final_nan_count = final_total - final_valid_count

        print(f"[INFO] Efficiency after cleaning: {final_valid_count:,}/{final_total:,} ({final_valid_count/final_total:.2%})")

        return data, {
            "total_samples": final_total,
            "valid_samples": final_valid_count,
            "nan_samples": final_nan_count,
            "valid_rate": final_valid_count/final_total
        }

    except Exception as e:
        print(f"[ERROR] Target variable NaN cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return data, {"status": "error", "message": str(e)}

def aff_train_window(
        instruments: str = None,
        freq: str = None,
        cuda: int = None,
        global_start: str = None,
        global_end: str = None,
        train_start: str = None,
        train_end: str = None,
        valid_start: str = None,
        valid_end: str = None,
        test_start: str = None,
        test_end: str = None,
        round_num: int = DEFAULT_ROUND_NUM,
        seeds: str = None,
        corr_thresh: float = DEFAULT_CORR_THRESH,
        ic_thresh: float = DEFAULT_IC_THRESH,
        icir_thresh: float = DEFAULT_ICIR_THRESH,
        ric_thresh: float = DEFAULT_RIC_THRESH,
        ricir_thresh: float = DEFAULT_RICIR_THRESH,
        enable_rankic: bool = DEFAULT_ENABLE_RANKIC,
        save_name: str = None,
        zoo_size: int = DEFAULT_ZOO_SIZE,
        factor_pool_in = None,
        window_idx: int = None,
        total_windows: int = None,
        clean_target_nan: bool = DEFAULT_CLEAN_TARGET_NAN,  # New parameter: whether to enable target variable NaN cleanup
        current_seed: int = None,  # New parameter: current seed information
        plot_loss: bool = DEFAULT_PLOT_LOSS,  # New parameter: whether to draw epoch-loss diagram
        use_excess_ic: bool = DEFAULT_USE_EXCESS_IC,  # New parameter: whether to use excess IC as the prediction target
        benchmark_returns: torch.Tensor = None,  # New parameter: Benchmark return (time x stock)
        use_adaptive_loss: bool = DEFAULT_USE_ADAPTIVE_LOSS,  # New parameter: whether to use adaptive loss function
        filter_zero_targets: bool = DEFAULT_FILTER_ZERO_TARGETS,  # New parameter: whether to filter zero value target data
        use_y_preprocessing: bool = DEFAULT_USE_Y_PREPROCESSING,  # New parameter: whether to enable y preprocessing
        use_adaptive_collection: bool = DEFAULT_USE_ADAPTIVE_COLLECTION,  # New parameter: whether to use round-by-round expression collection strategy
        use_precise_dates: bool = DEFAULT_USE_PRECISE_DATES,  # New parameter: whether to use precise date data loading function
        model_config: str = DEFAULT_MODEL_CONFIG,  # New parameters: model configuration selection
        only_corr_rebalance: bool = DEFAULT_ONLY_CORR_REBALANCE,  # New parameter: Relevance rebalancing only
        operators_tags: str = None,  # Combined by tags, such as ”std, cstm, qlib”
        **kwargs
):
    # Configuration operator library (versioned)
    from alphagen import config as _ag_config
    tags_to_use = operators_tags or DEFAULT_OPERATORS_TAGS
    _ag_config.initialize_operators_by_tags(tags_to_use)
    from alphagen.rl.env.wrapper import SIZE_ACTION
    print(f"[INFO] Initialization operator library label: {tags_to_use} | Number of operators: {len(_ag_config.OPERATORS)}|Action space size: {SIZE_ACTION}")

    # Check save_name parameters
    if save_name is None:
        raise ValueError("The save_name parameter cannot be None and must be specified by the caller")
    print(f"[INFO] Save name: {save_name}")

    # Compatibility switch status display
    print(f"[INFO] Compatibility switch setting:")
    print(f"  - Excess ic training: {'enable' if use_excess_ic else 'Disable'}")
    print(f"  - Accurate date data loading: {'enable' if use_precise_dates else 'Disable'}")
    print(f"  - y preprocessing: {'enable' if use_y_preprocessing else 'Disable'}")
    print(f"  - Adaptive loss function: {'enable' if use_adaptive_loss else 'Disable'}")
    print(f"  - Zero value target filtering: {'enable' if filter_zero_targets else 'Disable'}")
    print(f"  - Collection of expressions in rounds: {'enable' if use_adaptive_collection else 'Disable'}")
    print(f"  - Relevance rebalancing only: {'enable' if only_corr_rebalance else 'Disable'}")
    print(f"  - rankic calculation: {'enable' if enable_rankic else 'Disable'}")

    # Sliding window unique print (retained only once)
    if window_idx is not None and total_windows is not None:
        print("\n" + "="*100)
        print("="*100)
        print(f"[WINDOW] No.{window_idx+1}/{total_windows} windows start training")
        print(f"During training: {train_start} ~ {train_end}")
        print(f"Validation period: {valid_start} ~ {valid_end}")
        print(f"During the test: {test_start} ~ {test_end}")
        print("="*100)
        print("="*100)
        print()
    if isinstance(seeds, str):
        seeds = eval(seeds)
    assert isinstance(seeds, list)
    device = torch.device("cpu")
    device_str = str(device)
    # Data loading
    import qlib
    from qlib.config import REG_CN
    QLIB_PATH = os.environ.get("QLIB_PATH")
    if QLIB_PATH is None:
        raise ValueError("Please set the environment variable QLIB_PATH to point to the Qlib data directory")
    qlib.init(provider_uri=QLIB_PATH, region=REG_CN)
    # Load global window data (24 months, including training/validation/testing) to data
    # While retaining the original training/validation/test loading logic: the original training set is renamed to data_train
    if use_precise_dates:
        print(f"[INFO] Load function using exact date data (Global Window)")
        if global_start is None or global_end is None:
            # Fallback to infer global scope with train_start/train_end
            _gs = train_start
            _ge = test_end if test_end is not None else train_end
        else:
            _gs, _ge = global_start, global_end
        # Load global 24 monthly data as data
        global_returned = get_data_by_year_with_dates(
            train_start=_gs,
            train_end=_ge,
            valid_start=_gs,
            valid_end=_ge,
            test_start=_gs,
            test_end=_ge,
            instruments=instruments,
            target=target,
            freq=freq
        )
        data_all, data, _, _, _, _, _ = global_returned

        # Load three original segments separately (only for retention, not used in training)
        split_returned = get_data_by_year_with_dates(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            instruments=instruments,
            target=target,
            freq=freq
        )
    else:
        print(f"[INFO] Use year data loading function")
        # Compatible with the original year interface, global_start/global_end degenerates into year boundaries
        if global_start is None or global_end is None:
            train_start_year = int(train_start[:4])
            test_year = int((test_end or train_end)[:4])
            gs_year, ge_year = train_start_year, test_year
        else:
            gs_year, ge_year = int(global_start[:4]), int(global_end[:4])

        global_returned = get_data_by_year(
            train_start=gs_year,
            train_end=ge_year,
            valid_year=ge_year,
            test_year=ge_year,
            instruments=instruments,
            target=target,
            freq=freq
        )
        data_all, data, _, _, _, _, _ = global_returned

        # Original segmentation
        train_start_year = int(train_start[:4])
        train_end_year = int(train_end[:4])
        valid_year = int(valid_start[:4])
        test_year = int(test_start[:4])
        split_returned = get_data_by_year(
            train_start=train_start_year,
            train_end=train_end_year,
            valid_year=valid_year,
            test_year=test_year,
            instruments=instruments,
            target=target,
            freq=freq
        )

    # Unpack the original segmentation and rename the training set
    _, data_train, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = split_returned
    print(f"=== [INFO] Data loading parameters: global_start={global_start or _gs}, global_end={global_end or _ge}, train_start={train_start}, train_end={train_end}, valid_start={valid_start}, test_start={test_start}, freq={freq} ===")
    print(f"[INFO] Global data shape: {data.data.shape} (number of days × number of features × number of stocks)")
    print(f"[INFO] data_train shape: {data_train.data.shape} (number of days × number of features × number of stocks)")

    # === Generate a valid data mask with the same logic for backtesting and save it ===
    if DEFAULT_ENABLE_MASK:
        try:
            # Unify global start and end date strings
            gs_str = (global_start or _gs)
            ge_str = (global_end or _ge)
            print(f"[INFO] Construct the global mask interval: {gs_str} ~ {ge_str}")
            # Construct tradable days and indexes directly from the loaded global data without reading OHLCV again
            dates_idx = pd.to_datetime(pd.Index(data._dates))
            stocks_idx = pd.Index(list(map(str, data._stock_ids)))
            arr = data.data.detach().cpu().numpy()  # (days, features, stocks)

            # Obtain the index through the feature name to avoid hard-coded serial numbers
            from alphagen_qlib.stock_data import FeatureType
            features = data._features if hasattr(data, '_features') else list(FeatureType)
            feature_to_idx = {f: i for i, f in enumerate(features)}

            try:
                open_idx = feature_to_idx[FeatureType.OPEN]
                close_idx = feature_to_idx[FeatureType.CLOSE]
                volume_idx = feature_to_idx[FeatureType.VOLUME]
            except KeyError as e:
                print(f"[WARNING] Failed to obtain feature index {e}, fall back to the default sequence number")
                open_idx, close_idx, volume_idx = 0, 1, 4

            open_df = pd.DataFrame(arr[:, open_idx, :], index=dates_idx, columns=stocks_idx)
            close_df = pd.DataFrame(arr[:, close_idx, :], index=dates_idx, columns=stocks_idx)
            volume_df = pd.DataFrame(arr[:, volume_idx, :], index=dates_idx, columns=stocks_idx)
            data_tradable = ((~open_df.isna()) & (~close_df.isna()) & (volume_df > 0)).astype('int8')

            # Optimization: Read instruments/all.txt directly at the beginning of the window to avoid repeated reading inside the function
            all_txt_path = os.path.join(QLIB_PATH, 'instruments', 'all.txt')
            mask_df = None
            try:
                inst_df = pd.read_csv(
                    all_txt_path,
                    sep='\t',
                    header=None,
                    names=['instrument', 'start_date', 'end_date'],
                    dtype={'instrument': str, 'start_date': str, 'end_date': str}
                )
                inst_df['start_date'] = pd.to_datetime(inst_df['start_date'], format='%Y-%m-%d', errors='coerce')
                inst_df['end_date'] = pd.to_datetime(inst_df['end_date'], format='%Y-%m-%d', errors='coerce')
                dates = dates_idx
                stocks = stocks_idx
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
                mask_df = (data_listed & data_tradable).astype('int8')
            except Exception as _:
                # Fallback to built-in function
                factor_data_for_mask = FactorData(qlib_path=QLIB_PATH, load_ohlcv=True)
                factor_data_for_mask.load_all_data(
                    start_time=gs_str,
                    end_time=ge_str,
                    instruments=list(map(str, data._stock_ids)),
                    include_filters=False,
                    load_ohlcv=True,
                )
                mask_df = factor_data_for_mask.load_valid_mask(start_time=gs_str, end_time=ge_str, instruments=list(map(str, data._stock_ids)))
            if mask_df is not None:
                # Aligned with global data for training (date, stock)
                mask_df = mask_df.reindex(index=dates_idx, columns=stocks_idx)
                mask_df = mask_df.fillna(0).astype('int8')

                # === Apply mask to global data tensor (core optimization) ===
                print(f"[INFO] Apply mask to global data tensor...")
                # Broadcast mask_df (dates x stocks, 0/1) to (dates x features x stocks)
                mask_np = mask_df.values.astype(np.float32)  # (T, N)
                mask_t = torch.from_numpy(mask_np).to(device=data.data.device)  # (T, N)
                mask_3d = mask_t.unsqueeze(1).expand(-1, data.data.shape[1], -1)  # (T, F, N)
                nan_tensor = torch.tensor(float('nan'), device=data.data.device)

                # Set the invalid position to NaN
                data.data = torch.where(mask_3d > 0.5, data.data, nan_tensor)

                # Statistical application effect
                total_elements = data.data.numel()
                nan_count = torch.isnan(data.data).sum().item()
                valid_ratio = (total_elements - nan_count) / total_elements
                print(f"[INFO] Mask application completed: total elements ={total_elements:,}, valid element ={(total_elements - nan_count):,}, efficient ={valid_ratio:.4%}")

                # Optional save
                if DEFAULT_SAVE_MASK:
                    masks_dir = f"out/{save_name}/masks"
                    os.makedirs(masks_dir, exist_ok=True)
                    win_number = (window_idx + 1) if window_idx is not None else 1
                    win_tag = f"w {win_number:02d}"
                    mask_path = os.path.join(masks_dir, f"{win_tag}_data_valid.pkl")
                    mask_df.to_pickle(mask_path)
                    print(f"[INFO] Global mask saved: {mask_path} | Shape: {mask_df.shape} | Mask effective proportion: {mask_df.values.mean():.4%}")
                else:
                    print(f"[INFO] Mask has been generated, shape: {mask_df.shape} | Mask effective proportion: {mask_df.values.mean():.4%}")
            else:
                print("[WARNING] Mask construction returns None, application skipped")
        except Exception as e:
            print(f"[ERROR] Failed to generate/save global mask: {e}")
    else:
        print("[INFO] The mask global switch has been turned off, and mask generation and application has been skipped")
    print("=== [INFO] Data loading completed ===")
    # Data verification and diagnosis
    print("\n" + "="*50)
    print("Data quality check:")
    print("="*50)
    print(f"[INFO] Target variable expression: {target}")
    try:
        target_sample = target.evaluate(data)
        target_nan_ratio = torch.isnan(target_sample).float().mean().item()
        target_finite_count = torch.isfinite(target_sample).sum().item()
        target_total = target_sample.numel()
        print(f"[INFO] Target variable shape: {target_sample.shape}")
        print(f"[INFO] Target variable NaN ratio: {target_nan_ratio:.2%}")
        print(f"[INFO] Valid target value: {target_finite_count:,}/{target_total:,}")
        if target_finite_count > 0:
            finite_targets = target_sample[torch.isfinite(target_sample)]
            print(f"[INFO] Target value statistics: minimum ={finite_targets.min():.6f}, maximum={finite_targets.max():.6f}, mean={finite_targets.mean():.6f}")
        else:
            print(f"[WARNING] No valid target value!")
    except Exception as e:
        print(f"[ERROR] Target variable evaluation failed: {e}")
    try:
        data_tensor = data.data
        data_nan_ratio = torch.isnan(data_tensor).float().mean().item()
        data_finite_count = torch.isfinite(data_tensor).sum().item()
        data_total = data_tensor.numel()
        print(f"[INFO] Data tensor NaN scale: {data_nan_ratio:.2%}")
        print(f"[INFO] Valid data points: {data_finite_count:,}/{data_total:,}")
        if data_finite_count > 0:
            finite_data = data_tensor[torch.isfinite(data_tensor)]
            print(f"[INFO] Data value statistics: minimum ={finite_data.min():.6f}, maximum={finite_data.max():.6f}, mean={finite_data.mean():.6f}")
    except Exception as e:
        print(f"[ERROR] Data tensor check failed: {e}")

    # Optional target variable NaN sample cleaning
    if clean_target_nan:
        data, nan_clean_result = clean_target_nan_samples(data, target, verbose=True)
    else:
        print(f"[INFO] The target variable NaN sample cleaning function is disabled")

    print("="*50)
    # Initialization factor pool
    if factor_pool_in is not None:
        zoo_blds = factor_pool_in
        # Clear old evaluation results because the new window has a different time range
        zoo_blds.examined = False
        zoo_blds.scores = []
        zoo_blds.multi_scores = []
        zoo_blds.ret_list = []
        print(f"[INFO] Inherited factor pool: {zoo_blds.batch_size} factors, old evaluation results have been cleared")
    else:
        zoo_blds = Builders(0, max_len=20, n_actions=SIZE_ACTION)

    for seed in seeds:
        reseed_everything(seed)

        import types
        cfg = types.SimpleNamespace(
            name=f'{save_name}_{instruments}_{train_start}_{seed}',
            max_len=20,
            batch_size=256,
            potential_size=100,
            n_layers=2,
            d_model=128,
            dropout=0.2,
            num_factors=zoo_size,
            num_epochs_g=200,
            g_es_score='max',
            g_es=10,
            g_hidden=128,
            g_lr=1e-3,
            p_hidden=128,
            p_lr=1e-3,
            es_p=10,
            batch_size_p=64,
            num_epochs_p=100,
            data_keep_p=20000,
            f_corr_thresh=corr_thresh,
            f_add_thresh=corr_thresh,
            f_score_thresh=ic_thresh,
            f_multi_score_thresh={'icir': icir_thresh},
            f_ric_thresh=ric_thresh,
            f_ricir_thresh=ricir_thresh,
            f_valid_ratio_thresh=0.1,
            f_unique_ratio_thresh=0.005,
            enable_rankic=enable_rankic,
            # ===== Generator loss function weight configuration =====
            l_pred=1,           # Predictor loss weight: encourage the generation of high-quality expressions (default 1, recommended range: 0.1-10.0)
            l_simi=10.,          # Similarity loss weight: encourage expression diversity and avoid duplication (default 10, recommended range: 1.0-50.0)
            l_simi_thresh=0.4,   # Similarity threshold: only punish when the expression similarity exceeds this value (recommended range: 0.2-0.8)
            l_potential=10.,     # Latent space loss weight: Encourage distinctive features in the predictor latent space (default 10, recommended range: 1.0-50.0)
            l_potential_thresh=0.4,  # Latent space similarity threshold: only punish when the latent feature similarity exceeds this value (recommended range: 0.2-0.8)
            l_potential_epsilon=1e-7, # Latent space epsilon: avoid numerically unstable clipping boundaries
            l_entropy=0,         # Entropy loss weight: encourage uncertainty and avoid deterministic output (default 0, recommended range: 0.0-1.0, currently disabled)
            # ======================================
            device=str(device_str),
            plot_loss=plot_loss,  # Whether to draw the epoch-loss diagram (passed in from parameters)
            use_y_preprocessing=use_y_preprocessing,  # Whether to enable y preprocessing (passed in from parameters)
            use_excess_ic=use_excess_ic,  # Whether to use excess IC as predictor target (for diagnostic printing and metric configuration)
            only_corr_rebalance=only_corr_rebalance, # Relevance rebalancing switch only
        )
        # Configuration parameter print, strictly align train_AFF_test.py format
        print("\n" + "="*50)
        print("Training parameter configuration:")
        print("="*50)
        print(f"Training rounds: {round_num}")
        print(f"IC threshold: {cfg.f_score_thresh}")
        print(f"ICIR threshold: {cfg.f_multi_score_thresh.get('icir', 'Not set')}")
        if cfg.enable_rankic:
            print(f"RankIC threshold: {cfg.f_ric_thresh}")
            print(f"RankICIR threshold: {cfg.f_ricir_thresh}")
        print(f"Effective sample proportion threshold: {cfg.f_valid_ratio_thresh}")
        print(f"Unique value proportion threshold: {cfg.f_unique_ratio_thresh}")
        print(f"Factor correlation threshold: {cfg.f_corr_thresh}")
        print("="*50)

        # ==================== Model configuration selection ====================

        print("Model configuration selection")
        print("="*50)

        if model_config == 'cnn_cnn':
            # CNN + CNN (recommended): efficient and stable
            from gan.network.predictor import NetP_CNN
            from gan.network.generater import NetG_CNN
            NetG_CLS = NetG_CNN
            NetP_CLS = NetP_CNN
            config_name = "CNN + CNN"

        elif model_config == 'dcgan_netp':
            # DCGAN + NetP: high quality generation
            from gan.network.predictor import NetP
            from gan.network.generater import NetG_DCGAN
            NetG_CLS = NetG_DCGAN
            NetP_CLS = NetP
            config_name = "DCGAN + NetP"

        elif model_config == 'lstm_netp':
            # LSTM + NetP: strong sequence modeling
            from gan.network.predictor import NetP
            from gan.network.generater import NetG_Lstm
            NetG_CLS = NetG_Lstm
            NetP_CLS = NetP
            config_name = "LSTM + NetP"

        else:
            raise ValueError(f"Unsupported model configuration: {model_config}. Support: ’cnn_cnn’, ’dcgan_netp’, ’lstm_netp’")

        # Print model configuration information
        print(f"Generator model: {NetG_CLS.__name__}")
        print(f"Predictor model: {NetP_CLS.__name__}")
        print(f"Model configuration: {config_name}")
        print("="*50)
        def random_call(z):
            return z.normal_()

        # Use different parameters according to different generator types
        if model_config == 'lstm_netp':
            # LSTM generator uses different parameter names
            netG = NetG_CLS(
                n_chars=SIZE_ACTION,
                n_layers=cfg.n_layers,
                d_model=cfg.d_model,
                dropout=cfg.dropout,
                seq_len=cfg.max_len,
                potential_size=cfg.potential_size
            ).to(device)
        else:
            # DCGAN and CNN generators use standard parameters
            netG = NetG_CLS(
                n_chars=SIZE_ACTION,
                latent_size=cfg.potential_size,
                seq_len=cfg.max_len,
                hidden=cfg.g_hidden
            ).to(device)

        netM = NetM(max_len=cfg.max_len, size_action=SIZE_ACTION).to(device)
        netP = NetP_CLS(n_chars=SIZE_ACTION, seq_len=cfg.max_len, hidden=cfg.p_hidden).to(device)
        z = torch.zeros([cfg.batch_size, cfg.potential_size], device=device)
        random_call(z)
        empty_metric = get_metric(Builders(0, max_len=cfg.max_len, n_actions=SIZE_ACTION), device=device, corr_thresh=cfg.f_corr_thresh, enable_rankic=cfg.enable_rankic)
        # Stage 1: Initial expression collection
        print("\n" + "="*50)
        print("Stage 1: Initial expression collection")
        print("="*50)
        # reset only once before collection
        coll = Collector(seq_len=cfg.max_len, n_actions=SIZE_ACTION)
        # Calculate initial benchmark income
        if use_excess_ic and benchmark_returns is None:
            initial_benchmark = calculate_benchmark_returns(data)
        else:
            initial_benchmark = benchmark_returns

        initial_metric = get_training_metric(device='cpu', use_excess_ic=use_excess_ic, benchmark_returns=initial_benchmark)
        coll.reset(data, target, initial_metric)
        coll.collect_target_num(netG, netM, z, data, target, initial_metric, target_num=10000, reset_net=True, drop_invalid=False, randomly=False, random_method=random_call, max_iter=200)
        # Only keep the collection completed print
        print(f"[INFO] Initial expression collection completed, total collection {coll.blds.batch_size} valid expressions")

        # Phase 2: iterative training mining phase
        print(f"\n" + "="*50)
        print("Phase 2: iterative training mining phase")
        print("="*50)
        for t in range(round_num):
            if window_idx is not None and total_windows is not None:
                # Add seed identification in training round information
                seed_info = f"[SEED {current_seed}] " if current_seed is not None else ""
                print(f"\n--- {seed_info}[WINDOW {window_idx+1}/{total_windows}] (TRAIN: {train_start}~{train_end}) [ROUND {t+1}/{round_num}] N_BLDS: {len(zoo_blds)} ---")
            else:
                # Add seed identification in training round information
                seed_info = f"[SEED {current_seed}] " if current_seed is not None else ""
                print(f"\n--- {seed_info}[ROUND {t+1}/{round_num}] N_BLDS: {len(zoo_blds)}")
            import time
            round_start_time = time.time()
            try:
                if not zoo_blds.examined:
                    print("[INFO] Evaluation factor pool expression...")
                    zoo_blds.evaluate(data, target, empty_metric, verbose=True)
                metric = get_metric(zoo_blds, device='cpu', corr_thresh=cfg.f_corr_thresh, enable_rankic=cfg.enable_rankic)

                # Calculate base income
                if use_excess_ic and benchmark_returns is None:
                    # Dynamically calculate benchmark income
                    calculated_benchmark = calculate_benchmark_returns(data)
                    print(f"[INFO] Use the current window dynamic benchmark return (market average return)")
                else:
                    calculated_benchmark = benchmark_returns

                training_metric = get_training_metric(device='cpu', use_excess_ic=use_excess_ic, benchmark_returns=calculated_benchmark)
                print("[INFO] Prepare training data...")
                # Recovery condition evaluation - only evaluate when needed
                # Force re-evaluation during the first training to ensure that the correct metric is used
                if t == 0 or not coll.blds.examined:
                    print(f"[INFO] Evaluate the currently collected expression (turn {t+1})...")
                    coll.blds.evaluate(data, target, training_metric, verbose=True)
                if coll.blds_bak.batch_size > cfg.data_keep_p:
                    print(f'[INFO] Training data sampling: {coll.blds_bak.batch_size} -> {cfg.data_keep_p}')
                    indices = np.random.choice(np.arange(coll.blds_bak.batch_size), cfg.data_keep_p, replace=False)
                    coll.blds_bak = coll.blds_bak.filter_by_index(indices)
                if not coll.blds_bak.examined:
                    # The backup pool evaluation also uses metric consistent with training, ensuring that multi_scores contains ’excess_ic’
                    backup_metric = get_training_metric(device='cpu', use_excess_ic=use_excess_ic, benchmark_returns=calculated_benchmark)
                    coll.blds_bak.evaluate(data, target, backup_metric, verbose=True)
                if coll.blds_bak.batch_size > 0:
                    blds_list = [coll.blds_bak, coll.blds]
                    weight_list = [1, 2]
                else:
                    blds_list = [coll.blds]
                    weight_list = [1]
                x, y, weights = blds_list_to_tensor(blds_list, weight_list)

                # y preprocessing
                if getattr(cfg, 'use_y_preprocessing', True):  # Enabled by default
                    y = pre_process_y(y)
                else:
                    print("[INFO] y preprocessing disabled, use original IC value")
                print("[INFO] Start training predictor (Predictor)...")
                netP.initialize_parameters()
                train_net_p_with_weight(cfg, netP, x, y, weights, device, lr=cfg.p_lr, use_adaptive_loss=use_adaptive_loss, filter_zero_targets=filter_zero_targets)
                print("[INFO] Predictor training completed")
                print("[INFO] Start training generator (Generator)...")
                netG.initialize_parameters()
                blds_in_train = train_network_generator(netG, netM, netP, cfg, data, target, t, random_method=random_call, metric=training_metric, lr=cfg.g_lr, n_actions=SIZE_ACTION)
                print("[INFO] Generator training completed")
                print("[INFO] Generate a new Alpha expression...")
                # Expression collection strategy selection
                if use_adaptive_collection:
                    # Round-by-round strategy: retain the initial expression in the first round, and follow-up rounds are normal reset
                    if t == 0:
                        print("[INFO] Round strategy - first round: retain initial expression to participate in screening")
                        coll.collect_target_num(netG, netM, z, data, target, training_metric, target_num=1000, reset_net=False, drop_invalid=False, randomly=False, random_method=random_call, max_iter=100)
                    else:
                        print("[INFO] Round strategy - subsequent rounds: normal reset process")
                        coll.reset(data, target, training_metric)
                        coll.collect_target_num(netG, netM, z, data, target, training_metric, target_num=1000, reset_net=False, drop_invalid=False, randomly=False, random_method=random_call, max_iter=100)
                else:
                    # Unified strategy: reset every time (consistent with default/train_AFF.py and advanced-operators branches)
                    print("[INFO] Unified strategy: reset and regenerate the expression every time")
                    coll.reset(data, target, training_metric)
                    coll.collect_target_num(netG, netM, z, data, target, training_metric, target_num=1000, reset_net=False, drop_invalid=False, randomly=False, random_method=random_call, max_iter=100)
                lengh_s = {"train": len(blds_in_train)}
                lengh_s['new'] = len(coll.blds)
                coll.blds = coll.blds + blds_in_train
                coll.blds.drop_duplicated()
                lengh_s['all_new'] = len(coll.blds)
                print(f"[INFO] Expression merging: training generation {lengh_s['train']} + new collection {lengh_s['new']} = total {lengh_s['all_new']}")
                print("[INFO] Screen high-quality factors and expand the factor pool...")
                full_metric = get_metric(coll.blds, device='cpu', corr_thresh=cfg.f_corr_thresh, enable_rankic=cfg.enable_rankic, use_excess_ic=use_excess_ic, benchmark_returns=calculated_benchmark)
                coll.blds.evaluate(data, target, full_metric, verbose=True)

                # Pre-screening diagnosis (before screening, output the mean/standard deviation/range of the main indicators)
                try:
                    print(f"[DIAG] ============ Detailed diagnosis before screening ============")
                    all_scores = coll.blds.scores
                    ic_vals = np.array(list(all_scores), dtype=float)
                    if ic_vals.size > 0:
                        print(f"[DIAG] IC: mean={np.mean(ic_vals):.6f}, std={np.std(ic_vals):.6f}, range=[{np.min(ic_vals):.6f}, {np.max(ic_vals):.6f}]")
                    else:
                        print(f"[DIAG] IC: No valid data")
                    ms_list = coll.blds.multi_scores
                    def metric_array(key: str):
                        vals = []
                        for ms in ms_list:
                            v = ms.get(key, None)
                            if v is None or not np.isfinite(v):
                                continue
                            vals.append(float(v))
                        return np.array(vals, dtype=float)
                    icir_vals = metric_array('icir')
                    if icir_vals.size > 0:
                        print(f"[DIAG] ICIR: mean={np.mean(icir_vals):.6f}, std={np.std(icir_vals):.6f}, range=[{np.min(icir_vals):.6f}, {np.max(icir_vals):.6f}]")
                    else:
                        print(f"[DIAG] ICIR: No valid data")
                    ret_vals = metric_array('ret')
                    if ret_vals.size > 0:
                        print(f"[DIAG] RET: mean={np.mean(ret_vals):.6f}, std={np.std(ret_vals):.6f}, range=[{np.min(ret_vals):.6f}, {np.max(ret_vals):.6f}]")
                    else:
                        print(f"[DIAG] RET: No valid data")
                    sharpe_vals = metric_array('sharpe')
                    if sharpe_vals.size > 0:
                        print(f"[DIAG] SHARPE: mean={np.mean(sharpe_vals):.6f}, std={np.std(sharpe_vals):.6f}, range=[{np.min(sharpe_vals):.6f}, {np.max(sharpe_vals):.6f}]")
                    else:
                        print(f"[DIAG] SHARPE: No valid data")
                    # EXCESS_IC
                    excess_ic_vals = metric_array('excess_ic')
                    if getattr(cfg, 'use_excess_ic', False):
                        if excess_ic_vals.size > 0:
                            print(f"[DIAG] EIC: mean={np.mean(excess_ic_vals):.6f}, std={np.std(excess_ic_vals):.6f}, range=[{np.min(excess_ic_vals):.6f}, {np.max(excess_ic_vals):.6f}]")
                        else:
                            print(f"[DIAG] EIC: No valid data")
                    print("[DIAG] ========================================")
                except Exception as _:
                    print("[DIAG] Pre-screening: Statistics failed")
                new_zoo = filter_valid_blds(
                    coll.blds,
                    corr_thresh=cfg.f_add_thresh,
                    score_thresh=cfg.f_score_thresh,
                    multi_score_thresh=cfg.f_multi_score_thresh,
                    valid_ratio_thresh=cfg.f_valid_ratio_thresh,
                    unique_ratio_thresh=cfg.f_unique_ratio_thresh,
                    ric_thresh=cfg.f_ric_thresh if cfg.enable_rankic else None,
                    ricir_thresh=cfg.f_ricir_thresh if cfg.enable_rankic else None,
                    device='cpu',
                    verbose=True,
                    only_corr=False,
                )
                zoo_blds = zoo_blds + new_zoo
                if t % 5 == 2:
                    print(f"[INFO] Perform factor pool rebalancing... (Current factor pool size: {len(zoo_blds)})")
                    rebalance_zoo = filter_valid_blds(
                        zoo_blds,
                        corr_thresh=cfg.f_add_thresh,
                        score_thresh=cfg.f_score_thresh,
                        multi_score_thresh=cfg.f_multi_score_thresh,
                        valid_ratio_thresh=cfg.f_valid_ratio_thresh,
                        unique_ratio_thresh=cfg.f_unique_ratio_thresh,
                        ric_thresh=cfg.f_ric_thresh if cfg.enable_rankic else None,
                        ricir_thresh=cfg.f_ricir_thresh if cfg.enable_rankic else None,
                        device='cpu',
                        verbose=True,
                        # When rebalancing, only correlation can be used according to the configuration
                        only_corr=getattr(cfg, 'only_corr_rebalance', False),
                    )
                    print(f"[INFO] Rebalance results: {len(zoo_blds)} -> {len(rebalance_zoo)}")
                    zoo_blds = rebalance_zoo
                    print(f"[INFO] Factor pool size after rebalancing: {len(zoo_blds)}")
                save_blds(zoo_blds, f"out/{save_name}", 'zoo_final')
                print(f"[INFO] The factor pool has been saved to: out/{save_name}")
                del x, y, weights
                gc.collect()
                torch.cuda.empty_cache()
                round_end_time = time.time()
                round_duration = round_end_time - round_start_time
                print(f"[INFO] Training rounds {t+1} Completed, time consuming: {round_duration:.1f} seconds ({round_duration/60:.1f} minutes)")
                t += 1
            except Exception as e:
                round_end_time = time.time()
                round_duration = round_end_time - round_start_time
                print(f"[ERROR] Training rounds {t+1} An error occurred: {e}")
                print(f"[INFO] Wrong round time consumption: {round_duration:.1f} seconds ({round_duration/60:.1f} minutes)")
                import traceback; traceback.print_exc()
                break
        print(f"\n" + "="*50)
        print("Phase 3: Training completion and final evaluation")
        print("="*50)
        empty_blds = Builders(0, max_len=cfg.max_len, n_actions=SIZE_ACTION)
        final_metric = get_metric(empty_blds, device='cpu', corr_thresh=cfg.f_corr_thresh, enable_rankic=cfg.enable_rankic)
        print("[INFO] Evaluate the final factor pool...")
        if not zoo_blds.examined:
            zoo_blds.evaluate(data, target, final_metric, verbose=True)
        else:
            print("[INFO] Factor pool has been evaluated, skip repeated evaluation")
        # Finally save the factor pool to the total training folder
        # [SW] Under sliding window, rebalance before final saving
        if window_idx is not None and total_windows is not None:
            print("[SW] Factor pool rebalancing before final saving...")
            before_size = zoo_blds.batch_size
            rebalance_zoo = filter_valid_blds(
                zoo_blds,
                corr_thresh=cfg.f_add_thresh,
                score_thresh=cfg.f_score_thresh,
                multi_score_thresh=cfg.f_multi_score_thresh,
                valid_ratio_thresh=cfg.f_valid_ratio_thresh,
                unique_ratio_thresh=cfg.f_unique_ratio_thresh,
                ric_thresh=cfg.f_ric_thresh if cfg.enable_rankic else None,
                ricir_thresh=cfg.f_ricir_thresh if cfg.enable_rankic else None,
                device='cpu',
                verbose=True,
                only_corr=getattr(cfg, 'only_corr_rebalance', False),
            )
            after_size = rebalance_zoo.batch_size
            print(f"[SW] Factor pool size after rebalancing: {before_size} -> {after_size}")
            zoo_blds = rebalance_zoo
            # Optional: Save the factor pool correlation coefficient matrix after rebalancing
            if DEFAULT_SAVE_REBALANCE_CORR and len(zoo_blds) > 1:
                try:
                    # The sorting and factor number consistent with csv_zoo_final: in descending order of scores
                    df_for_order = pd.DataFrame({'scores': zoo_blds.scores})
                    df_for_order = df_for_order.sort_values('scores', ascending=False)
                    order_idx = df_for_order.index.tolist()
                    rets = np.vstack([zoo_blds.ret_list[i] for i in order_idx])  # (n_factors, n_days)
                    corr = np.corrcoef(rets)
                    idx = order_idx
                    out_dir = os.path.join(f"out/{save_name}")
                    os.makedirs(out_dir, exist_ok=True)
                    # Save csv (column and column names are factor numbers, four decimal places)
                    corr_df = pd.DataFrame(np.round(corr, 4), index=idx, columns=idx)
                    corr_path = os.path.join(out_dir, "corr.csv")
                    corr_df.to_csv(corr_path)
                    # Calculate the mean correlation (excluding diagonals, only upper triangles), consistent with corr_cal
                    try:
                        iu = np.triu_indices_from(corr, k=1)
                        tri_vals = corr[iu]
                        tri_vals = tri_vals[np.isfinite(tri_vals)]
                        mean_corr = float(tri_vals.mean()) if tri_vals.size > 0 else float('nan')
                        mean_abs_corr = float(np.abs(tri_vals).mean()) if tri_vals.size > 0 else float('nan')
                        num_factors = corr.shape[0]  # The number of factors is equal to the number of rows of the matrix
                        # Write to text (controlled by global switch) and print
                        if DEFAULT_SAVE_CORR_MEAN_TXT:
                            with open(os.path.join(out_dir, "corr_mean.txt"), "w") as f:
                                f.write(f"num_factors = {num_factors}\n")
                                f.write(f"mean_corr = {mean_corr:.6f}\n")
                                f.write(f"mean_abs_corr = {mean_abs_corr:.6f}\n")
                        print(f"[INFO] The correlation matrix after rebalancing has been saved: {corr_path} | Shape: {corr.shape}")
                        print(f"[INFO] num_factors={num_factors} | mean(corr)={mean_corr:.6f} | mean(|corr|)={mean_abs_corr:.6f}")
                    except Exception as _e:
                        print(f"[WARNING] Correlation mean calculation failed: {_e}")
                except Exception as e:
                    print(f"[WARNING] Failed to save rebalance correlation matrix: {e}")
        save_blds(zoo_blds, f"out/{save_name}", 'zoo_final')
        print(f"[INFO] The final factor pool has been saved to: out/{save_name}")
        # Calculate final statistical information
        final_scores = [score for score in zoo_blds.scores if score > 0]
        if final_scores:
            max_score = max(final_scores)
            avg_score = np.mean(final_scores)
            std_score = np.std(final_scores)
            min_score = min(final_scores)
            print(f"[SEED {seed}] WINDOW {window_idx+1}/{total_windows} Training completed | Factor pool: {len(zoo_blds)}/{cfg.num_factors}")
            print(f"[SCORE] Mean: {avg_score:.4f} | Standard deviation: {std_score:.4f} | Range: [{min_score:.4f},{max_score:.4f}]")
        else:
            print(f"[WARNING] No valid factor found!")
        save_blds(zoo_blds, f"out/{save_name}", 'zoo_final')
        print(f"[INFO] Result saved: out/{save_name}")

    # Summary after completion of all seed training
    print(f"\n" + "="*100)
    print("="*100)
    print("[Sliding window training master] All seed training completed")
    print("="*100)
    print(f"[INFO] The results are saved in the following directory:")
    for seed in seeds:
        result_dir = f"out/{save_name}_{instruments}_{train_start}_{seed}"
        print(f"  - seed {seed}: {result_dir}")
    print("="*100)
    print("="*100)
    return zoo_blds
