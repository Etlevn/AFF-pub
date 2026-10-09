import torch
import os
import json
import time
from typing import Optional
from datetime import datetime

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, SingleAlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore

# ======================== RL CONFIG ========================

DEFAULT_INSTRUMENTS = "all"
DEFAULT_FREQ = 'day'
DEFAULT_SEEDS = [0]

DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END   = "2023-12-31"
DEFAULT_VALID_START = "2024-01-01"
DEFAULT_VALID_END   = "2024-12-31"
DEFAULT_TEST_START  = "2025-01-01"
DEFAULT_TEST_END    = "2025-06-30"

DEFAULT_CUDA = '0'
DEFAULT_DEVICE = 'cpu'

DEFAULT_NAME_SUFFIX = "test"

# The default number of training steps corresponding to different pool capacities
DEFAULT_POOL_CAPACITIES = [10, 20, 50, 100]
DEFAULT_STEPS_BY_CAPACITY = {
    10: 10_000, # 250_000
    20: 10_000, # 300_000
    50: 10_000, # 350_000
    100: 10_000, # 400_000
}


DEFAULT_N_STEPS = 2048                           # The number of steps collected each time rollout

# Log and output control
DEFAULT_TENSORBOARD_LOG = True                   # Whether to enable TensorBoard log
DEFAULT_SAVE_CHECKPOINTS = True                  # Whether to save checkpoints
DEFAULT_SAVE_ICS_RET = True                      # Whether to include single factor IC when saving the pool file

DEFAULT_SAVE_RL_POOL = True                      # Whether to generate/update rl_pool.json (summary of optimal combinations of each capacity)
DEFAULT_RL_POOL_SELECT_METRIC = 'best_ic_ret'    # Combined optimal screening index (currently supported: ’best_ic_ret’)

DEFAULT_SAVE_RL_SGL = True                       # Whether to generate/update rl_sgl.json (global optimal single factor)
DEFAULT_SGL_METRIC = 'ics_ret'                   # Single factor screening index (currently supported: ’ics_ret’)
DEFAULT_TOP_SINGLE_COUNT = 10                    # Single factor candidate Top-N

DEFAULT_SAVE_RL_SUM = True                       # Whether to save step-by-step training summary rl_sum.csv

# ============================================================

import pickle
def save_pickle(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)
def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 train_data: StockData,
                 train_target: Expression,
                 valid_data: StockData,
                 valid_target: Expression,
                 test_data: StockData,
                 test_target: Expression,
                 name_prefix: str = 'rl_model',
                 total_timesteps: int = 0,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.train_data = train_data
        self.train_target = train_target
        self.valid_data = valid_data
        self.valid_target = valid_target
        self.test_data = test_data
        self.test_target = test_target
        self.total_timesteps = total_timesteps
        self._rollout_ended = False

        # Variables used to track time-consuming policy network updates
        self.policy_update_start_time = None

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)



    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # If the previous rollout has ended, display the progress prompt (after the SB3 table is output and before the expression is displayed)
        if hasattr(self, '_rollout_ended') and self._rollout_ended:
            # Calculate the actual time consumption of policy network update and display the number of rounds
            if self.policy_update_start_time is not None:
                elapsed_time = time.time() - self.policy_update_start_time
                current_round = self.num_timesteps // DEFAULT_N_STEPS
                print(f"[PROG] UPDATE ({elapsed_time:.1f} s)")
                self.policy_update_start_time = None
            self._rollout_ended = False  # Reset flag

        # Display the current round number start prompt
        current_round = (self.num_timesteps // DEFAULT_N_STEPS) + 1
        # Calculate training progress (using the total number of steps passed in)
        total_steps = getattr(self, 'total_timesteps', 0)
        progress_pct = (self.num_timesteps / total_steps * 100) if total_steps > 0 else 0
        # Get the current pool status
        pool_size = self.pool.size if hasattr(self, 'pool') else 0
        best_ic = self.pool.best_ic_ret if hasattr(self, 'pool') and hasattr(self.pool, 'best_ic_ret') else 0.0
        eval_count = self.pool.eval_cnt if hasattr(self, 'pool') and hasattr(self.pool, 'eval_cnt') else 0
        print("\n"+"=" * 40+"\n")
        print(f"ROUND {current_round} | STEPS {current_round * DEFAULT_N_STEPS} | PROGRESS {progress_pct:.2f}%")
        print(f"POOL size={pool_size} | best_IC={best_ic:.5f} | evals={eval_count}")
        print("\n"+"=" * 40+"\n")

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        # Record to SB3 logger
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)

        ic_train, rank_ic_train = self.pool.test_ensemble(self.train_data, self.train_target)
        self.logger.record('_train/ic', ic_train)
        self.logger.record('_train/rank_ic', rank_ic_train)

        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_data, self.test_target)
        self.logger.record('_test/ic', ic_test)
        self.logger.record('_test/rank_ic', rank_ic_test)

        ic_valid, rank_ic_valid = self.pool.test_ensemble(self.valid_data, self.valid_target)
        self.logger.record('_valid/ic', ic_valid)
        self.logger.record('_valid/rank_ic', rank_ic_valid)

        # Save checkpoint: restore old behavior - save every time rollout ends (controlled by switch)
        if DEFAULT_SAVE_CHECKPOINTS:
            self.save_checkpoint()
            # Synchronize/update global optimal summary file rl_pool.json
            if DEFAULT_SAVE_RL_POOL:
                try:
                    self._update_experiment_pool_summary()
                except Exception:
                    pass
            # Synchronize/update global optimal single factor rl_sgl.json
            if DEFAULT_SAVE_RL_SGL:
                try:
                    self._update_experiment_single_summary()
                except Exception:
                    pass

        # Setting the flag indicates that rollout has ended
        self._rollout_ended = True

        # Start Policy Network Update Indicator
        # Record policy network update start time
        self.policy_update_start_time = time.time()


    def save_checkpoint(self):
        # Save directly under the capacity subdirectory: {checkpoints}/{pool_capacity}/{num_steps}_steps
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        # Display the saved results
        print(f"\n[SAVE] checkpoint -> {path}\n")
        save_pickle(self.pool,path+'_pool.pkl')
        with open(f'{path}_pool.json', 'w') as f:
            # Select the export structure according to the global configuration
            if DEFAULT_SAVE_ICS_RET:
                json.dump(self._serializable_pool_state(), f, ensure_ascii=False, indent=2)
            else:
                json.dump(self.pool.to_dict(), f, ensure_ascii=False, indent=2)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _serializable_pool_state(self) -> dict:
        state = self.pool.state
        return {
            'exprs': [str(e) for e in state.get('exprs', [])],
            'weights': list(state.get('weights', [])),
            'ics_ret': list(state.get('ics_ret', [])),
            'best_ic_ret': state.get('best_ic_ret', 0.0),
        }

    def _serializable_portfolio(self) -> dict:
        # Save only what is needed for the combination: expressions and weights
        state = self.pool.state
        return {
            'exprs': [str(e) for e in state.get('exprs', [])],
            'weights': list(state.get('weights', [])),
        }

    def _update_experiment_pool_summary(self) -> None:
        # save_path: out_rl/{experiment}/checkpoints/{capacity}
        # Target: out_rl/{experiment}/rl_pool.json (structural benchmark gp_pool.json)
        experiment_dir = os.path.dirname(os.path.dirname(self.save_path))
        summary_path = os.path.join(experiment_dir, 'rl_pool.json')

        capacity = int(getattr(self.pool, 'capacity', 0))
        # Select the indicators for model selection according to the global configuration (currently only best_ic_ret is supported)
        if DEFAULT_RL_POOL_SELECT_METRIC == 'best_ic_ret':
            current_metric_value = float(getattr(self.pool, 'best_ic_ret', 0.0))
        else:
            current_metric_value = float(getattr(self.pool, 'best_ic_ret', 0.0))

        # Read the existing best
        prev = None
        try:
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as rf:
                    prev = json.load(rf)
        except Exception:
            prev = None

        should_update = False
        if prev is None:
            should_update = True
        else:
            try:
                prev_best = float(prev.get('metrics', {}).get('best_ic_ret', float('-inf')))
                if current_metric_value > prev_best:
                    should_update = True
            except Exception:
                should_update = True

        if should_update:
            record = {
                'step': int(self.num_timesteps),
                'pool': capacity,
                'metrics': {
                    'best_ic_ret': current_metric_value
                },
                'portfolio': self._serializable_portfolio()
            }
            with open(summary_path, 'w') as wf:
                json.dump(record, wf, ensure_ascii=False, indent=2)

        # Synchronous writing rl_sum.csv (controlled by switch)
        if DEFAULT_SAVE_RL_SUM:
            sum_csv = os.path.join(experiment_dir, 'rl_sum.csv')
            header_needed = not os.path.exists(sum_csv)
            with open(sum_csv, 'a') as fcsv:
                if header_needed:
                    fcsv.write('pool,step,best_ic_ret\n')
                fcsv.write(f"{capacity},{int(self.num_timesteps)},{current_metric_value}\n")

    def _update_experiment_single_summary(self) -> None:
        # Target: out_rl/{experiment}/rl_sgl.json (structure benchmark gp_sgl.json: Top-N list)
        experiment_dir = os.path.dirname(os.path.dirname(self.save_path))
        summary_path = os.path.join(experiment_dir, 'rl_sgl.json')

        # Current pool single factor information
        state = self.pool.state
        exprs = list(state.get('exprs', []))
        ics = list(state.get('ics_ret', []))
        if not exprs or not ics:
            return

        if DEFAULT_SGL_METRIC != 'ics_ret':
            return

        # Take the candidates of the current pool and merge them with the historical files to make global Top-N
        import numpy as _np
        order = _np.argsort(_np.array(ics))[::-1]
        current_pairs = [(str(exprs[int(i)]), float(ics[int(i)])) for i in order]

        # Merge history (if present)
        merged: dict = {}
        try:
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as rf:
                    history = json.load(rf)
                if isinstance(history, list):
                    for it in history:
                        e = it.get('expr')
                        v = it.get('ic')
                        if isinstance(e, str) and isinstance(v, (int, float)):
                            merged[e] = max(float(v), merged.get(e, float('-inf')))
        except Exception:
            pass

        # Merge current
        for e, v in current_pairs:
            merged[e] = max(v, merged.get(e, float('-inf')))

        # Global sorting takes Top-N
        top_n = min(DEFAULT_TOP_SINGLE_COUNT, len(merged))
        top_items = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out_list = [{ 'idx': i, 'expr': e, 'ic': v } for i, (e, v) in enumerate(top_items)]

        with open(summary_path, 'w') as wf:
            json.dump(out_list, wf, ensure_ascii=False, indent=2)

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore

def main(
    seed: int = 0,
    instruments: str = DEFAULT_INSTRUMENTS,
    pool_capacity: int = 10,
    steps: int = 200_000,
    raw: bool = False,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    freq: str = DEFAULT_FREQ,
    device: str = DEFAULT_DEVICE,
):
    reseed_everything(seed)

    device = torch.device(device)
    close = Feature(FeatureType.CLOSE)

    from alphagen_generic.features import open_
    from gan.utils import Builders
    from gan.utils.data import get_data_by_year_with_dates
    import os


    reseed_everything(seed)

    # ==================== Data loading phase ====================
    print(f"[DATA] Load data: instruments={instruments}, freq={freq}")
    print(f"[DATA] Time range: train={train_start}~{train_end}, valid={valid_start}~{valid_end}, test={test_start}~{test_end}")

    returned = get_data_by_year_with_dates(
        train_start = train_start, train_end = train_end,
        valid_start = valid_start, valid_end = valid_end,
        test_start = test_start,  test_end = test_end,
        instruments = instruments, target = target, freq = freq,
    )
    data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
    print(f"[DATA] Data loading completed: {name}\n")

    # ==================== Environment initialization phase ====================
    pool = AlphaPool(
        capacity=pool_capacity,
        stock_data=data,
        target=target,
        ic_lower_bound=None
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    # Unified naming: hard-coded prefix rl_ + instruments + year range + seed + optional suffix
    year_start = str(train_start)[:4]
    year_end = str(train_end)[:4]
    base_name = f"rl_{instruments}_{year_start}-{year_end}_{seed}"
    suffix = ("_" + DEFAULT_NAME_SUFFIX) if DEFAULT_NAME_SUFFIX else ""
    experiment_name = f"{base_name}{suffix}"

    # Create experimental directory structure: out_rl/{experiment_name}/{checkpoints,log}/
    experiment_dir = f"out_rl/{experiment_name}"
    checkpoints_dir = f"{experiment_dir}/checkpoints"
    log_dir = f"{experiment_dir}/log"

    # Create subdirectories for different pool capacities
    pool_checkpoint_dir = f"{checkpoints_dir}/{pool_capacity}"

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path=pool_checkpoint_dir,
        train_data=data,
        train_target=target,
        valid_data=data_valid,
        valid_target=target,
        test_data=data_test,
        test_target=target,
        name_prefix=f"{experiment_name}_{pool_capacity}",
        total_timesteps=steps,
        verbose=1,
    )

    # ==================== Model training phase ====================

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        n_steps=DEFAULT_N_STEPS,
        tensorboard_log=log_dir if DEFAULT_TENSORBOARD_LOG else None,
        device=device,
        verbose=1,
    )

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=str(pool_capacity),
    )


from gan.utils.qlib import get_data_my
if __name__ == '__main__':
    # Set CUDA visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_CUDA)

    # Print configuration summary
    print("\n" + "=" * 100)
    print("[RL CONFIG]")
    print(f"  instruments : {DEFAULT_INSTRUMENTS}")
    print(f"  freq        : {DEFAULT_FREQ}")
    print(f"  seeds       : {DEFAULT_SEEDS}")
    print(f"  cuda        : {DEFAULT_CUDA}")
    print(f"  train       : {DEFAULT_TRAIN_START} ~ {DEFAULT_TRAIN_END}")
    print(f"  valid       : {DEFAULT_VALID_START} ~ {DEFAULT_VALID_END}")
    print(f"  test        : {DEFAULT_TEST_START} ~ {DEFAULT_TEST_END}")
    print(f"  capacities  : {DEFAULT_POOL_CAPACITIES}")
    print(f"  steps/cap   : {DEFAULT_STEPS_BY_CAPACITY}")
    print(f"  n_steps     : {DEFAULT_N_STEPS}")
    print(f"  name_suffix : '{DEFAULT_NAME_SUFFIX}'" if DEFAULT_NAME_SUFFIX else "  name_suffix : (none)")
    print(f"  save_ckpt   : {DEFAULT_SAVE_CHECKPOINTS}")
    print(f"  tensorboard : {DEFAULT_TENSORBOARD_LOG}")
    print("=" * 100 + "\n")

    steps = DEFAULT_STEPS_BY_CAPACITY
    instruments_list = [DEFAULT_INSTRUMENTS]
    seeds = DEFAULT_SEEDS
    capacities = DEFAULT_POOL_CAPACITIES

    for capacity in capacities:
        for seed in seeds:
            for instruments in instruments_list:
                main(
                    seed=seed,
                    instruments=instruments,
                    pool_capacity=capacity,
                    steps=steps[capacity],
                    raw=True,
                    train_start=DEFAULT_TRAIN_START,
                    train_end=DEFAULT_TRAIN_END,
                    valid_start=DEFAULT_VALID_START,
                    valid_end=DEFAULT_VALID_END,
                    test_start=DEFAULT_TEST_START,
                    test_end=DEFAULT_TEST_END,
                    freq=DEFAULT_FREQ,
                    device=DEFAULT_DEVICE,
                )
