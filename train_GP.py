
import os

# ======================== GP CONFIG ========================

DEFAULT_INSTRUMENTS = 'all'
DEFAULT_FREQ = 'day'
DEFAULT_SEEDS = [0]

DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END   = "2023-12-31"
DEFAULT_VALID_START = "2024-01-01"
DEFAULT_VALID_END   = "2024-12-31"
DEFAULT_TEST_START  = "2025-01-01"
DEFAULT_TEST_END    = "2025-06-30"

DEFAULT_CUDA = '0'

DEFAULT_NAME_SUFFIX = "test"

# GP training hyperparameters
DEFAULT_GENERATIONS = 5
DEFAULT_POPULATION_SIZE = 300
DEFAULT_TOURNAMENT_SIZE = 50
# Pool optimized hyperparameters
DEFAULT_POOL_OPT_ALPHA = 5e-3
DEFAULT_POOL_OPT_LR = 5e-4
DEFAULT_POOL_OPT_STEPS = 2000

# Log and output control
DEFAULT_SAVE_SNAPSHOTS = True       # Whether to save each round of snapshots to snapshots/

DEFAULT_SAVE_GP_POOL = True         # Whether to save gp_pool.json (best in history)
BEST_PRIMARY_KEY = 'ic_valid'       # The best judgment main standard in the past dynasties
BEST_SECONDARY_KEY = 'ric_valid'    # The best judgment sub-standard in the past dynasties

DEFAULT_SAVE_SINGLE_EXPRS = True    # Whether to save a single expression to gp_sgl.json (global optimal single factor)
DEFAULT_SGL_METRIC = 'ic'           # Single factor screening index (currently supported: ’ic’)
DEFAULT_TOP_SINGLE_COUNT = 10       # Save the first N best single expressions

DEFAULT_SAVE_GP_SUM = True          # Whether to save gp_sum.csv (summary results)

# ============================================================

INSTRUMENTS = DEFAULT_INSTRUMENTS
SEEDS = DEFAULT_SEEDS
TRAIN_START = DEFAULT_TRAIN_START
TRAIN_END = DEFAULT_TRAIN_END
VALID_START = DEFAULT_VALID_START
VALID_END = DEFAULT_VALID_END
TEST_START = DEFAULT_TEST_START
TEST_END = DEFAULT_TEST_END
FREQ = DEFAULT_FREQ
CUDA_VISIBLE = str(DEFAULT_CUDA)
NAME_SUFFIX = DEFAULT_NAME_SUFFIX
POOL_OPT_ALPHA = DEFAULT_POOL_OPT_ALPHA
POOL_OPT_LR = DEFAULT_POOL_OPT_LR
POOL_OPT_STEPS = DEFAULT_POOL_OPT_STEPS
SAVE_SNAPSHOTS = DEFAULT_SAVE_SNAPSHOTS
SAVE_SINGLE_EXPRS = DEFAULT_SAVE_SINGLE_EXPRS
SAVE_GP_POOL = DEFAULT_SAVE_GP_POOL
TOP_SINGLE_COUNT = DEFAULT_TOP_SINGLE_COUNT
SGL_METRIC = DEFAULT_SGL_METRIC
SAVE_GP_SUM = DEFAULT_SAVE_GP_SUM
_BEST_RECORD = None  # In the shape of {’gen’: int, ’pool’: int, ’metrics’: {...}}
_SUMMARY_CSV_PATH = None

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE
print("\n" + "=" * 100)
print("[CONFIG] Training configuration summary")
print(f"[CONFIG] instruments: {INSTRUMENTS}")
print(f"[CONFIG] seeds: {SEEDS}")
print(f"[CONFIG] train: {TRAIN_START} ~ {TRAIN_END}")
print(f"[CONFIG] valid: {VALID_START} ~ {VALID_END}")
print(f"[CONFIG] test : {TEST_START} ~ {TEST_END}")
print(f"[CONFIG] freq: {FREQ}")
print(f"[CONFIG] cuda: {CUDA_VISIBLE}")
print(f"[CONFIG] name_suffix: '{NAME_SUFFIX}'" if NAME_SUFFIX else "[CONFIG] name_suffix: (none)")
print(f"[CONFIG] pool_opt: alpha={POOL_OPT_ALPHA} lr={POOL_OPT_LR} steps={POOL_OPT_STEPS}")
print(f"[CONFIG] save_snapshots: {SAVE_SNAPSHOTS}")
print(f"[CONFIG] save_single_exprs: {SAVE_SINGLE_EXPRS} (top_{TOP_SINGLE_COUNT})")
print(f"[CONFIG] sgl_metric: '{SGL_METRIC}'")
print("=" * 100 + "\n")


import json
from collections import Counter

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as gp_funcs
from alphagen_generic.features import *
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gan.utils.data import get_data_by_year_with_dates
from datetime import datetime, timedelta
import time
from tqdm.auto import tqdm

def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.

    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor)
        ic = torch.nan_to_num(ic).mean().item()
    except OutOfDataRangeError:
        ic = -1.
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic




def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    try:
        v_valid = eval(top_key).evaluate(data_valid)
        v_test = eval(top_key).evaluate(data_test)
        ic_test = batch_pearsonr(v_test, target_factor_test)
        ic_test = torch.nan_to_num(ic_test,nan=0,posinf=0,neginf=0).mean().item()
        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid,nan=0,posinf=0,neginf=0).mean().item()
        ric_test = batch_spearmanr(v_test, target_factor_test)
        ric_test = torch.nan_to_num(ric_test,nan=0,posinf=0,neginf=0).mean().item()
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid,nan=0,posinf=0,neginf=0).mean().item()
        return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}
    except OutOfDataRangeError:
        print ('Out of data range')
        print(top_key)
        exit()
        return {'ic_test': -1., 'ic_valid': -1., 'ric_test': -1., 'ric_valid': -1.}


def try_pool(capacity, prev_exprs=None):
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    # Display a disappearable progress bar when loading an expression
    load_pbar = tqdm(total=len(exprs), desc=f"[DEBUG] Load pool={capacity}", leave=False)
    def _load_hook(incr):
        load_pbar.update(incr)
    try:
        pool.force_load_exprs(exprs, progress_hook=_load_hook)
    finally:
        load_pbar.close()
    # Pre-evaluation (before optimization)
    examples_preview = ", ".join([str(e) for e in exprs[:3]])
    print(f" ----- [pool={capacity}] -----")
    pre_ic_test, pre_ric_test = pool.test_ensemble(data_test, target)
    pre_ic_valid, pre_ric_valid = pool.test_ensemble(data_valid, target)
    print(f"  Pre : IC(test)={pre_ic_test:.6f} IC(valid)={pre_ic_valid:.6f} | RIC(test)={pre_ric_test:.6f} RIC(valid)={pre_ric_valid:.6f}")

    # Optimization
    # Use tqdm progress bar (based on progress_hook real-time update)
    pbar = tqdm(total=POOL_OPT_STEPS, desc=f" Opt pool={capacity}", leave=False, dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]')
    def _hook(it, total, curr_loss, best_loss):
        # it starts to increase from 0 and is affected by early stop; use min to ensure that it does not exceed total
        pbar.n = min(int(it), POOL_OPT_STEPS)
        pbar.set_postfix({
            'loss': f"{curr_loss:.4f}",
            'best': f"{best_loss:.4f}"
        })
        pbar.refresh()
    try:
        pool._optimize(alpha=POOL_OPT_ALPHA, lr=POOL_OPT_LR, n_iter=POOL_OPT_STEPS, progress_hook=_hook)
    finally:
        try:
            pbar.clear()
        except Exception:
            pass
        pbar.close()
    # Display the ”marginally added” Top-3 after optimization (relative to the previous capacity difference set), otherwise fall back to the global Top-3
    try:
        import numpy as _np
        curr_exprs = [str(pool.exprs[i]) for i in range(pool.size)]
        prev_set = set(prev_exprs) if prev_exprs else set()
        diffs = []
        for i in range(pool.size):
            exs = curr_exprs[i]
            if exs not in prev_set:
                diffs.append((abs(float(pool.weights[i])), float(pool.weights[i]), exs))
        if not diffs:
            top_idx = _np.argsort(-_np.abs(pool.weights[:pool.size]))[:3]
            diffs = [(abs(float(pool.weights[i])), float(pool.weights[i]), str(pool.exprs[i])) for i in top_idx]
        # Clearly sort in the order of |w| (descending order), if equal, then use |w| as the secondary keyword to avoid the influence of strings
        diffs.sort(key=lambda t: (t[0], abs(t[1])), reverse=True)
        print(f"  Opt :")
        for _, w, exs in diffs[:3]:
            print(f"    {w:+0.4f} | {exs}")
    except Exception:
        curr_exprs = [str(pool.exprs[i]) for i in range(pool.size)]
        pass

    # Post-evaluation (after optimization)
    ic_test, ric_test = pool.test_ensemble(data_test, target)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    print(f"  Post: IC(test)={ic_test:.6f} IC(valid)={ic_valid:.6f} | RIC(test)={ric_test:.6f} RIC(valid)={ric_valid:.6f}")
    # Return the expression and corresponding weight to facilitate saving the ”best combination” in gp.json
    factors_expr = [str(pool.exprs[i]) for i in range(pool.size)]
    factors_w = [float(pool.weights[i]) for i in range(pool.size)]
    return {
        'ic_test': ic_test,
        'ic_valid': ic_valid,
        'ric_test': ric_test,
        'ric_valid': ric_valid,
        'exprs': factors_expr,
        'weights': factors_w
    }, set(curr_exprs)




def ev():
    global generation
    generation += 1
    print(f"\n[GP] ========== Generation {generation} Start ==========")
    _t_gen = time.time()
    res = []
    # res.append({'pool': 0, 'res': try_single()})
    prev_exprs = set()
    for cap in (10, 20, 50, 100):
        r, curr_exprs = try_pool(cap, prev_exprs)
        res.append({'pool': cap, 'res': r})
        prev_exprs = curr_exprs
    # Select the best of this generation based on the validation set indicators (first ic_valid, then ric_valid)
    def _best_key(item):
        r = item['res']
        return (r.get(BEST_PRIMARY_KEY, float('-inf')), r.get(BEST_SECONDARY_KEY, float('-inf')))
    best_this_gen = max(res, key=_best_key)

    # Maintain the best in the past (save the expression and weight of the pool as the ”best combination”)
    global _BEST_RECORD
    if (_BEST_RECORD is None) or (_best_key(best_this_gen) > _best_key({'res': _BEST_RECORD['metrics']})):
        _metrics_src = best_this_gen['res']
        _metrics_clean = {
            'ic_test': _metrics_src.get('ic_test'),
            'ic_valid': _metrics_src.get('ic_valid'),
            'ric_test': _metrics_src.get('ric_test'),
            'ric_valid': _metrics_src.get('ric_valid')
        }
        _BEST_RECORD = {
            'gen': generation,
            'pool': best_this_gen['pool'],
            'metrics': _metrics_clean,
            'portfolio': {
                'exprs': best_this_gen['res'].get('exprs', []),
                'weights': best_this_gen['res'].get('weights', [])
            }
        }

    elapsed_gen = time.time() - _t_gen
    print(f"\n[GP] Gen={generation} | cache={len(cache)} | elapsed={elapsed_gen:.2f} s")
    # Neat form output
    header = f"{'Pool':>6} | {'IC(test)':>9} {'IC(valid)':>10} | {'RIC(test)':>9} {'RIC(valid)':>10}"
    print(header)
    print('-'*len(header))
    for item in res:
        cap = item['pool']
        r = item['res']
        print(f"{cap:6d} | {r['ic_test']:9.6f} {r['ic_valid']:10.6f} | {r['ric_test']:9.6f} {r['ric_valid']:10.6f}")
    # Print the current cache Top-3 candidate (if it exists)
    if cache:
        print("\n[GP] Top3 cache entries:")
        for k, v in list(Counter(cache).most_common(3)):
            # v may be a floating point score or count, the unified format is width 6, right aligned
            if isinstance(v, (int,)):
                print(f"   {v:6d} | {k}")
            elif isinstance(v, float):
                print(f"   {v:6.3f} | {k}")
            else:
                print(f"   {str(v):>6} | {k}")
    global save_dir
    dir_ = save_dir
    os.makedirs(dir_, exist_ok=True)
    # Record/append each generation gp_sum.csv (controlled by switch)
    save_gp_sum = SAVE_GP_SUM
    if save_gp_sum:
        global _SUMMARY_CSV_PATH
        if _SUMMARY_CSV_PATH is None:
            _SUMMARY_CSV_PATH = os.path.join(dir_, 'gp_sum.csv')
            if not os.path.exists(_SUMMARY_CSV_PATH):
                with open(_SUMMARY_CSV_PATH, 'w') as fcsv:
                    fcsv.write('generation,pool,ic_test,ic_valid,ric_test,ric_valid\n')
        with open(_SUMMARY_CSV_PATH, 'a') as fcsv:
            for item in res:
                r = item['res']
                fcsv.write(f"{generation},{item['pool']},{r.get('ic_test')},{r.get('ic_valid')},{r.get('ric_test')},{r.get('ric_valid')}\n")
    # Save each generation snapshot under snapshots/ (controlled by switch) - only save cache/res (consistent with the original version)
    if SAVE_SNAPSHOTS:
        snap_dir = os.path.join(dir_, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        snap_path = os.path.join(snap_dir, f'{generation}.json')
        with open(snap_path, 'w') as f:
            f.write(json.dumps(
                {
                    'cache': cache,
                    'res': [{ 'pool': it['pool'], 'res': {k: it['res'][k] for k in ('ic_test','ic_valid','ric_test','ric_valid')} } for it in res]
                },
                indent=2,
                ensure_ascii=False,
                separators=(',', ': ')
            ) + "\n")
        print(f"[SAVE] Snapshot: {snap_path}")
    # Total result file (controlled by switch): coverage and update for each generation, only the best best of all generations is saved
    if SAVE_GP_POOL and _BEST_RECORD is not None:
        final_path = os.path.join(dir_, 'gp_pool.json')
        with open(final_path, 'w') as f:
            f.write(json.dumps(
                _BEST_RECORD,
                indent=2,
                ensure_ascii=False,
                separators=(',', ': ')
            ) + "\n")
        print(f"[SAVE] Updated pool result: {final_path}")

    # Save a single expression result (if enabled, the default dynamic global Top-N, the structure benchmarks the existing gp_sgl.json)
    if SAVE_SINGLE_EXPRS:
        single_path = os.path.join(dir_, 'gp_sgl.json')
        # Merge history + current, remove duplication according to expression and retain maximum ic
        merged = {}
        # History
        try:
            if os.path.exists(single_path):
                with open(single_path, 'r') as rf:
                    history = json.load(rf)
                if isinstance(history, list):
                    for it in history:
                        e = it.get('expr')
                        v = it.get('ic')
                        if isinstance(e, str) and isinstance(v, (int, float)):
                            merged[e] = max(float(v), merged.get(e, float('-inf')))
        except Exception:
            pass
        # Current (Counter may be count or floating point, only floating point is accepted)
        for key, ic in Counter(cache).most_common(TOP_SINGLE_COUNT * 5):
            if isinstance(ic, (int,)):
                continue
            try:
                if SGL_METRIC == 'ic':
                    merged[key] = max(float(ic), merged.get(key, float('-inf')))
            except Exception:
                continue
        # Global sorting takes Top-N
        top_n = min(TOP_SINGLE_COUNT, len(merged))
        top_items = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out_list = [{ 'idx': i, 'expr': e, 'ic': v } for i, (e, v) in enumerate(top_items)]
        with open(single_path, 'w') as f:
            f.write(json.dumps(out_list, indent=2, ensure_ascii=False, separators=(',', ': ')) + "\n")
        print(f"[SAVE] Updated single expressions (global Top-{top_n}): {single_path}")
    print(f"\n[GP] ========== Generation {generation} End ========== \n")


def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


for seed in SEEDS:
    # Directory naming uses train/test year to avoid inconsistency with AFF/DSO style
        # Unified naming: hard-coded prefix gp_ + instruments + year range + seed + optional suffix
        year_start = str(TRAIN_START)[:4]
        year_end = str(TRAIN_END)[:4]
        base_name = f"gp_{INSTRUMENTS}_{year_start}-{year_end}_{seed}"
        save_name = f"{base_name}_{NAME_SUFFIX}" if NAME_SUFFIX else base_name
        save_dir = f'out_gp/{save_name}'
        print(f"[INFO] Start training: seed={seed}, save directory ={save_dir}")

        Metric = make_fitness(function=_metric, greater_is_better=True)
        funcs = [make_function(**func._asdict()) for func in gp_funcs]

        generation = 0
        cache = {}

        reseed_everything(seed)


        print(f"[DATA] Load data (precise date): instruments={INSTRUMENTS}, freq={FREQ}")
        # Directly use the precise date of global configuration (no more derivation)
        train_start_date = TRAIN_START
        train_end_date   = TRAIN_END
        valid_start_date = VALID_START
        valid_end_date   = VALID_END
        test_start_date  = TEST_START
        test_end_date    = TEST_END

        returned = get_data_by_year_with_dates(
            train_start = train_start_date, train_end = train_end_date,
            valid_start = valid_start_date, valid_end = valid_end_date,
            test_start = test_start_date,  test_end = test_end_date,
            instruments = INSTRUMENTS, target = target, freq = FREQ,
        )
        data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, name = returned
        print(f"[DATA] Data loading completed: {name}")

        pool = AlphaPool(capacity=10,
                        stock_data=data,
                        target=target,
                        ic_lower_bound=None)

        print("[DATA] Calculate target factor (train/valid/test) ...")
        target_factor = target.evaluate(data)
        target_factor_valid = target.evaluate(data_valid)
        target_factor_test = target.evaluate(data_test)
        print("[DATA] Target factor calculation completed")


        features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
        constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
        terminals = features + constants

        # Maintain the original string terminal input and cooperate with the string parsing in the customized Metric
        X_train = np.array([terminals])
        y_train = np.array([1])  # One dimension to eliminate sklearn warning

        print("[INFO] Initialization SymbolicRegressor ...")
        est_gp = SymbolicRegressor(population_size=DEFAULT_POPULATION_SIZE,
                                generations=DEFAULT_GENERATIONS,
                                init_depth=(2, 6),
                                tournament_size=DEFAULT_TOURNAMENT_SIZE,
                                stopping_criteria=1.,
                                p_crossover=0.3,
                                p_subtree_mutation=0.1,
                                p_hoist_mutation=0.01,
                                p_point_mutation=0.1,
                                p_point_replace=0.6,
                                max_samples=0.9,
                                verbose=2,
                                parsimony_coefficient=0.,
                                random_state=seed,
                                function_set=funcs,
                                metric=Metric,
                                const_range=None,
                                n_jobs=1)
        print(f"[INFO] Initialize the first generation population: population={est_gp.population_size}, generations={est_gp.generations}, n_jobs={est_gp.n_jobs}")
        print(f"[INFO] Start training...\n")
        # First generation generation progress bar

        init_total = est_gp.population_size
        init_pbar = tqdm(total=init_total, desc='[PROG] Init population', leave=False)
        def _init_hook(increment):
            init_pbar.update(increment)
        # Bind hook to the instance for capture by genetic._parallel_evolve
        est_gp._init_progress_hook = _init_hook
        train_start_time = datetime.now()
        est_gp.fit(X_train, y_train, callback=ev)
        init_pbar.close()
        elapsed = datetime.now() - train_start_time
        print(f"[GP] Training time: {str(elapsed)}")
        print("[GP] Training completed")

        # Display the best single expression found by the GP algorithm
        best_single_expr = str(est_gp._program)
        print(f"[GP] Best single expression: {best_single_expr}")
        print(f"[GP] Best factor combination: gen={_BEST_RECORD['gen']} | pool={_BEST_RECORD['pool'] if _BEST_RECORD else 'N/A'})")
