import os
import sys
import json
import argparse
import importlib
import pandas as pd
import warnings

# ============================== [CONFIG] ==============================

RUN_DIR = 'test_x'
# Operating mode:
# - ’dft’: Read select/factor_select.json and output to out/<RUN_DIR>_0
# - 'opt': Read select/factor_select.json and output to out/<RUN_DIR>_
# - ’llm’: Read llm/llm_opt.json and output to out/<RUN_DIR>_x
MODE = 'opt'

# ======================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gan.utils.builder import Builders, save_pickle  # Use the existing Builders structure

# Dynamically import all available operators into the eval environment

def load_operator_env():
    env = {}
    # Basic expressions and types
    from alphagen.data.expression import (
        Expression, UnaryOperator, BinaryOperator, RollingOperator, PairRollingOperator,
        Feature, FeatureType, Constant, DeltaTime,
    )
    env.update({
        'Expression': Expression,
        'UnaryOperator': UnaryOperator,
        'BinaryOperator': BinaryOperator,
        'RollingOperator': RollingOperator,
        'PairRollingOperator': PairRollingOperator,
        'Feature': Feature,
        'FeatureType': FeatureType,
        'Constant': Constant,
        'DeltaTime': DeltaTime,
    })

    # Load general operators
    modules = [
        'alphagen_generic.operators',
        'alphagen_generic.operators_qlib',
        'alphagen_generic.operators_talib',
        'alphagen_generic.operators_cstm',
    ]
    for m in modules:
        try:
            mod = importlib.import_module(m)
            for k, v in mod.__dict__.items():
                if not k.startswith('_'):
                    env[k] = v
        except Exception:
            pass

    # Commonly used built-in functions (if necessary)
    env.update({'min': min, 'max': max, 'abs': abs})
    return env


def preprocess_dollar_features(expr: str) -> str:
    # Map $close/$open/$high/$low/$volume/$vwap to Feature (FeatureType.X)
    mapping = {
        '$close': 'Feature(FeatureType.CLOSE)',
        '$open': 'Feature(FeatureType.OPEN)',
        '$high': 'Feature(FeatureType.HIGH)',
        '$low': 'Feature(FeatureType.LOW)',
        '$volume': 'Feature(FeatureType.VOLUME)',
        '$vwap': 'Feature(FeatureType.VWAP)',
    }
    out = expr
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def parse_expr(expr_str: str, env: dict):
    code = preprocess_dollar_features(expr_str)
    # Directly eval constructs the expression tree (all operators/operands are in env)
    return eval(code, env, {})


def _fix_parentheses_minimal(s: str) -> tuple[str, bool]:
    """Minimum bracket fix (aligned with llm_builder):
    1) Remove redundant leading right brackets (scan from left to right, skip ’)’ if there is no matching ’(’)
    2) If there are more ’(’ than ’)’ in the end, fill in the missing ’)’ at the end
    Return (string after repair, whether repair occurred)
    """
    changed = False
    buf = []
    balance = 0
    for ch in s:
        if ch == '(':
            balance += 1
            buf.append(ch)
        elif ch == ')':
            if balance > 0:
                balance -= 1
                buf.append(ch)
            else:
                # Skip extra closing brackets
                changed = True
                continue
        else:
            buf.append(ch)
    fixed = ''.join(buf)
    if balance > 0:
        fixed = fixed + (')' * balance)
        changed = True
    return fixed, changed


def build_builders_from_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Compatible with two structures: top-level array or {"factors": [...]} object
    if isinstance(data, list):
        factors = data
    else:
        factors = data.get('factors', [])

    # Keep only the items of enabled and sort them in ascending order by idx
    enabled_items = [x for x in factors if x.get('enabled', False)]
    enabled_items.sort(key=lambda x: x.get('idx', 0))

    env = load_operator_env()

    # Structure Builders
    n = len(enabled_items)
    blds = Builders(batch_size=n, max_len=1, n_actions=1)

    exprs = []
    exprs_str = []
    csv_rows = []

    tried = 0
    succeeded = 0
    failed = 0

    for item in enabled_items:
        idx = item.get('idx')
        expr_raw = item.get('expr', '').strip()
        if not expr_raw:
            continue
        tried += 1
        expr_try = expr_raw
        # Minimum bracket fix (consistent with llm_builder)
        expr_fixed, changed = _fix_parentheses_minimal(expr_try)
        if changed:
            print(f"[BUILDER] Expression brackets have been automatically repaired idx={idx}")
        try:
            expr_obj = parse_expr(expr_fixed, env)
        except Exception as e:
            failed += 1
            msg = (
                "\n[BUILDER] Skip the invalid expression \n"
                f"  - idx  : {idx}\n"
                f"  - expr : {expr_raw[:160]}{'...' if len(expr_raw) > 160 else ''}\n"
                f"  - error: {type(e).__name__}: {e}\n"
            )
            warnings.warn(msg)
            continue
        succeeded += 1
        expr_text = str(expr_obj)
        exprs.append(expr_obj)
        exprs_str.append(expr_text)
        # Output only the columns required for factor_process: ’Unnamed: 0’, ’exprs’, ’scores’
        csv_rows.append({
            'Unnamed: 0': idx,     # factor_process will be renamed to ’index’
            'exprs': expr_text,
            'scores': 0.0
        })

    blds.exprs = exprs
    blds.exprs_str = exprs_str
    stats = {'tried': tried, 'succeeded': succeeded, 'failed': failed}
    return blds, pd.DataFrame(csv_rows), stats


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    run_dir = (RUN_DIR or '').strip()
    mode = (MODE or 'dft').strip().lower()

    # Select the input JSON path
    if mode == 'llm':
        # LLM mode: output out/<RUN_DIR>_x
        json_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'analysis', run_dir, 'llm', 'llm_opt.json'))
        out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'out', f"{run_dir}_x"))
    elif mode == 'opt':
        # OPT mode: Output out/<RUN_DIR>_ (trailing underscore)
        json_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'analysis', run_dir, 'select', 'factor_select.json'))
        out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'out', f"{run_dir}_"))
    else:
        # DFT mode: output out/<RUN_DIR>_0
        json_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'analysis', run_dir, 'select', 'factor_select.json'))
        out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'out', f"{run_dir}_0"))

    if not os.path.exists(json_path):
        raise FileNotFoundError(f'JSON does not exist: {json_path}')

    blds, df, stats = build_builders_from_json(json_path)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'csv_zoo_final.csv')
    pkl_path = os.path.join(out_dir, 'z_bld_zoo_final.pkl')

    # Write CSV (use the string of the expression object to ensure consistency with PKL)
    # Keep writing in the original order to avoid redundant column conflicts
    df.to_csv(csv_path, index=False)

    # Write PKL (Save Builders)
    save_pickle(blds, pkl_path)

    print(f"[BUILDER] MODE={mode}  RUN_DIR={run_dir}")
    print(f"[BUILDER] Tried {stats['tried']} expressions, succeeded {stats['succeeded']}, failed {stats['failed']}")
    print(f'[BUILDER] CSV has been written: {csv_path}')
    print(f'[BUILDER] PKL has been written: {pkl_path}')


if __name__ == '__main__':
    main()
