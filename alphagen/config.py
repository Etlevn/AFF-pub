from alphagen.data.expression import *
from typing import List, Type


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

# =============== Basic operator set definition ===============
# OPERATORS_STD: from operators.py
from alphagen_generic.operators import unary_ops, binary_ops, rolling_ops, rolling_binary_ops
OPERATORS_STD = unary_ops + binary_ops + rolling_ops + rolling_binary_ops

# OPERATORS_CSTM: from operators_cstm.py
try:
    from alphagen_generic.operators_cstm import cstm_unary_ops, cstm_rolling_ops, cstm_rolling_binary_ops
    OPERATORS_CSTM = cstm_unary_ops + cstm_rolling_ops + cstm_rolling_binary_ops
except Exception:
    OPERATORS_CSTM = []

# Optional extension: QLIB
try:
    from alphagen_generic.operators_qlib import qlib_unary_ops, qlib_binary_ops, qlib_rolling_ops, qlib_rolling_binary_ops
    OPERATORS_QLIB = qlib_unary_ops + qlib_binary_ops + qlib_rolling_ops + qlib_rolling_binary_ops
except Exception:
    OPERATORS_QLIB = []

# Optional extension: TALIB
try:
    from alphagen_generic.operators_talib import (
        talib_unary_ops,
        talib_binary_ops,
        talib_rolling_ops,
        talib_rolling_binary_ops,
    )
    OPERATORS_TALIB = talib_unary_ops + talib_binary_ops + talib_rolling_ops + talib_rolling_binary_ops
except Exception:
    OPERATORS_TALIB = []

# =============== Global constants ===============

# Global constant
CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]
REWARD_PER_STEP = 0.
DELTA_TIMES = [1, 5, 10, 20, 30] # 40 50

# Expose a global OPERATORS by default
OPERATORS = OPERATORS_STD

# =============== Tag combination (std, cstm, qlib, talib) ===============
def get_operators_by_tags(tags: str):
    """Freely combine operator libraries based on comma-separated tags.
    Available tags: std, cstm, qlib, talib; it is no longer mandatory to include std.
    Specify at least one valid tag; merge in declaration order and remove duplicates.
    """
    if not isinstance(tags, str):
        print("[ERROR] operators tags must be a comma-separated string")
        raise TypeError("operators tags must be string")
    parts = [t.strip().lower() for t in tags.split(',') if t.strip()]
    if not parts:
        print("[ERROR] operators tags cannot be empty")
        raise ValueError("operators tags cannot be empty")

    ops: List[Type[Expression]] = []
    seen = set()

    def _extend(candidates: List[Type[Expression]]):
        for cls in candidates:
            if cls not in seen:
                seen.add(cls)
                ops.append(cls)

    for p in parts:
        if p == 'std':
            _extend(OPERATORS_STD)
        elif p == 'cstm':
            _extend(OPERATORS_CSTM)
            if not OPERATORS_CSTM:
                print("[ERROR] CSTM extension is not available or not loaded")
                raise ImportError("OPERATORS_CSTM is empty")
        elif p == 'qlib':
            if not OPERATORS_QLIB:
                print("[ERROR] QLIB extension is not available or not loaded")
                raise ImportError("OPERATORS_QLIB is empty")
            _extend(OPERATORS_QLIB)
        elif p == 'talib':
            if not OPERATORS_TALIB:
                print("[ERROR] TALIB extension is not available or not loaded")
                raise ImportError("OPERATORS_TALIB is empty")
            _extend(OPERATORS_TALIB)
        else:
            print(f"[ERROR] Unrecognized operator tag: {p}")
            raise ValueError(f"Unsupported operators tag: {p}")
    return ops

def initialize_operators_by_tags(tags: str) -> None:
    """Initialize the global OPERATORS according to the label combination and refresh the action space. Report errors strictly."""
    global OPERATORS
    operators = get_operators_by_tags(tags)
    OPERATORS = operators
    from alphagen.rl.env import wrapper as _wrapper
    if hasattr(_wrapper, 'update_action_space_constants'):
        _wrapper.update_action_space_constants()
