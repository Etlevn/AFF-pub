import gym
import gym.spaces
import numpy as np

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.rl.env.core import AlphaEnvCore

def _get_action_space_constants():
    """Dynamically calculate action space constants to ensure that advanced operators are included"""
    # Dynamically obtain the current operator list instead of using the cache during import
    from alphagen.config import OPERATORS as current_operators

    SIZE_NULL = 1
    SIZE_OP = len(current_operators)
    SIZE_FEATURE = len(FeatureType)
    SIZE_DELTA_TIME = len(DELTA_TIMES)
    SIZE_CONSTANT = len(CONSTANTS)
    SIZE_SEP = 1

    SIZE_ALL = SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP
    SIZE_ACTION = SIZE_ALL - SIZE_NULL

    OFFSET_OP = SIZE_NULL
    OFFSET_FEATURE = OFFSET_OP + SIZE_OP
    OFFSET_DELTA_TIME = OFFSET_FEATURE + SIZE_FEATURE
    OFFSET_CONSTANT = OFFSET_DELTA_TIME + SIZE_DELTA_TIME
    OFFSET_SEP = OFFSET_CONSTANT + SIZE_CONSTANT

    return {
        'SIZE_NULL': SIZE_NULL,
        'SIZE_OP': SIZE_OP,
        'SIZE_FEATURE': SIZE_FEATURE,
        'SIZE_DELTA_TIME': SIZE_DELTA_TIME,
        'SIZE_CONSTANT': SIZE_CONSTANT,
        'SIZE_SEP': SIZE_SEP,
        'SIZE_ALL': SIZE_ALL,
        'SIZE_ACTION': SIZE_ACTION,
        'OFFSET_OP': OFFSET_OP,
        'OFFSET_FEATURE': OFFSET_FEATURE,
        'OFFSET_DELTA_TIME': OFFSET_DELTA_TIME,
        'OFFSET_CONSTANT': OFFSET_CONSTANT,
        'OFFSET_SEP': OFFSET_SEP
    }

def update_action_space_constants():
    """Update action space constants (called after the advanced operator is loaded)"""
    global SIZE_NULL, SIZE_OP, SIZE_FEATURE, SIZE_DELTA_TIME, SIZE_CONSTANT, SIZE_SEP
    global SIZE_ALL, SIZE_ACTION, OFFSET_OP, OFFSET_FEATURE, OFFSET_DELTA_TIME, OFFSET_CONSTANT, OFFSET_SEP

    _constants = _get_action_space_constants()
    SIZE_NULL = _constants['SIZE_NULL']
    SIZE_OP = _constants['SIZE_OP']
    SIZE_FEATURE = _constants['SIZE_FEATURE']
    SIZE_DELTA_TIME = _constants['SIZE_DELTA_TIME']
    SIZE_CONSTANT = _constants['SIZE_CONSTANT']
    SIZE_SEP = _constants['SIZE_SEP']
    SIZE_ALL = _constants['SIZE_ALL']
    SIZE_ACTION = _constants['SIZE_ACTION']
    OFFSET_OP = _constants['OFFSET_OP']
    OFFSET_FEATURE = _constants['OFFSET_FEATURE']
    OFFSET_DELTA_TIME = _constants['OFFSET_DELTA_TIME']
    OFFSET_CONSTANT = _constants['OFFSET_CONSTANT']
    OFFSET_SEP = _constants['OFFSET_SEP']

# Initialize action space constants
def _init_action_space_constants():
    """Initialize action space constants"""
    global SIZE_NULL, SIZE_OP, SIZE_FEATURE, SIZE_DELTA_TIME, SIZE_CONSTANT, SIZE_SEP
    global SIZE_ALL, SIZE_ACTION, OFFSET_OP, OFFSET_FEATURE, OFFSET_DELTA_TIME, OFFSET_CONSTANT, OFFSET_SEP

    SIZE_NULL = 1
    SIZE_OP = len(OPERATORS)
    SIZE_FEATURE = len(FeatureType)
    SIZE_DELTA_TIME = len(DELTA_TIMES)
    SIZE_CONSTANT = len(CONSTANTS)
    SIZE_SEP = 1

    SIZE_ALL = SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP
    SIZE_ACTION = SIZE_ALL - SIZE_NULL

    OFFSET_OP = SIZE_NULL
    OFFSET_FEATURE = OFFSET_OP + SIZE_OP
    OFFSET_DELTA_TIME = OFFSET_FEATURE + SIZE_FEATURE
    OFFSET_CONSTANT = OFFSET_DELTA_TIME + SIZE_DELTA_TIME
    OFFSET_SEP = OFFSET_CONSTANT + SIZE_CONSTANT

# Initialize action space constants
_init_action_space_constants()


def action2token(action_raw: int) -> Token:
    # Dynamically obtain the current operator list and offset
    from alphagen.config import OPERATORS

    action = action_raw + 1
    if action < OFFSET_OP:
        raise ValueError
    elif action < OFFSET_FEATURE:
        op_idx = action - OFFSET_OP
        if op_idx < len(OPERATORS):
            return OperatorToken(OPERATORS[op_idx])
        else:
            raise IndexError(f"Operator index {op_idx} out of range (max: {len(OPERATORS)-1})")
    elif action < OFFSET_DELTA_TIME:
        return FeatureToken(FeatureType(action - OFFSET_FEATURE))
    elif action < OFFSET_CONSTANT:
        return DeltaTimeToken(DELTA_TIMES[action - OFFSET_DELTA_TIME])
    elif action < OFFSET_SEP:
        return ConstantToken(CONSTANTS[action - OFFSET_CONSTANT])
    elif action == OFFSET_SEP:
        return SequenceIndicatorToken(SequenceIndicatorType.SEP)
    else:
        assert False


class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(self, env: AlphaEnvCore):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(SIZE_ACTION)
        self.observation_space = gym.spaces.Box(low=0, high=SIZE_ALL - 1, shape=(MAX_EXPR_LENGTH, ), dtype=np.uint8)

    def reset(self, **kwargs) -> np.ndarray:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        # Return (obs, info) to be compatible with SB3 (gymnasium style)
        return self.state, {}

    def step(self, action: int):
        _, reward, done, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        terminated = bool(done)
        truncated = False
        return self.state, self.reward(reward), terminated, truncated, info

    def action(self, action: int) -> Token:
        return action2token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        # Dynamically obtain the current operator list
        from alphagen.config import OPERATORS

        res = np.zeros(SIZE_ACTION, dtype=bool)
        valid = self.env.valid_action_types()
        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            op_idx = i - OFFSET_OP
            if op_idx < len(OPERATORS) and valid['op'][OPERATORS[op_idx].category_type()]:
                res[i - 1] = True
        if valid['select'][1]:  # FEATURE
            for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
                res[i - 1] = True
        if valid['select'][2]:  # CONSTANT
            for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
                res[i - 1] = True
        if valid['select'][3]:  # DELTA_TIME
            for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
                res[i - 1] = True
        if valid['select'][4]:  # SEP
            res[OFFSET_SEP - 1] = True
        return res


def AlphaEnv(pool: AlphaPoolBase, **kwargs):
    return AlphaEnvWrapper(AlphaEnvCore(pool=pool, **kwargs))
