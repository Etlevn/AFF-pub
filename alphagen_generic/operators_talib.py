from typing import List, Type

import torch
import numpy as np
import pandas as pd
from torch import Tensor

from alphagen.data.expression import RollingOperator, UnaryOperator, PairRollingOperator, BinaryOperator


# ===============
# Cycle Indicators (HT_*)
# ===============

class ts_ht_dcperiod(RollingOperator):
    """Hilbert Transform - Dominant Cycle Period
    Torch path: Approximate calculation using the closing price within the window, simple fallback to the placeholder implementation of NA-safe (returning the tensor of the window length).
    Pandas path: call TA-Lib HT_DCPERIOD.
    """
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate placeholder: return window length, shape (L, S)
        n = operand.shape[-1]
        return torch.full_like(operand[..., 0], fill_value=float(n))

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        return pd.Series(talib.HT_DCPERIOD(data.to_numpy()), index=data.index)


class ts_ht_dcphase(RollingOperator):
    """Hilbert Transform - Dominant Cycle Phase"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate occupancy: using linear phase proxy
        n = operand.shape[-1]
        phase = torch.linspace(0.0, 2 * np.pi, steps=n, dtype=operand.dtype, device=operand.device)
        phase = phase.unsqueeze(0).unsqueeze(0).expand_as(operand)
        return phase[..., -1]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        return pd.Series(talib.HT_DCPHASE(data.to_numpy()), index=data.index)


class ts_ht_phasor_inphase(RollingOperator):
    """Hilbert Transform - Phasor Components (inphase)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate placeholder: take the last value after averaging within the window
        centered = operand - operand.mean(dim=-1, keepdim=True)
        return centered[..., -1]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        inphase, quadrature = talib.HT_PHASOR(data.to_numpy())
        return pd.Series(inphase, index=data.index)


class ts_ht_phasor_quadrature(RollingOperator):
    """Hilbert Transform - Phasor Components (quadrature)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate placeholder: the difference between the window and one lag is approximated as an orthogonal component
        diff = operand[..., 1:] - operand[..., :-1]
        pad = torch.zeros_like(operand[..., 0:1])
        quad = torch.cat([pad, diff], dim=-1)
        return quad[..., -1]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        inphase, quadrature = talib.HT_PHASOR(data.to_numpy())
        return pd.Series(quadrature, index=data.index)


class ts_ht_sine(RollingOperator):
    """Hilbert Transform - SineWave (sine)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate placeholder: do sine transformation on the last value
        last = operand[..., -1]
        return last.sin()

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        sine, leadsine = talib.HT_SINE(data.to_numpy())
        return pd.Series(sine, index=data.index)


class ts_ht_leadsine(RollingOperator):
    """Hilbert Transform - SineWave (leadsine)"""
    def _apply(self, operand: Tensor) -> Tensor:
        last = operand[..., -1]
        return (last + 0.5).sin()

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        sine, leadsine = talib.HT_SINE(data.to_numpy())
        return pd.Series(leadsine, index=data.index)


class ts_ht_trendmode(RollingOperator):
    """Hilbert Transform - Trend vs Cycle Mode
    Return 1/0 (trend/oscillation)
    """
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate placeholder: judge by window standard deviation threshold
        std = operand.std(dim=-1)
        thresh = torch.nan_to_num(std.median(dim=0).values, nan=0.0)
        thresh = torch.where(thresh == 0, torch.tensor(1e-6, dtype=std.dtype, device=std.device), thresh)
        return (std > thresh).to(operand.dtype)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        arr = talib.HT_TRENDMODE(data.to_numpy())
        return pd.Series(arr, index=data.index)

# ===============
# Overlap Studies
# ===============

class ts_sma_ta(RollingOperator):
    """Simple Moving Average (SMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.SMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_ema_ta(RollingOperator):
    """Exponential Moving Average (EMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 2.0 / (1.0 + n)
        # Approximate weight method (same idea as ts_ema)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = (1 - alpha) ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.EMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_wma_ta(RollingOperator):
    """Weighted Moving Average (WMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(1, n + 1, dtype=operand.dtype, device=operand.device)
        weights = weights / weights.sum()
        return (weights * operand).sum(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.WMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_trima_ta(RollingOperator):
    """Triangular Moving Average (TRIMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: SMA of SMA (symmetrical triangle weight approximation)
        sma = operand.mean(dim=-1, keepdim=True)
        # Use final value approximation and then smooth
        return sma.squeeze(-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.TRIMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_t3_ta(RollingOperator):
    """T3 - Triple Exponential Moving Average"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: combination of multiple EMA (simplification of placeholder)
        n = operand.shape[-1]
        alpha = 2.0 / (1.0 + n)
        def ema_last(x: Tensor) -> Tensor:
            power = torch.arange(n, 0, -1, dtype=x.dtype, device=x.device)
            w = (1 - alpha) ** power
            w /= w.sum()
            return (w * x).sum(dim=-1)
        e1 = ema_last(operand)
        # One-step back approximation: without further time expansion, directly return the result of EMA
        return e1

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.T3(data.to_numpy(), timeperiod=window), index=data.index)


class ts_kama_ta(RollingOperator):
    """Kaufman Adaptive Moving Average (KAMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Placeholder: Return SMA approximate
        return operand.mean(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.KAMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_bbands_upper_ta(RollingOperator):
    """Bollinger Bands Upper"""
    def _apply(self, operand: Tensor) -> Tensor:
        sma = operand.mean(dim=-1)
        std = operand.std(dim=-1)
        return sma + 2 * std

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20, nbdev: float = 2.0) -> pd.Series:
        import talib
        upper, middle, lower = talib.BBANDS(data.to_numpy(), timeperiod=window, nbdevup=nbdev, nbdevdn=nbdev)
        return pd.Series(upper, index=data.index)


class ts_bbands_middle_ta(RollingOperator):
    """Bollinger Bands Middle"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20, nbdev: float = 2.0) -> pd.Series:
        import talib
        upper, middle, lower = talib.BBANDS(data.to_numpy(), timeperiod=window, nbdevup=nbdev, nbdevdn=nbdev)
        return pd.Series(middle, index=data.index)


class ts_bbands_lower_ta(RollingOperator):
    """Bollinger Bands Lower"""
    def _apply(self, operand: Tensor) -> Tensor:
        sma = operand.mean(dim=-1)
        std = operand.std(dim=-1)
        return sma - 2 * std

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20, nbdev: float = 2.0) -> pd.Series:
        import talib
        upper, middle, lower = talib.BBANDS(data.to_numpy(), timeperiod=window, nbdevup=nbdev, nbdevdn=nbdev)
        return pd.Series(lower, index=data.index)

class ts_dema_ta(RollingOperator):
    """Double Exponential Moving Average (DEMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: First EMA approximation
        n = operand.shape[-1]
        alpha = 2.0 / (1.0 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = (1 - alpha) ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.DEMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_tema_ta(RollingOperator):
    """Triple Exponential Moving Average (TEMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: First EMA approximation
        n = operand.shape[-1]
        alpha = 2.0 / (1.0 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = (1 - alpha) ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.TEMA(data.to_numpy(), timeperiod=window), index=data.index)


class ts_midpoint_ta(RollingOperator):
    """MIDPOINT over period: (max + min) / 2 on a single series"""
    def _apply(self, operand: Tensor) -> Tensor:
        mx = operand.max(dim=-1)[0]
        mn = operand.min(dim=-1)[0]
        return (mx + mn) / 2.0

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.MIDPOINT(data.to_numpy(), timeperiod=window), index=data.index)


class ts_ht_trendline_ta(RollingOperator):
    """HT_TRENDLINE - Instantaneous Trendline"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: window mean
        return operand.mean(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 32) -> pd.Series:
        import talib
        return pd.Series(talib.HT_TRENDLINE(data.to_numpy()), index=data.index)

# ===============
# Momentum Indicators
# ===============

class ts_rsi_ta(RollingOperator):
    """Relative Strength Index (RSI)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximate implementation consistent with custom version
        price_diff = operand[..., 1:] - operand[..., :-1]
        gains = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff))
        losses = torch.where(price_diff < 0, -price_diff, torch.zeros_like(price_diff))
        avg_gain = gains.mean(dim=-1)
        avg_loss = losses.mean(dim=-1)
        avg_loss = torch.where(avg_loss < 1e-6, torch.full_like(avg_loss, 1e-6), avg_loss)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.RSI(data.to_numpy(), timeperiod=window), index=data.index)


class ts_mom_ta(RollingOperator):
    """Momentum (MOM)"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[..., -1] - operand[..., 0]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 10) -> pd.Series:
        import talib
        return pd.Series(talib.MOM(data.to_numpy(), timeperiod=window), index=data.index)


class ts_roc_ta(RollingOperator):
    """Rate of Change (ROC)"""
    def _apply(self, operand: Tensor) -> Tensor:
        first = operand[..., 0]
        last = operand[..., -1]
        denom = torch.where(first.abs() < 1e-12, torch.full_like(first, 1e-12), first)
        return (last / denom - 1.0) * 100.0

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 10) -> pd.Series:
        import talib
        return pd.Series(talib.ROC(data.to_numpy(), timeperiod=window), index=data.index)


class ts_trix_ta(RollingOperator):
    """TRIX - 1-day Rate-Of-Change of a Triple EMA"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Placeholder: Approximately the ratio of the difference between the current and the mean
        mean = operand.mean(dim=-1)
        last = operand[..., -1]
        denom = torch.where(mean.abs() < 1e-12, torch.full_like(mean, 1e-12), mean)
        return (last / denom - 1.0) * 100.0

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 15) -> pd.Series:
        import talib
        return pd.Series(talib.TRIX(data.to_numpy(), timeperiod=window), index=data.index)

def _ema_last(operand: Tensor, span: int) -> Tensor:
    n = operand.shape[-1]
    span = max(1, min(span, n))
    alpha = 2.0 / (1.0 + span)
    power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
    weights = (1 - alpha) ** power
    weights /= weights.sum()
    return (weights * operand).sum(dim=-1)


class ts_apo_ta(RollingOperator):
    """Absolute Price Oscillator (APO) = EMA(fast) - EMA(slow)"""
    def _apply(self, operand: Tensor) -> Tensor:
        slow = max(2, int(self._delta_time))
        fast = max(1, slow // 2)
        ema_fast = _ema_last(operand, fast)
        ema_slow = _ema_last(operand, slow)
        return ema_fast - ema_slow

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 26) -> pd.Series:
        import talib
        fast = max(1, window // 2)
        slow = window
        return pd.Series(talib.APO(data.to_numpy(), fastperiod=fast, slowperiod=slow), index=data.index)


class ts_ppo_ta(RollingOperator):
    """Percentage Price Oscillator (PPO) = (EMA(fast)-EMA(slow)) / EMA(slow) * 100"""
    def _apply(self, operand: Tensor) -> Tensor:
        slow = max(2, int(self._delta_time))
        fast = max(1, slow // 2)
        ema_fast = _ema_last(operand, fast)
        ema_slow = _ema_last(operand, slow)
        denom = torch.where(ema_slow.abs() < 1e-12, torch.full_like(ema_slow, 1e-12), ema_slow)
        return (ema_fast - ema_slow) / denom * 100.0

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 26) -> pd.Series:
        import talib
        fast = max(1, window // 2)
        slow = window
        return pd.Series(talib.PPO(data.to_numpy(), fastperiod=fast, slowperiod=slow), index=data.index)

# ===============
# Statistic Functions
# ===============

class ts_std_ta(RollingOperator):
    """Standard Deviation (STDDEV)"""
    def _apply(self, operand: Tensor) -> Tensor:
        if operand.shape[-1] <= 1:
            return torch.zeros_like(operand[..., 0])
        return operand.std(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.STDDEV(data.to_numpy(), timeperiod=window), index=data.index)


class ts_var_ta(RollingOperator):
    """Variance (VAR)"""
    def _apply(self, operand: Tensor) -> Tensor:
        if operand.shape[-1] <= 1:
            return torch.zeros_like(operand[..., 0])
        return operand.var(dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.VAR(data.to_numpy(), timeperiod=window), index=data.index)


def _linreg_params(operand: Tensor):
    n = operand.shape[-1]
    x = torch.arange(n, dtype=operand.dtype, device=operand.device)
    x = x.view(1, 1, n)
    sum_x = x.sum(dim=-1, keepdim=True)
    sum_y = operand.sum(dim=-1, keepdim=True)
    sum_xy = (x * operand).sum(dim=-1, keepdim=True)
    sum_x2 = (x * x).sum(dim=-1, keepdim=True)
    denom = (n * sum_x2 - sum_x * sum_x)
    denom = denom + (denom == 0).to(operand.dtype) * 1e-8
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


class ts_linearreg_ta(RollingOperator):
    """Linear Regression (value at last x)"""
    def _apply(self, operand: Tensor) -> Tensor:
        slope, intercept = _linreg_params(operand)
        n = operand.shape[-1]
        yhat_last = slope[..., 0] * (n - 1) + intercept[..., 0]
        return yhat_last

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.LINEARREG(data.to_numpy(), timeperiod=window), index=data.index)


class ts_linearreg_slope_ta(RollingOperator):
    """Linear Regression Slope"""
    def _apply(self, operand: Tensor) -> Tensor:
        slope, _ = _linreg_params(operand)
        return slope[..., 0]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.LINEARREG_SLOPE(data.to_numpy(), timeperiod=window), index=data.index)


class ts_linearreg_intercept_ta(RollingOperator):
    """Linear Regression Intercept"""
    def _apply(self, operand: Tensor) -> Tensor:
        _, intercept = _linreg_params(operand)
        return intercept[..., 0]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.LINEARREG_INTERCEPT(data.to_numpy(), timeperiod=window), index=data.index)


class ts_linearreg_angle_ta(RollingOperator):
    """Linear Regression Angle (degrees)"""
    def _apply(self, operand: Tensor) -> Tensor:
        slope, _ = _linreg_params(operand)
        return torch.atan(slope[..., 0]) * (180.0 / np.pi)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.LINEARREG_ANGLE(data.to_numpy(), timeperiod=window), index=data.index)


class ts_tsf_ta(RollingOperator):
    """Time Series Forecast (TSF) - regression last value"""
    def _apply(self, operand: Tensor) -> Tensor:
        slope, intercept = _linreg_params(operand)
        n = operand.shape[-1]
        return slope[..., 0] * (n - 1) + intercept[..., 0]

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.TSF(data.to_numpy(), timeperiod=window), index=data.index)

# ===============
# Math Transform(Unary)
# ===============

class ta_acos(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.acos(operand.clamp(-1 + 1e-12, 1 - 1e-12))
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.ACOS(data.to_numpy()), index=data.index)


class ta_asin(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.asin(operand.clamp(-1 + 1e-12, 1 - 1e-12))
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.ASIN(data.to_numpy()), index=data.index)


class ta_atan(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.atan(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.ATAN(data.to_numpy()), index=data.index)


class ta_ceil(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.ceil(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.CEIL(data.to_numpy()), index=data.index)


class ta_cos(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.cos(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.COS(data.to_numpy()), index=data.index)


class ta_cosh(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.cosh(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.COSH(data.to_numpy()), index=data.index)


class ta_exp(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.exp(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.EXP(data.to_numpy()), index=data.index)


class ta_floor(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.floor(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.FLOOR(data.to_numpy()), index=data.index)


class ta_ln(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.log(operand.clamp_min(1e-12))
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.LN(data.to_numpy()), index=data.index)


class ta_log10(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.log10(operand.clamp_min(1e-12))
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.LOG10(data.to_numpy()), index=data.index)


class ta_sin(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.sin(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.SIN(data.to_numpy()), index=data.index)


class ta_sinh(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.sinh(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.SINH(data.to_numpy()), index=data.index)


class ta_sqrt(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.sqrt(operand.clamp_min(0))
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.SQRT(data.to_numpy()), index=data.index)


class ta_tan(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.tan(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.TAN(data.to_numpy()), index=data.index)


class ta_tanh(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.tanh(operand)
    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.TANH(data.to_numpy()), index=data.index)

# ===============
# Math Operators: MAX/MIN/SUM
# ===============

class ts_max_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0]
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.MAX(data.to_numpy(), timeperiod=window), index=data.index)


class ts_min_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.min(dim=-1)[0]
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.MIN(data.to_numpy(), timeperiod=window), index=data.index)


class ts_sum_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sum(dim=-1)
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.SUM(data.to_numpy(), timeperiod=window), index=data.index)


talib_math_rolling_ops: List[Type[RollingOperator]] = [
    ts_max_ta, ts_min_ta, ts_sum_ta,
]


class ts_maxindex_ta(RollingOperator):
    """Index of highest value over period"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.argmax(dim=-1).to(dtype=operand.dtype)
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.MAXINDEX(data.to_numpy(), timeperiod=window), index=data.index)


class ts_minindex_ta(RollingOperator):
    """Index of lowest value over period"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.argmin(dim=-1).to(dtype=operand.dtype)
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.MININDEX(data.to_numpy(), timeperiod=window), index=data.index)

class ts_minmax_min_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.min(dim=-1)[0]
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        minv, maxv = talib.MINMAX(data.to_numpy(), timeperiod=window)
        return pd.Series(minv, index=data.index)


class ts_minmax_max_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0]
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        minv, maxv = talib.MINMAX(data.to_numpy(), timeperiod=window)
        return pd.Series(maxv, index=data.index)


class ts_minmaxindex_min_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.argmin(dim=-1).to(dtype=operand.dtype)
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        minidx, maxidx = talib.MINMAXINDEX(data.to_numpy(), timeperiod=window)
        return pd.Series(minidx, index=data.index)


class ts_minmaxindex_max_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.argmax(dim=-1).to(dtype=operand.dtype)
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30) -> pd.Series:
        import talib
        minidx, maxidx = talib.MINMAXINDEX(data.to_numpy(), timeperiod=window)
        return pd.Series(maxidx, index=data.index)

# ===============
# Math Operators(Binary)ADD/SUB/MULT/DIV
# ===============

class ta_add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs + rhs
    @staticmethod
    def pandas_apply(a: pd.Series, b: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.ADD(a.to_numpy(), b.to_numpy()))


class ta_sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs - rhs
    @staticmethod
    def pandas_apply(a: pd.Series, b: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.SUB(a.to_numpy(), b.to_numpy()))


class ta_mult(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs * rhs
    @staticmethod
    def pandas_apply(a: pd.Series, b: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.MULT(a.to_numpy(), b.to_numpy()))


class ta_div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs / rhs
    @staticmethod
    def pandas_apply(a: pd.Series, b: pd.Series) -> pd.Series:
        import talib
        return pd.Series(talib.DIV(a.to_numpy(), b.to_numpy()))

# ===============
# Price Transform(Unary on OHLC-like close)
# ===============

class ta_avgprice(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand  # A single sequence cannot be averaged OHLC and remains as is
    @staticmethod
    def pandas_apply(data) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            open_ = data['open'].to_numpy()
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            close_ = data['close'].to_numpy()
            return pd.Series(talib.AVGPRICE(open_, high_, low_, close_), index=data.index)
        raise TypeError('AVGPRICE requires DataFrame containing column open/high/low/close')


class ta_medprice(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand
    @staticmethod
    def pandas_apply(data) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            return pd.Series(talib.MEDPRICE(high_, low_), index=data.index)
        raise TypeError('MEDPRICE requires DataFrame containing column high/low')


class ta_typprice(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand
    @staticmethod
    def pandas_apply(data) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            close_ = data['close'].to_numpy()
            return pd.Series(talib.TYPPRICE(high_, low_, close_), index=data.index)
        raise TypeError('TYPPRICE requires DataFrame containing column high/low/close')


class ta_wclprice(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand
    @staticmethod
    def pandas_apply(data) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            close_ = data['close'].to_numpy()
            return pd.Series(talib.WCLPRICE(high_, low_, close_), index=data.index)
        raise TypeError('WCLPRICE requires DataFrame containing column high/low/close')

# ===============
# Volatility Indicators (single input approximation)
# ===============

class ts_trange_ta(RollingOperator):
    """True Range (TRANGE) - the single sequence is approximately max-min"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0] - operand.min(dim=-1)[0]
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            close_ = data['close'].to_numpy()
            return pd.Series(talib.TRANGE(high_, low_, close_), index=data.index)
        raise TypeError('TRANGE requires DataFrame containing column high/low/close')


class ts_atr_ta(RollingOperator):
    """Average True Range (ATR) - the single sequence is approximately rolling std"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.std(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            close_ = data['close'].to_numpy()
            return pd.Series(talib.ATR(high_, low_, close_, timeperiod=window), index=data.index)
        raise TypeError('ATR requires DataFrame containing column high/low/close')


class ts_natr_ta(RollingOperator):
    """Normalized Average True Range (NATR)"""
    def _apply(self, operand: Tensor) -> Tensor:
        rng = operand.max(dim=-1)[0] - operand.min(dim=-1)[0]
        denom = torch.where(operand[..., -1].abs() < 1e-12, torch.full_like(operand[..., -1], 1e-12), operand[..., -1])
        return rng / denom * 100.0
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            high_ = data['high'].to_numpy()
            low_ = data['low'].to_numpy()
            close_ = data['close'].to_numpy()
            return pd.Series(talib.NATR(high_, low_, close_, timeperiod=window), index=data.index)
        raise TypeError('NATR requires DataFrame containing column high/low/close')

# ===============
# Volume Indicators (single input approximation, no real trading volume)
# ===============

class ts_obv_ta(RollingOperator):
    """On Balance Volume (OBV) - a single sequence is approximately a cumulative increase or decrease"""
    def _apply(self, operand: Tensor) -> Tensor:
        # When the window is too small (W<=1), the difference cannot be made and 0 is returned directly to avoid the slicing of size 0 from causing out-of-bounds
        if operand.shape[-1] <= 1:
            return torch.zeros_like(operand[..., 0])
        diff = operand[..., 1:] - operand[..., :-1]
        sign = torch.sign(diff)
        csum = torch.cumsum(sign, dim=-1)
        return csum[..., -1]
    @staticmethod
    def pandas_apply(data, window: int = 0) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            close_ = data['close'].to_numpy()
            vol_ = data['volume'].to_numpy()
            return pd.Series(talib.OBV(close_, vol_), index=data.index)
        raise TypeError('OBV requires DataFrame containing column close/volume')

# ===============
# Momentum (OHLC/HL/HLCCV multiple inputs)
# ===============

class ts_aroon_up_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            down, up = talib.AROON(data['high'].to_numpy(), data['low'].to_numpy(), timeperiod=window)
            return pd.Series(up, index=data.index)
        raise TypeError('AROON requires DataFrame containing column high/low')


class ts_aroon_down_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            down, up = talib.AROON(data['high'].to_numpy(), data['low'].to_numpy(), timeperiod=window)
            return pd.Series(down, index=data.index)
        raise TypeError('AROON requires DataFrame containing column high/low')


class ts_aroonosc_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.AROONOSC(data['high'].to_numpy(), data['low'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('AROONOSC requires DataFrame containing column high/low')


class ts_plus_di_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.PLUS_DI(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('PLUS_DI requires DataFrame containing column high/low/close')


class ts_minus_di_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.MINUS_DI(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('MINUS_DI requires DataFrame containing column high/low/close')


class ts_plus_dm_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.PLUS_DM(data['high'].to_numpy(), data['low'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('PLUS_DM requires DataFrame containing column high/low')


class ts_minus_dm_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.MINUS_DM(data['high'].to_numpy(), data['low'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('MINUS_DM requires DataFrame containing column high/low')


class ts_dx_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.DX(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('DX requires DataFrame containing column high/low/close')


class ts_adx_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.ADX(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('ADX requires DataFrame containing column high/low/close')


class ts_adxr_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.ADXR(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('ADXR requires DataFrame containing column high/low/close')


class ts_mfi_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.MFI(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), data['volume'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('MFI requires DataFrame containing column high/low/close/volume')


class ts_willr_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.WILLR(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('WILLR requires DataFrame containing column high/low/close')


class ts_cci_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.CCI(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('CCI requires DataFrame containing column high/low/close')


class ts_ultosc_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 7) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            # Use the default settings of 7, 14, 28
            return pd.Series(talib.ULTOSC(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), timeperiod1=7, timeperiod2=14, timeperiod3=28), index=data.index)
        raise TypeError('ULTOSC requires DataFrame containing column high/low/close')


class ts_stoch_k_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            k, d = talib.STOCH(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), fastk_period=window, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            return pd.Series(k, index=data.index)
        raise TypeError('STOCH requires DataFrame containing column high/low/close')


class ts_stoch_d_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            k, d = talib.STOCH(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), fastk_period=window, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            return pd.Series(d, index=data.index)
        raise TypeError('STOCH requires DataFrame containing column high/low/close')


class ts_stochf_k_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            k, d = talib.STOCHF(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), fastk_period=window, fastd_period=3, fastd_matype=0)
            return pd.Series(k, index=data.index)
        raise TypeError('STOCHF requires DataFrame containing column high/low/close')


class ts_stochf_d_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            k, d = talib.STOCHF(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), fastk_period=window, fastd_period=3, fastd_matype=0)
            return pd.Series(d, index=data.index)
        raise TypeError('STOCHF requires DataFrame containing column high/low/close')

# ===============
# Overlap (additional OHLC indicator)
# ===============

class ts_sar_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 0) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.SAR(data['high'].to_numpy(), data['low'].to_numpy()), index=data.index)
        raise TypeError('SAR requires DataFrame containing column high/low')


class ts_sarext_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, window: int = 0) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.SAREXT(data['high'].to_numpy(), data['low'].to_numpy()), index=data.index)
        raise TypeError('SAREXT requires DataFrame containing column high/low')

# ===============
# Adaptive/Variable Period Averages: MAMA/FAMA, MAVP
# ===============

class ts_mama_main_ta(RollingOperator):
    """MESA Adaptive Moving Average (MAMA) - main line"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, fastlimit: float = 0.5, slowlimit: float = 0.05) -> pd.Series:
        import talib
        if isinstance(data, pd.Series):
            mama, fama = talib.MAMA(data.to_numpy(), fastlimit=fastlimit, slowlimit=slowlimit)
            return pd.Series(mama, index=data.index)
        if isinstance(data, pd.DataFrame):
            close_ = data['close'].to_numpy()
            mama, fama = talib.MAMA(close_, fastlimit=fastlimit, slowlimit=slowlimit)
            return pd.Series(mama, index=data.index)
        raise TypeError('MAMA requires Series (close) or DataFrame containing close')


class ts_mama_fama_ta(RollingOperator):
    """MESA Adaptive Moving Average (MAMA) - following AMA (FAMA)"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, fastlimit: float = 0.5, slowlimit: float = 0.05) -> pd.Series:
        import talib
        if isinstance(data, pd.Series):
            mama, fama = talib.MAMA(data.to_numpy(), fastlimit=fastlimit, slowlimit=slowlimit)
            return pd.Series(fama, index=data.index)
        if isinstance(data, pd.DataFrame):
            close_ = data['close'].to_numpy()
            mama, fama = talib.MAMA(close_, fastlimit=fastlimit, slowlimit=slowlimit)
            return pd.Series(fama, index=data.index)
        raise TypeError('MAMA (FAMA) requires Series (close) or DataFrame containing close')


class ts_mavp_ta(RollingOperator):
    """Moving Average with Variable Period (MAVP)
    The window length vector of per-period needs to be passed in.
    pandas_apply accepts DataFrame containing columns close and period; minperiod/maxperiod will automatically take min/max of period.
    """
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, matype: int = 0) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns or 'period' not in data.columns:
                raise TypeError('MAVP requires DataFrame containing column close/period')
            close_ = data['close'].to_numpy()
            period_ = data['period'].to_numpy().astype(np.int32)
            minp = int(np.nanmin(period_)) if np.isfinite(period_).any() else 2
            maxp = int(np.nanmax(period_)) if np.isfinite(period_).any() else 30
            return pd.Series(talib.MAVP(close_, period_, minperiod=minp, maxperiod=maxp, matype=matype), index=data.index)
        raise TypeError('MAVP requires DataFrame containing column close/period')

# ===============
# Momentum extension: ROCP/ROCR/ROCR100/CMO/STOCHRSI (%K)
# ===============

class ts_rocp_ta(RollingOperator):
    """Rate of Change Percentage: (price - prev)/prev"""
    def _apply(self, operand: Tensor) -> Tensor:
        first = operand[..., 0]
        last = operand[..., -1]
        denom = torch.where(first.abs() < 1e-12, torch.full_like(first, 1e-12), first)
        return (last - first) / denom
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 10) -> pd.Series:
        import talib
        return pd.Series(talib.ROCP(data.to_numpy(), timeperiod=window), index=data.index)


class ts_rocr_ta(RollingOperator):
    """Rate of Change Ratio: price/prev"""
    def _apply(self, operand: Tensor) -> Tensor:
        first = operand[..., 0]
        last = operand[..., -1]
        denom = torch.where(first.abs() < 1e-12, torch.full_like(first, 1e-12), first)
        return last / denom
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 10) -> pd.Series:
        import talib
        return pd.Series(talib.ROCR(data.to_numpy(), timeperiod=window), index=data.index)


class ts_rocr100_ta(RollingOperator):
    """Rate of Change Ratio 100 scale: price/prev*100"""
    def _apply(self, operand: Tensor) -> Tensor:
        first = operand[..., 0]
        last = operand[..., -1]
        denom = torch.where(first.abs() < 1e-12, torch.full_like(first, 1e-12), first)
        return last / denom * 100.0
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 10) -> pd.Series:
        import talib
        return pd.Series(talib.ROCR100(data.to_numpy(), timeperiod=window), index=data.index)


class ts_cmo_ta(RollingOperator):
    """Chande Momentum Oscillator (CMO)"""
    def _apply(self, operand: Tensor) -> Tensor:
        diff = operand[..., 1:] - operand[..., :-1]
        up = torch.clamp(diff, min=0).sum(dim=-1)
        down = torch.clamp(-diff, min=0).sum(dim=-1)
        denom = torch.where((up + down) < 1e-12, torch.full_like(up, 1e-12), up + down)
        return (up - down) / denom * 100.0
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        return pd.Series(talib.CMO(data.to_numpy(), timeperiod=window), index=data.index)


class ts_stochrsi_k_ta(RollingOperator):
    """Stochastic RSI %K (only output K line, default fastk=fastd=3 from dt)"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: normalize the window to min-max
        mn = operand.min(dim=-1)[0]
        mx = operand.max(dim=-1)[0]
        rng = torch.where((mx - mn) < 1e-12, torch.full_like(mx, 1e-12), mx - mn)
        return (operand[..., -1] - mn) / rng * 100.0
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        import talib
        k, d = talib.STOCHRSI(data.to_numpy(), timeperiod=window, fastk_period=3, fastd_period=3, fastd_matype=0)
        return pd.Series(k, index=data.index)

# ===============
# Momentum: MACD / MACDFIX (return to MACD line)
# ===============

class ts_macd_ta(RollingOperator):
    """MACD: EMA(fast)-EMA(slow) with signal; Return to MACD main line"""
    def _apply(self, operand: Tensor) -> Tensor:
        slow = max(2, int(self._delta_time))
        fast = max(1, slow // 2)
        macd = _ema_last(operand, fast) - _ema_last(operand, slow)
        return macd

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 26) -> pd.Series:
        import talib
        fast = max(1, window // 2)
        slow = window
        macd, signal, hist = talib.MACD(data.to_numpy(), fastperiod=fast, slowperiod=slow, signalperiod=9)
        return pd.Series(macd, index=data.index)

class ts_macdfix_ta(RollingOperator):
    """MACDFIX 12/26 (TA-Lib fixed fast=12, slow=26, signal=9) return MACD main line"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Use dt to approximate slow, fast takes dt//2
        slow = max(2, int(self._delta_time))
        fast = max(1, slow // 2)
        return _ema_last(operand, fast) - _ema_last(operand, slow)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 26) -> pd.Series:
        import talib
        macd, signal, hist = talib.MACDFIX(data.to_numpy(), signalperiod=9)
        return pd.Series(macd, index=data.index)

# MACDEXT (controllable moving average type/period), return to MACD main line
class ts_macdext_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        slow = max(2, int(self._delta_time))
        fast = max(1, slow // 2)
        return _ema_last(operand, fast) - _ema_last(operand, slow)

    @staticmethod
    def pandas_apply(
        data: pd.Series,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
        fastmatype: int = 0,
        slowmatype: int = 0,
        signalmatype: int = 0,
    ) -> pd.Series:
        import talib
        macd, signal, hist = talib.MACDEXT(
            data.to_numpy(),
            fastperiod=fastperiod,
            fastmatype=fastmatype,
            slowperiod=slowperiod,
            slowmatype=slowmatype,
            signalperiod=signalperiod,
            signalmatype=signalmatype,
        )
        return pd.Series(macd, index=data.index)

# ===============
# Statistic Functions(Pair Rolling): BETA/CORREL
# ===============

class ts_beta_ta(PairRollingOperator):
    """BETA - Pearson beta between two series (rolling)"""
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_returns = torch.log(lhs[..., 1:] / lhs[..., :-1])
        rhs_returns = torch.log(rhs[..., 1:] / rhs[..., :-1])
        cl = lhs_returns - lhs_returns.mean(dim=-1, keepdim=True)
        cr = rhs_returns - rhs_returns.mean(dim=-1, keepdim=True)
        cov = (cl * cr).mean(dim=-1)
        var = (cr ** 2).mean(dim=-1)
        var = torch.where(var < 1e-12, torch.full_like(var, 1e-12), var)
        return cov / var

    @staticmethod
    def pandas_apply(a: pd.Series, b: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.BETA(a.to_numpy(), b.to_numpy(), timeperiod=window), index=a.index)


class ts_correl_ta(PairRollingOperator):
    """CORREL - Pearson correlation (rolling)"""
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        cl = lhs - lhs.mean(dim=-1, keepdim=True)
        cr = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (cl * cr).sum(dim=-1)
        nlvar = (cl ** 2).sum(dim=-1)
        nrvar = (cr ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul = torch.where(stdmul < 1e-12, torch.full_like(stdmul, 1.0), stdmul)
        return ncov / stdmul

    @staticmethod
    def pandas_apply(a: pd.Series, b: pd.Series, window: int = 30) -> pd.Series:
        import talib
        return pd.Series(talib.CORREL(a.to_numpy(), b.to_numpy(), timeperiod=window), index=a.index)

class ts_ma_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 30, matype: int = 0) -> pd.Series:
        import talib
        return pd.Series(talib.MA(data.to_numpy(), timeperiod=window, matype=matype), index=data.index)


class ts_midprice_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return (operand.max(dim=-1)[0] + operand.min(dim=-1)[0]) / 2
    @staticmethod
    def pandas_apply(data, window: int = 14) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.MIDPRICE(data['high'].to_numpy(), data['low'].to_numpy(), timeperiod=window), index=data.index)
        raise TypeError('MIDPRICE requires DataFrame containing column high/low')


class ts_ad_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.AD(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), data['volume'].to_numpy()), index=data.index)
        raise TypeError('AD requires DataFrame containing column high/low/close/volume')


class ts_adosc_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data, fastperiod: int = 3, slowperiod: int = 10) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.ADOSC(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), data['volume'].to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod), index=data.index)
        raise TypeError('ADOSC requires DataFrame containing column high/low/close/volume')


class ts_bop_ta(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    @staticmethod
    def pandas_apply(data) -> pd.Series:
        import talib
        if isinstance(data, pd.DataFrame):
            return pd.Series(talib.BOP(data['open'].to_numpy(), data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy()), index=data.index)
        raise TypeError('BOP requires DataFrame containing column open/high/low/close')


# =========================
# Export list (according to TA-Lib native grouping + unified list)
# =========================

from typing import Dict, Iterable
import inspect as _inspect


def _discover_classes() -> Dict[str, Iterable[type]]:
    unary: List[Type[UnaryOperator]] = []
    binary: List[Type[BinaryOperator]] = []
    rolling: List[Type[RollingOperator]] = []
    rolling_binary: List[Type[PairRollingOperator]] = []
    for _name, _obj in list(globals().items()):
        if not _inspect.isclass(_obj):
            continue
        if _obj in (UnaryOperator, BinaryOperator, RollingOperator, PairRollingOperator):
            continue
        try:
            if issubclass(_obj, PairRollingOperator):
                rolling_binary.append(_obj)  # type: ignore
            elif issubclass(_obj, RollingOperator):
                rolling.append(_obj)  # type: ignore
            elif issubclass(_obj, UnaryOperator):
                unary.append(_obj)  # type: ignore
            elif issubclass(_obj, BinaryOperator):
                binary.append(_obj)  # type: ignore
        except Exception:
            continue
    return {
        'unary': unary,
        'binary': binary,
        'rolling': rolling,
        'rolling_binary': rolling_binary,
    }


def _build_groups(rolling_all: Iterable[type], rolling_pair_all: Iterable[type]):
    try:
        import talib as _talib
    except Exception:
        return {}
    _fn_to_group: Dict[str, str] = {}
    for _group, _names in _talib.get_function_groups().items():
        for _n in _names:
            _fn_to_group[_n] = _group

    def _detect_func_name(cls: type) -> str:
        pa = getattr(cls, 'pandas_apply', None)
        if pa is None:
            return ''
        try:
            src = _inspect.getsource(pa)
        except Exception:
            return ''
        import re as _re
        m = _re.search(r"talib\.([A-Z0-9_]+)\(", src)
        return m.group(1) if m else ''

    groups: Dict[str, List[type]] = {}
    for cls in list(rolling_all) + list(rolling_pair_all):
        fn = _detect_func_name(cls)
        gp = _fn_to_group.get(fn, None)
        if gp is None:
            continue
        groups.setdefault(gp, []).append(cls)
    return groups


_discovered = _discover_classes()
talib_unary_ops: List[Type[UnaryOperator]] = list(_discovered['unary'])  # type: ignore
talib_binary_ops: List[Type[BinaryOperator]] = list(_discovered['binary'])  # type: ignore
talib_rolling_ops: List[Type[RollingOperator]] = list(_discovered['rolling'])  # type: ignore
talib_rolling_binary_ops: List[Type[PairRollingOperator]] = list(_discovered['rolling_binary'])  # type: ignore

talib_groups = _build_groups(talib_rolling_ops, talib_rolling_binary_ops)

__all__ = [
    'talib_unary_ops',
    'talib_binary_ops',
    'talib_rolling_ops',
    'talib_rolling_binary_ops',
    'talib_groups',
]
