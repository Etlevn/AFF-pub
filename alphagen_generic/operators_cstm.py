from typing import List, Type

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from alphagen.data.expression import UnaryOperator, RollingOperator, PairRollingOperator

# =========================
# One-yuan operator (Unary)
# =========================

class Sqrt(UnaryOperator):
    """Square root operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.sqrt(torch.abs(operand))

    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        """Pandas version implementation"""
        return np.sqrt(np.abs(data))

class Exp(UnaryOperator):
    """Exponential operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.exp(torch.clamp(operand, max=10))  # Prevent overflow

    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        """Pandas version implementation"""
        return np.exp(np.clip(data, a_max=10, a_min=None))

class Tanh(UnaryOperator):
    """Hyperbolic tangent operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.tanh(operand)

    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        """Pandas version implementation"""
        return np.tanh(data)

class Sigmoid(UnaryOperator):
    """Sigmoid operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        return torch.sigmoid(operand)

    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        """Pandas version implementation"""
        return 1 / (1 + np.exp(-np.clip(data, -10, 10)))

class Square(UnaryOperator):
    """Square operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand ** 2

    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        """Pandas version implementation"""
        return data ** 2

class Cube(UnaryOperator):
    """Cubic operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        return operand ** 3

    @staticmethod
    def pandas_apply(data: pd.Series) -> pd.Series:
        """Pandas version implementation"""
        return data ** 3

# =========================
# Time series rolling operator (Rolling)
# =========================

class ts_volatility(RollingOperator):
    """Time series volatility operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand is already rolling window data (L, S, W)
        # Calculate logarithmic rate of return
        log_returns = torch.log(operand[..., 1:] / operand[..., :-1])
        # Calculate standard deviation
        return torch.std(log_returns, dim=-1)

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        returns = np.log(data / data.shift(1))
        return returns.rolling(window=window).std()

class ts_sharpe_ratio(RollingOperator):
    """Time series Sharpe ratio operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand is already rolling window data (L, S, W)
        # Calculate logarithmic rate of return
        log_returns = torch.log(operand[..., 1:] / operand[..., :-1])
        # Calculate Sharpe ratio
        mean_return = torch.mean(log_returns, dim=-1)
        std_return = torch.std(log_returns, dim=-1)
        std_return = torch.where(std_return < 1e-6, 1e-6, std_return)
        return mean_return / std_return

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        returns = np.log(data / data.shift(1))
        mean_return = returns.rolling(window=window).mean()
        std_return = returns.rolling(window=window).std()
        return mean_return / std_return

class ts_momentum(RollingOperator):
    """Time series momentum operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand is already rolling window data (L, S, W)
        # Calculate simple moving average
        sma = torch.mean(operand, dim=-1)
        # Calculate momentum (deviation of current value relative to mean)
        current = operand[..., -1]  # Take the last time point
        return (current - sma) / sma

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        sma = data.rolling(window=window).mean()
        return (data - sma) / sma

class ts_bollinger_upper(RollingOperator):
    """Bollinger Band track operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand is already rolling window data (L, S, W)
        sma = torch.mean(operand, dim=-1)
        std = torch.std(operand, dim=-1)
        return sma + 2 * std

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        return sma + 2 * std

class ts_bollinger_lower(RollingOperator):
    """Bollinger Band Lower Track Operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        sma = torch.mean(operand, dim=-1)
        std = torch.std(operand, dim=-1)
        lower = sma - 2 * std
        return lower

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        return sma - 2 * std

class ts_bollinger_width(RollingOperator):
    """Bollinger band width operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        # Calculate the standard deviation of each window, then multiply by 4 to obtain the Bollinger Band width
        std = torch.std(operand, dim=-1)  # (L, S)
        width = 4 * std
        return width

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        std = data.rolling(window=window).std()
        return 4 * std

class ts_price_position(RollingOperator):
    """Price position operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        min_price = torch.min(operand, dim=-1)[0]
        max_price = torch.max(operand, dim=-1)[0]
        current_price = operand[..., -1]  # Latest price
        price_range = max_price - min_price
        price_range = torch.where(price_range < 1e-6, 1e-6, price_range)
        position = (current_price - min_price) / price_range
        return position

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        min_price = data.rolling(window=window).min()
        max_price = data.rolling(window=window).max()
        return (data - min_price) / (max_price - min_price)

class ts_rsi(RollingOperator):
    """Relative Strength Index (RSI) operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        # Calculate price changes
        price_diff = operand[..., 1:] - operand[..., :-1]

        # Calculate rise and fall
        gains = torch.where(price_diff > 0, price_diff, 0.0)
        losses = torch.where(price_diff < 0, -price_diff, 0.0)

        # Calculate average gains and losses
        avg_gain = torch.mean(gains, dim=-1)
        avg_loss = torch.mean(losses, dim=-1)
        avg_loss = torch.where(avg_loss < 1e-6, 1e-6, avg_loss)

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        """Pandas version implementation"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class ts_macd(RollingOperator):
    """MACD indicator operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        # Simplified version MACD, using the difference between fast and slow moving averages
        fast_window = max(1, self._delta_time // 2)
        slow_window = self._delta_time

        # Calculate fast line EMA (using the first half of the window)
        fast_ema = operand[..., -fast_window:].mean(dim=-1)
        # Calculate slow line EMA (use all windows)
        slow_ema = operand.mean(dim=-1)

        # MACD = Express - Slow line
        return fast_ema - slow_ema

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 26) -> pd.Series:
        """Pandas version implementation"""
        fast_window = max(1, window // 2)
        slow_window = window

        fast_ema = data.ewm(span=fast_window).mean()
        slow_ema = data.ewm(span=slow_window).mean()

        return fast_ema - slow_ema

class ts_stoch(RollingOperator):
    """Random oscillator (Stochastic) operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        lowest_low = torch.min(operand, dim=-1)[0]
        highest_high = torch.max(operand, dim=-1)[0]
        current_close = operand[..., -1]  # Latest price

        # Calculate %K
        price_range = highest_high - lowest_low
        price_range = torch.where(price_range < 1e-6, 1e-6, price_range)
        stoch_k = 100 * (current_close - lowest_low) / price_range

        return stoch_k

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        """Pandas version implementation"""
        lowest_low = data.rolling(window=window).min()
        highest_high = data.rolling(window=window).max()

        stoch_k = 100 * (data - lowest_low) / (highest_high - lowest_low)
        return stoch_k

class ts_williams_r(RollingOperator):
    """William index (%R) operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        lowest_low = torch.min(operand, dim=-1)[0]
        highest_high = torch.max(operand, dim=-1)[0]
        current_close = operand[..., -1]  # Latest price

        # Calculate Williams %R
        price_range = highest_high - lowest_low
        price_range = torch.where(price_range < 1e-6, 1e-6, price_range)
        williams_r = -100 * (highest_high - current_close) / price_range

        return williams_r

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        """Pandas version implementation """
        lowest_low = data.rolling(window=window).min()
        highest_high = data.rolling(window=window).max()

        williams_r = -100 * (highest_high - data) / (highest_high - lowest_low)
        return williams_r

class ts_cci(RollingOperator):
    """Commodity channel index (CCI) operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        # Calculate the moving average of typical prices
        sma = torch.mean(operand, dim=-1)
        current_price = operand[..., -1]  # Latest price

        # Calculate mean absolute deviation
        deviations = torch.abs(operand - sma.unsqueeze(-1))
        mean_deviation = torch.mean(deviations, dim=-1)
        mean_deviation = torch.where(mean_deviation < 1e-6, 1e-6, mean_deviation)

        # Calculate CCI
        cci = (current_price - sma) / (0.015 * mean_deviation)

        return cci

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        sma = data.rolling(window=window).mean()
        mean_deviation = data.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        cci = (data - sma) / (0.015 * mean_deviation)
        return cci

class ts_atr(RollingOperator):
    """Average true range (ATR) operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # operand shape: (L, S, W)
        # Simplified version ATR, using the standard deviation of the price range
        true_range = torch.std(operand, dim=-1)
        return true_range

    @staticmethod
    def pandas_apply(data: pd.Series, window: int = 14) -> pd.Series:
        """Pandas version implementation"""
        # Simplified version ATR
        return data.rolling(window=window).std()

# =========================
# Binary rolling operator (Pair Rolling)
# =========================

class ts_beta(PairRollingOperator):
    """Beta coefficient operator"""
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # lhs, rhs shape: (L, S, W)
        # Calculate logarithmic rate of return
        lhs_returns = torch.log(lhs[..., 1:] / lhs[..., :-1])
        rhs_returns = torch.log(rhs[..., 1:] / rhs[..., :-1])

        # Calculate covariance and variance
        lhs_mean = torch.mean(lhs_returns, dim=-1, keepdim=True)
        rhs_mean = torch.mean(rhs_returns, dim=-1, keepdim=True)

        covariance = torch.mean((lhs_returns - lhs_mean) * (rhs_returns - rhs_mean), dim=-1)
        variance = torch.mean((rhs_returns - rhs_mean) ** 2, dim=-1)
        variance = torch.where(variance < 1e-6, 1e-6, variance)

        beta = covariance / variance
        return beta

    @staticmethod
    def pandas_apply(stock_data: pd.Series, market_data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        stock_returns = np.log(stock_data / stock_data.shift(1))
        market_returns = np.log(market_data / market_data.shift(1))

        covariance = stock_returns.rolling(window=window).cov(market_returns)
        market_variance = market_returns.rolling(window=window).var()

        return covariance / market_variance

class ts_alpha(PairRollingOperator):
    """Alpha coefficient operator"""
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # lhs, rhs shape: (L, S, W)
        # Calculate logarithmic rate of return
        lhs_returns = torch.log(lhs[..., 1:] / lhs[..., :-1])
        rhs_returns = torch.log(rhs[..., 1:] / rhs[..., :-1])

        lhs_mean = torch.mean(lhs_returns, dim=-1)
        rhs_mean = torch.mean(rhs_returns, dim=-1)

        alpha = lhs_mean - rhs_mean
        return alpha

    @staticmethod
    def pandas_apply(stock_data: pd.Series, market_data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        stock_returns = np.log(stock_data / stock_data.shift(1))
        market_returns = np.log(market_data / market_data.shift(1))

        stock_mean = stock_returns.rolling(window=window).mean()
        market_mean = market_returns.rolling(window=window).mean()

        return stock_mean - market_mean

class ts_treynor_ratio(PairRollingOperator):
    """Treynor ratio operator"""
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # lhs, rhs shape: (L, S, W)
        # This is simplified to the Sharpe ratio of a single asset
        # In practical applications, it can be extended to the real Treynor ratio
        log_returns = torch.log(lhs[..., 1:] / lhs[..., :-1])

        mean_return = torch.mean(log_returns, dim=-1)
        std_return = torch.std(log_returns, dim=-1)
        std_return = torch.where(std_return < 1e-6, 1e-6, std_return)
        treynor = mean_return / std_return

        return treynor

    @staticmethod
    def pandas_apply(stock_data: pd.Series, market_data: pd.Series, window: int = 20) -> pd.Series:
        """Pandas version implementation"""
        returns = np.log(stock_data / stock_data.shift(1))
        mean_return = returns.rolling(window=window).mean()
        std_return = returns.rolling(window=window).std()
        return mean_return / std_return

# =========================
# Export list (for configuration selection splicing)
# =========================

# Uniary operator list
cstm_unary_ops = [Sqrt, Exp, Tanh, Sigmoid, Square, Cube]

# Rolling operator list
cstm_rolling_ops = [
    ts_volatility, ts_sharpe_ratio, ts_momentum,
    ts_bollinger_upper, ts_bollinger_lower, ts_bollinger_width,
    ts_price_position, ts_rsi, ts_macd, ts_stoch, ts_williams_r, ts_cci, ts_atr
]

# Binary rolling operator list
cstm_rolling_binary_ops = [ts_beta, ts_alpha, ts_treynor_ratio]
