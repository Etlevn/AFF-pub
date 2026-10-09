from typing import List, Type

import torch
from torch import Tensor

from alphagen.data.expression import (
    Expression,
    UnaryOperator,
    BinaryOperator,
    RollingOperator,
)


# =========================
# Logic and comparison operator (Binary)
# =========================

class Eq(BinaryOperator):
    qlib_alias = 'Eq'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs == rhs).to(dtype=lhs.dtype)


class Ne(BinaryOperator):
    qlib_alias = 'Ne'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs != rhs).to(dtype=lhs.dtype)


class Gt(BinaryOperator):
    qlib_alias = 'Gt'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs > rhs).to(dtype=lhs.dtype)


class Ge(BinaryOperator):
    qlib_alias = 'Ge'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs >= rhs).to(dtype=lhs.dtype)


class Lt(BinaryOperator):
    qlib_alias = 'Lt'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs < rhs).to(dtype=lhs.dtype)


class Le(BinaryOperator):
    qlib_alias = 'Le'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs <= rhs).to(dtype=lhs.dtype)


class And(BinaryOperator):
    qlib_alias = 'And'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        l = (lhs != 0)
        r = (rhs != 0)
        return (l & r).to(dtype=lhs.dtype)


class Or(BinaryOperator):
    qlib_alias = 'Or'
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        l = (lhs != 0)
        r = (rhs != 0)
        return (l | r).to(dtype=lhs.dtype)


# =========================
# Conditional control and mask (Unary/Binary)
# =========================

class Not(UnaryOperator):
    qlib_alias = 'Not'
    def _apply(self, operand: Tensor) -> Tensor:
        return (~(operand != 0)).to(dtype=operand.dtype)


class If(BinaryOperator):
    qlib_alias = 'If'
    """
    Condition selection: the binary form of If (condition, x) is not enough;
    The semantics of BinaryOperator are used here: lhs is used as condition, and rhs is a placeholder (not used).
    It is recommended for actual use that Helper is packaged as If3 (condition, a, b).
    For the sake of compatibility, only the minimum implementation is provided here: If(cond, x) -> cond!=0 ? x : 0
    """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        cond = (lhs != 0)
        return torch.where(cond, rhs, torch.zeros_like(rhs))


class Mask(BinaryOperator):
    qlib_alias = 'Mask'
    """Mask operator - used for conditional filtering"""
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # lhs: Condition (0/1 or True/False)
        # rhs: data
        # Return: Return data when the condition is true, otherwise return 0
        condition = (lhs != 0)
        return torch.where(condition, rhs, torch.zeros_like(rhs))


# =========================
# Time series rolling operator (Rolling)
# =========================

class ts_count(RollingOperator):
    qlib_alias = 'Count'
    def _apply(self, operand: Tensor) -> Tensor:
        # Count the number of non-NaN
        return (~operand.isnan()).sum(dim=-1).to(dtype=operand.dtype)


class ts_idxmax(RollingOperator):
    qlib_alias = 'IdxMax'
    def _apply(self, operand: Tensor) -> Tensor:
        # Return the index of argmax in the window (0..W-1)
        return operand.argmax(dim=-1).to(dtype=operand.dtype)


class ts_idxmin(RollingOperator):
    qlib_alias = 'IdxMin'
    def _apply(self, operand: Tensor) -> Tensor:
        # Return the index of argmin in the window (0..W-1)
        return operand.argmin(dim=-1).to(dtype=operand.dtype)


class ts_quantile25(RollingOperator):
    # Comparison purpose: Quantile mapped to qlib (q=0.25)
    qlib_alias = 'Quantile'
    """25% quantile operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Calculate 25% quantile
        return torch.quantile(operand, 0.25, dim=-1)


class ts_quantile75(RollingOperator):
    # Comparison purpose: Quantile mapped to qlib (q=0.75)
    qlib_alias = 'Quantile'
    """75% quantile operator"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Calculate 75% quantile
        return torch.quantile(operand, 0.75, dim=-1)


# =========================
# Regression statistics (rolling window)
# =========================

class Resi(RollingOperator):
    qlib_alias = 'Resi'
    """Residual operator - used for regression analysis"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Calculate linear regression residuals
        # operand shape: (L, S, W)
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, W)

        # Calculate linear regression coefficients
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)

        # Slope: (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        # Intercept: (sum_y - slope*sum_x) / n
        intercept = (sum_y - slope * sum_x) / n

        # Calculate residual: y - (slope*x + intercept)
        predicted = slope * x + intercept
        residual = operand - predicted

        # Returns the standard deviation of the residuals
        return residual.std(dim=-1)


class Rsquare(RollingOperator):
    qlib_alias = 'Rsquare'
    """R square operator - used for goodness of fit"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Calculate the R square of linear regression
        # operand shape: (L, S, W)
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, W)

        # Calculate linear regression coefficients
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        sum_y2 = (operand * operand).sum(dim=-1, keepdim=True)

        # Slope: (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        # Intercept: (sum_y - slope*sum_x) / n
        intercept = (sum_y - slope * sum_x) / n

        # Calculate the square of R: 1 - SS_res/SS_tot
        predicted = slope * x + intercept
        ss_res = ((operand - predicted) ** 2).sum(dim=-1)
        ss_tot = ((operand - operand.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)

        r_square = 1 - ss_res / (ss_tot + 1e-8)
        return r_square


class Slope(RollingOperator):
    qlib_alias = 'Slope'
    """Slope operator - used for trend analysis"""
    def _apply(self, operand: Tensor) -> Tensor:
        # Calculate linear regression slope
        # operand shape: (L, S, W)
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, W)

        # Calculate linear regression coefficients
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)

        # Slope: (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        return slope.squeeze(-1)


# =========================
# Return related alias (rolling/expanding)
# =========================

class rolling_resi(RollingOperator):
    qlib_alias = 'rolling_resi'
    def _apply(self, operand: Tensor) -> Tensor:
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        intercept = (sum_y - slope * sum_x) / n
        predicted = slope * x + intercept
        residual = operand - predicted
        return residual.std(dim=-1)


class rolling_rsquare(RollingOperator):
    qlib_alias = 'rolling_rsquare'
    def _apply(self, operand: Tensor) -> Tensor:
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        intercept = (sum_y - slope * sum_x) / n
        predicted = slope * x + intercept
        ss_res = ((operand - predicted) ** 2).sum(dim=-1)
        ss_tot = ((operand - operand.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
        return 1 - ss_res / (ss_tot + 1e-8)


class rolling_slope(RollingOperator):
    qlib_alias = 'rolling_slope'
    def _apply(self, operand: Tensor) -> Tensor:
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        return slope.squeeze(-1)


class expanding_resi(RollingOperator):
    qlib_alias = 'expanding_resi'
    def _apply(self, operand: Tensor) -> Tensor:
        # Approximation: According to the semantics of ”accumulation from the starting point to the current” within the provided window, return the final cumulative value (equivalent to full window regression)
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        intercept = (sum_y - slope * sum_x) / n
        predicted = slope * x + intercept
        residual = operand - predicted
        return residual.std(dim=-1)


class expanding_rsquare(RollingOperator):
    qlib_alias = 'expanding_rsquare'
    def _apply(self, operand: Tensor) -> Tensor:
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        intercept = (sum_y - slope * sum_x) / n
        predicted = slope * x + intercept
        ss_res = ((operand - predicted) ** 2).sum(dim=-1)
        ss_tot = ((operand - operand.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
        return 1 - ss_res / (ss_tot + 1e-8)


class expanding_slope(RollingOperator):
    qlib_alias = 'expanding_slope'
    def _apply(self, operand: Tensor) -> Tensor:
        x = torch.arange(operand.shape[-1], dtype=operand.dtype, device=operand.device)
        x = x.unsqueeze(0).unsqueeze(0)
        n = operand.shape[-1]
        sum_x = x.sum(dim=-1, keepdim=True)
        sum_y = operand.sum(dim=-1, keepdim=True)
        sum_xy = (x * operand).sum(dim=-1, keepdim=True)
        sum_x2 = (x * x).sum(dim=-1, keepdim=True)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        return slope.squeeze(-1)

# =========================
# Export list (for configuration selection splicing)
# =========================

qlib_unary_ops: List[Type[Expression]] = [
    Not,
]

qlib_binary_ops: List[Type[Expression]] = [
    Eq, Ne, Gt, Ge, Lt, Le,
    And, Or,
    If, Mask,
]

qlib_rolling_ops: List[Type[Expression]] = [
    ts_count, ts_idxmax, ts_idxmin,
    ts_quantile25, ts_quantile75,
    Resi, Rsquare, Slope,
    rolling_resi, rolling_rsquare, rolling_slope,
    expanding_resi, expanding_rsquare, expanding_slope,
]

qlib_rolling_binary_ops: List[Type[Expression]] = []
