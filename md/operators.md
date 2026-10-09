# Operator Reference

## Expression rules

- Use only the operator names listed in this reference and enabled in the runtime.
  Do not invent functions or symbols.
- Match each function's documented argument count.
- Close every parenthesis; do not add unmatched closing parentheses.
- Use the input feature names: `$open`, `$high`, `$low`, `$close`, `$volume`, and
  `$vwap`. Benchmark examples using `$index` require that feature to be available.
- `N` is a positive integer window length; common examples use 5, 10, 20, 30, or 60.
- Apply robust transforms such as `S_log1p`, `Log`, or `CSRank` before amplifying
  values with `Pow`, `Square`, or `Cube`.

| Operator family | Signature |
| --- | --- |
| Unary | `Op(x)` |
| Binary | `Op(x,y)` |
| Rolling, one series | `ts_op(x,N)` |
| Rolling, two series | `ts_op(x,y,N)` |

Standard pairwise rolling operators are `ts_cov` and `ts_corr`. The custom
pairwise operators listed below also accept two series and a window. All other
rolling operators listed here accept one series and a window.

## Standard unary operators

| Operator | Meaning | Example |
| --- | --- | --- |
| `Inv(x)` | Reciprocal with safeguards against division by zero | `Inv(Log($volume))` |
| `S_log1p(x)` | Robust log1p transform to moderate extreme values | `S_log1p($close-$open)` |
| `Abs(x)` | Absolute value | `Abs($close-$open)` |
| `Sign(x)` | Sign: positive = 1, zero = 0, negative = -1 | `Sign(ts_delta($close,1))` |
| `Log(x)` | Natural logarithm with handling for nonpositive values | `Log($volume)` |
| `CSRank(x)` | Cross-sectional rank at each time point | `CSRank(Abs($close-$open))` |

## Standard binary operators

| Operator | Meaning | Example |
| --- | --- | --- |
| `Add(x,y)` | Addition | `Add($close,$open)` |
| `Sub(x,y)` | Subtraction | `Sub($close,$open)` |
| `Mul(x,y)` | Multiplication | `Mul(CSRank($close),CSRank($volume))` |
| `Div(x,y)` | Division with numerical safeguards | `Div(($close-$low),$low)` |
| `Pow(x,y)` | Exponentiation with internal clipping | `Pow($close/$low,2)` |
| `Greater(x,y)` | Comparison of x and y | `Greater($close,$open)` |
| `Less(x,y)` | Comparison of x and y | `Less($close,$open)` |

## Standard rolling operators

These operators take one series `x` and one window `N`.

| Operator | Meaning |
| --- | --- |
| `Ref(x,N)` | Value N periods ago |
| `ts_mean(x,N)` | Rolling mean |
| `ts_sum(x,N)` | Rolling sum |
| `ts_std(x,N)`, `ts_var(x,N)` | Rolling standard deviation and variance |
| `ts_max(x,N)`, `ts_min(x,N)` | Rolling maximum and minimum |
| `ts_med(x,N)`, `ts_mad(x,N)` | Rolling median and mean absolute deviation |
| `ts_div(x,N)` | Window-based normalization or relative value, depending on the implementation |
| `ts_pctchange(x,N)` | Percentage change: x_t / x_(t-N) - 1 |
| `ts_delta(x,N)` | Difference: x_t - x_(t-N) |
| `ts_wma(x,N)`, `ts_ema(x,N)` | Weighted and exponential moving averages |
| `ts_skew(x,N)`, `ts_kurt(x,N)` | Rolling skewness and kurtosis |
| `ts_rank(x,N)` | Rank of the current value within the time window |
| `ts_ir(x,N)` | Information-ratio-style mean divided by standard deviation |
| `ts_min_max_diff(x,N)` | Rolling range |
| `ts_max_diff(x,N)`, `ts_min_diff(x,N)` | Differences from rolling extrema |

Examples: `Ref($close,5)`, `ts_mean($close,20)`, `ts_div($close,20)`.

## Standard pairwise rolling operators

| Operator | Meaning | Example |
| --- | --- | --- |
| `ts_cov(x,y,N)` | Rolling covariance | `ts_cov($close,$volume,60)` |
| `ts_corr(x,y,N)` | Rolling correlation | `ts_corr(CSRank($close),CSRank($volume),30)` |

## Custom unary operators

| Operator | Meaning | Example |
| --- | --- | --- |
| `Sqrt(x)` | Square root, with handling for negative inputs | `Sqrt(Abs($close-$open))` |
| `Exp(x)` | Exponential with overflow safeguards | `Exp(CSRank($volume))` |
| `Tanh(x)` | Hyperbolic tangent, mapping into (-1,1) | `Tanh(ts_delta($close,5))` |
| `Sigmoid(x)` | Sigmoid transform, mapping into (0,1) | `Sigmoid(ts_rank($close,20))` |
| `Square(x)`, `Cube(x)` | Square and cube | `Square(CSRank($close))` |

## Custom rolling operators

Each operator below takes one series and one window. Do not supply a
multi-series `(high,low,close)` argument list.

| Operator | Meaning | Example |
| --- | --- | --- |
| `ts_volatility(x,N)` | Volatility of log returns | `ts_volatility($close,20)` |
| `ts_sharpe_ratio(x,N)` | Mean-to-standard-deviation approximation with safeguards | `ts_sharpe_ratio($close,60)` |
| `ts_momentum(x,N)` | Deviation from the rolling mean | `ts_momentum($close,20)` |
| `ts_bollinger_upper(x,N)` | Upper Bollinger band | `ts_bollinger_upper($close,20)` |
| `ts_bollinger_lower(x,N)` | Lower Bollinger band | `ts_bollinger_lower($close,20)` |
| `ts_bollinger_width(x,N)` | Bollinger band width | `ts_bollinger_width($close,20)` |
| `ts_price_position(x,N)` | Position within the rolling minimum/maximum range | `ts_price_position($close,20)` |
| `ts_rsi(x,N)` | Relative strength index | `ts_rsi($close,14)` |
| `ts_macd(x,N)` | Single-window MACD approximation | `ts_macd($close,26)` |
| `ts_stoch(x,N)` | Single-window stochastic %K approximation | `ts_stoch($close,14)` |
| `ts_williams_r(x,N)` | Single-window Williams %R approximation | `ts_williams_r($close,14)` |
| `ts_cci(x,N)` | Commodity channel index | `ts_cci($close,20)` |
| `ts_atr(x,N)` | ATR-style approximation based on the implementation | `ts_atr($close,14)` |

## Custom pairwise rolling operators

| Operator | Meaning | Example |
| --- | --- | --- |
| `ts_beta(x,y,N)` | Covariance divided by benchmark variance | `ts_beta($close,$index,60)` |
| `ts_alpha(x,y,N)` | Asset mean return minus benchmark mean return | `ts_alpha($close,$index,60)` |
| `ts_treynor_ratio(x,y,N)` | Simplified Treynor-style implementation | `ts_treynor_ratio($close,$index,60)` |

## Examples

- Cross-sectional price movement: `CSRank(Abs($close-$open))`
- Smoothed momentum: `ts_mean(Tanh(ts_delta($close,1)),10)`
- Price/volume correlation: `ts_corr(CSRank($close),CSRank($volume),30)`
- RSI: `ts_rsi($close,14)`
- Simplified MACD: `ts_macd($close,26)`

The implementation and enabled operator tags determine the exact numerical
behavior. Standard operators live in `alphagen_generic/operators.py`, custom
operators in `operators_cstm.py`, and optional Qlib and TA-Lib extensions in
`operators_qlib.py` and `operators_talib.py`.
