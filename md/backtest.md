# Backtest Metrics

Metric names are shared by backtest exports, selection weights, statistics, and
LLM prompts.

| Metric | Meaning |
| --- | --- |
| `IC` | Cross-sectional Pearson correlation between factor values and future returns |
| `RankIC` | Cross-sectional Spearman correlation between factor values and future returns |
| `ExcessIC` | Cross-sectional correlation between factor values and future returns above the benchmark |

The following metrics use three portfolio suffixes: `long_excess` for the long
portfolio relative to its benchmark, `long_short` for the long-short portfolio,
and `long_abs` for the long portfolio's absolute returns.

| Metric prefix | Meaning |
| --- | --- |
| `annualized` | Annualized return |
| `cumulative` | Cumulative return over the evaluation period |
| `sharpe` | Annualized Sharpe ratio |
| `win_rate` | Fraction of periods with a positive return |
| `drawdown` | Absolute maximum drawdown of the return curve |

For example, `annualized_long_excess` is the long portfolio's annualized excess
return, `sharpe_long_short` is the long-short portfolio's Sharpe ratio, and
`drawdown_long_abs` is the long portfolio's maximum drawdown.

The selection pipeline ranks `IC`, `RankIC`, and `ExcessIC` by absolute value.
Individual factor analysis retains their signs. Calculation details, benchmark
construction, and annualization are implemented in `backtest/factor_utils.py`.
