import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from tqdm import tqdm


class FactorUtils:
    """
    Factor analysis tool class, including preprocessing and index calculation functions
    """

    def __init__(self, data_close=None, data_open=None, data_volume=None,
                 data_high=None, data_low=None, ST=None, new=None, yjl=None, marketcap=None):
        """
        Initialization factor tool class

        Parameters:
        - data_close: closing price data
        - data_open: opening price data
        - data_volume: Volume data
        - data_high: Highest price data
        - data_low: lowest price data
        - ST: ST stock data
        - new: New stock data
        - yjl: Premium rate data
        - marketcap: Market value data
        """
        self.data_close = data_close
        self.data_open = data_open
        self.data_volume = data_volume
        self.data_high = data_high
        self.data_low = data_low
        self.ST = ST
        self.new = new
        self.yjl = yjl
        self.marketcap = marketcap

        # Date index of standardized data
        if self.data_close is not None:
            self.data_close.index = pd.to_datetime(self.data_close.index)
        if self.data_open is not None:
            self.data_open.index = pd.to_datetime(self.data_open.index)
        if self.data_volume is not None:
            self.data_volume.index = pd.to_datetime(self.data_volume.index)
        if self.data_high is not None:
            self.data_high.index = pd.to_datetime(self.data_high.index)
        if self.data_low is not None:
            self.data_low.index = pd.to_datetime(self.data_low.index)
        if self.ST is not None:
            self.ST.index = pd.to_datetime(self.ST.index)
        if self.new is not None:
            self.new.index = pd.to_datetime(self.new.index)

    def apply_valid_mask(self, factor_df: pd.DataFrame, data_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Unified mask function: filter the factor DataFrame by data_valid (1=valid, 0=invalid)

        Parameters:
        - factor_df: DataFrame, row=date, column=stock (same dimension as data_valid)
        - data_valid: DataFrame, mask matrix (1=valid, 0=invalid)

        Returns:
        - Filtered DataFrame
        """
        if data_valid is not None:
            # Align rows and columns to avoid index inconsistency
            factor_df = factor_df.reindex(index=data_valid.index, columns=data_valid.columns)
            factor_df = factor_df.where(data_valid == 1)
        return factor_df

    # ==================== Preprocessing function ====================

    def exclude_ST(self, factor):
        """
        Exclude the factor value of ST shares when the underlying stock on that day is eliminated

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Processed factor data
        """
        if self.ST is None:
            print("Warning: ST data is not provided, skip ST filtering")
            return factor

        for date in factor.keys():
            factor[date] = factor[date][self.ST.loc[date] == 0]
        return factor

    def exclude_newbond(self, factor):
        """
        Exclude the factor values that are new stocks on the day

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Processed factor data
        """
        if self.new is None:
            print("Warning: New stock data is not provided, skip new stock filtering")
            return factor

        for date in factor.keys():
            factor[date] = factor[date][self.new.loc[date] == 0]
        return factor

    def winsorize(self, factor, low=0.01, up=0.99):
        """
        Winsorization replaces factors exceeding quantiles with boundary values

        Parameters:
        - factor: Factor data in dict format
        - low: lower quantile, default 0.01
        - up: upper quantile, default 0.99

        Returns:
        - Processed factor data
        """
        for date in factor.keys():
            s = factor[date].copy()
            lower = s.quantile(low)
            upper = s.quantile(up)
            factor[date] = s.clip(lower, upper)
        return factor

    def factor_std(self, factor):
        """
        Standardization process

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Processed factor data
        """
        for date in factor.keys():
            factor[date] = (factor[date] - factor[date].mean()) / factor[date].std()
        return factor

    def win_std(self, factor):
        """
        Winsor and standardize, factor is dataframe format, return dataframe format

        Parameters:
        - factor: Factor data in DataFrame format

        Returns:
        - processed DataFrame format factor data
        """
        return pd.DataFrame(self.factor_std(self.winsorize(self.factor_dict(factor)))).T

    def factor_dict(self, factor):
        """
        Change the factor data table in dataframe format into dict format

        Parameters:
        - factor: Factor data in DataFrame format

        Returns:
        - Factor data in dict format
        """
        factor_dict = {}
        for day in list(factor.index):
            factor_dict[day] = factor.loc[day].sort_values()
        return factor_dict

    def net_yjl(self, factor):
        """
        Neutralize the premium rate

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Processed factor data
        """
        if self.yjl is None:
            print("Warning: Premium rate data is not provided, skip premium rate neutralization")
            return factor

        data_yjl = self.factor_dict(self.yjl)
        for date in factor.keys():
            y = factor[date].dropna()
            x = data_yjl[date].dropna()
            x, y = x.align(y, join='inner')
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            factor[date] = model.resid.sort_values()
        return factor

    def net_marketcap(self, factor):
        """
        Neutral to market capitalization

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Processed factor data
        """
        if self.marketcap is None:
            print("Warning: Market capitalization data is not provided, skip market capitalization neutralization")
            return factor

        data_marketcap = self.factor_dict(self.marketcap)
        for date in factor.keys():
            y = factor[date].dropna()
            x = data_marketcap[date].dropna()
            x, y = x.align(y, join='inner')
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            factor[date] = model.resid.sort_values()
        return factor

    # ==================== Indicator calculation function ====================

    def cal_IC(self, factor):
        """
        Daily IC

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - IC sequence
        """
        # Factor table & next day income table (days x stocks)
        factor_df = pd.DataFrame(factor).T
        returns = (self.data_close.shift(-1) / self.data_close - 1)
        # Align date and stock
        common_idx = factor_df.index.intersection(returns.index)
        common_cols = factor_df.columns.intersection(returns.columns)
        f = factor_df.loc[common_idx, common_cols]
        r = returns.loc[common_idx, common_cols]
        # Daily cross section Pearson correlation
        ic = f.corrwith(r, axis=1, method='pearson')
        return ic

    def cal_RankIC(self, factor):
        """
        Daily RankIC

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - RankIC sequence
        """
        # Factor table & next day income table (days x stocks)
        factor_df = pd.DataFrame(factor).T
        returns = (self.data_close.shift(-1) / self.data_close - 1)
        # Align date and stock
        common_idx = factor_df.index.intersection(returns.index)
        common_cols = factor_df.columns.intersection(returns.columns)
        f = factor_df.loc[common_idx, common_cols]
        r = returns.loc[common_idx, common_cols]
        # Daily cross section Spearman correlation
        rank_ic = f.corrwith(r, axis=1, method='spearman')
        return rank_ic

    def cal_excess_ic(self, factor):
        """
        Daily excess IC

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Excess IC sequence
        """
        # Factor table & next day income table (days x stocks)
        factor_df = pd.DataFrame(factor)
        returns = (self.data_close.shift(-1)/self.data_close - 1).T
        # Calculate the benchmark rate of return (equally weighted average)
        benchmark_returns = returns.mean(axis=1)
        # Calculate excess returns
        excess_returns = returns.sub(benchmark_returns, axis=0)
        # Align date and stock
        common_idx = factor_df.index.intersection(returns.index)
        common_cols = factor_df.columns.intersection(returns.columns)
        f = factor_df.loc[common_idx, common_cols]
        r = excess_returns.loc[common_idx, common_cols]
        excess_ic = f.corrwith(r, axis=1, method='spearman')
        return excess_ic

    def cal_IC_mean(self, factor):
        """
        Daily IC average

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - IC mean
        """
        IC = self.cal_IC(factor).mean()
        return IC

    def cal_ICIR(self, factor):
        """
        Daily ICIR value

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - ICIR value
        """
        IC = self.cal_IC(factor)
        ICIR = IC.mean()/IC.std() if IC.std() != 0 else np.nan
        return ICIR

    def cal_RankIC_mean(self, factor):
        """
        Daily RankIC average

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - RankIC mean
        """
        RankIC = self.cal_RankIC(factor).mean()
        return RankIC

    def cal_RankICIR(self, factor):
        """
        Daily RankICIR value

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - RankICIR value
        """
        RankIC = self.cal_RankIC(factor)
        ICIR = RankIC.mean() / RankIC.std() if RankIC.std() != 0 else np.nan
        return ICIR

    def cal_excess_ic_mean(self, factor):
        """
        Daily excess IC average

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Excess IC mean
        """
        EIC = self.cal_excess_ic(factor)
        return EIC.mean()

    def cal_excess_icir(self, factor):
        """
        Daily excess ICIR value

        Parameters:
        - factor: Factor data in dict format

        Returns:
        - Excess ICIR value
        """
        EIC = self.cal_excess_ic(factor)
        EICIR = EIC.mean() / EIC.std() if EIC.std() != 0 else np.nan
        return EICIR

    def cal_ann_return(self, factor, N):
        """
        Annualized rate of return

        Parameters:
        - factor: Factor data in dict format
        - N: Number of groups

        Returns:
        - Annualized rate of return sequence
        """
        return (pd.DataFrame(self.cal_group_return(factor, N)).T).mean() * 252

    def cal_cum_return(self, factor, N):
        """
        Cumulative rate of return

        Parameters:
        - factor: Factor data in dict format
        - N: Number of groups

        Returns:
        - Cumulative rate of return sequence
        """
        df = pd.DataFrame(self.cal_group_return(factor, N)).T
        if df.empty:
            return pd.Series(dtype=float)
        def _cum_last(s):
            s = s.dropna()
            return np.nan if s.empty else float((1.0 + s).cumprod().iloc[-1] - 1.0)
        return df.apply(_cum_last, axis=0)

    def cal_sharp(self, factor, N, rate=0.02):
        """
        Annualized Sharpe Ratio

        Parameters:
        - factor: Factor data in dict format
        - N: Number of groups
        - rate: risk-free interest rate, default 0.02

        Returns:
        - Sharpe ratio sequence
        """
        annual_return = self.cal_ann_return(factor, N) - rate
        std = pd.DataFrame(self.cal_group_return(factor, N)).T.std() * math.sqrt(252)
        sharp = annual_return / std
        return sharp

    def cal_winrate(self, factor, N):
        """
        Daily winning rate

        Parameters:
        - factor: Factor data in dict format
        - N: Number of groups

        Returns:
        - winning rate sequence
        """
        group_return = pd.DataFrame(self.cal_group_return(factor, N)).T
        return (group_return > 0).sum() / len(group_return.index)

    def cal_MDD(self, factor, N):
        """
        Maximum drawdown

        Parameters:
        - factor: Factor data in dict format
        - N: Number of groups

        Returns:
        - maximum drawdown sequence
        """
        df = pd.DataFrame(self.cal_group_return(factor, N)).T
        if df.empty:
            return pd.Series(dtype=float)
        def _col_mdd(s):
            s = s.dropna()
            if s.empty:
                return np.nan
            nav = (1.0 + s).cumprod()
            dd  = nav / nav.cummax() - 1.0
            return float(abs(dd.min()))
        return df.apply(_col_mdd, axis=0)

    def cal_group_return(self, factor, N):
        """
        Calculate group income

        Parameters:
        - factor: Factor data in dict format
        - N: Number of groups

        Returns:
        - Grouping income dictionary
        """
        date_list = list(factor.keys())
        group_return = {}

        # Determine the direction based on RankIC
        is_positive_rankic = self.cal_RankIC_mean(factor) > 0

        for date in date_list:
            # Available stock collection for the day (dropna has automatically excluded invalid samples)
            stock_series = factor[date].dropna()
            stock_series = stock_series.sort_values()
            stock_all = stock_series.index.tolist()
            stock_return = {}

            # Income of each group
            for i in range(1, N+1):
                l = int(len(stock_series) / N * (i - 1))
                r = int(len(stock_series) / N * i)
                stock_i = stock_series.iloc[l:r].index.tolist()
                if len(stock_i) == 0:
                    stock_i_return = float('nan')
                else:
                    rets = (self.data_close.shift(-1) / self.data_close - 1).loc[date, stock_i]
                    stock_i_return = rets.mean()
                stock_return['group' + str(i)] = stock_i_return

            # Benchmark
            if len(stock_all) == 0:
                bench = float('nan')
            else:
                bench = ((self.data_close.loc[:, stock_all].shift(-1) / self.data_close.loc[:, stock_all] - 1)
                         .loc[date].mean())
            stock_return['benchmark'] = bench

            # Long and short/long absolute/long excess
            if is_positive_rankic:
                stock_return['long-short'] = stock_return['group' + str(N)] - stock_return['group1']
                stock_return['long_abs'] = stock_return['group' + str(N)]
                stock_return['long_excess'] = stock_return['group' + str(N)] - stock_return['benchmark']
            else:
                stock_return['long-short'] = stock_return['group1'] - stock_return['group' + str(N)]
                stock_return['long_abs'] = stock_return['group1']
                stock_return['long_excess'] = stock_return['group1'] - stock_return['benchmark']

            group_return[date] = pd.Series(stock_return)

        return group_return
