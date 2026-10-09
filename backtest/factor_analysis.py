"""
Factor Analysis Module
Factor analysis module - contains functions under the headings Analysis, Plot, Backtest

Function extracted from high_freq_factor_anal.ipynb:
- factor_analyse_simple: Simplified factor analysis function (automatically prints core indicators)
- plot_return: Drawing function (only saves the image, does not display)
- save_backtest_results_by_period: Save backtest results
- analyze_backtest_statistics: Analyze backtest statistics

Core indicator output:
- Excess IC
- annualized_long_excess
- sharpe_long_excess
- Display in 3 rows according to TRAIN, VALID, TEST
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from backtest.metric_schema import normalize_metric_columns

warnings.filterwarnings('ignore')


class FactorAnalysis:
    """
    Factor analysis class - including analysis, drawing, backtest result processing functions
    """

    def __init__(self, factor_utils=None):
        """
        Initialize factor analyzer

        Parameters:
        - factor_utils: FactorUtils instance, used for index calculation
        """
        self.factor_utils = factor_utils

    def factor_analyse_simple(self, factor, train_period=None, val_period=None, test_period=None,
                            yjl=False, mar=False, win=True, std=True, N=None,
                            factor_name="Factor", expression=None, alphaforge_score=None):
        """
        Simplified factor analysis function

        Parameters:
        - factor: DataFrame, factor data
        - train_period: tuple, training period (start_date, end_date)
        - val_period: tuple, validation period (start_date, end_date)
        - test_period: tuple, test period (start_date, end_date)
        - yjl: bool, whether to neutralize the premium rate
        - mar: bool, whether to perform market value neutralization
        - win: bool, whether to perform shrinking processing
        - std: bool, whether to standardize
        - N: int, number of groups
        - factor_name: str, factor name
        - expression: str, factor expression
        - alphaforge_score: float, AlphaForge score

        Returns:
        - DataFrame: Analysis results
        """
        if self.factor_utils is None:
            raise ValueError("Need to provide FactorUtils instance")

        # Core three items + annualization/cumulative/Sharpe/win rate/drawdown
        features = [
            'IC', 'RankIC', 'ExcessIC',
            'annualized_long_excess','cumulative_long_excess','sharpe_long_excess','win_rate_long_excess','drawdown_long_excess',
            'annualized_long_short','cumulative_long_short','sharpe_long_short','win_rate_long_short','drawdown_long_short',
            'annualized_long_abs','cumulative_long_abs','sharpe_long_abs','win_rate_long_abs','drawdown_long_abs',
        ]

        def calculate_period_metrics(period, period_name):
            """Calculate indicators for a specified period (update progress item by item)"""
            if period is None:
                return [np.nan] * len(features)

            try:
                start_date, end_date = period
                factor_df = factor.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                factor_dict_format = self.factor_utils.factor_dict(factor_df)

                # Data preprocessing
                if yjl:
                    factor_dict_format = self.factor_utils.net_yjl(factor_dict_format)
                if mar:
                    factor_dict_format = self.factor_utils.net_marketcap(factor_dict_format)
                if win:
                    factor_dict_format = self.factor_utils.winsorize(factor_dict_format)
                if std:
                    factor_dict_format = self.factor_utils.factor_std(factor_dict_format)

                with tqdm(total=8, desc=f"{period_name} Index calculation", leave=False) as pbar:
                    values = [np.nan] * len(features)

                    # Three core items
                    values[0] = self.factor_utils.cal_IC_mean(factor_dict_format); pbar.update(1)
                    values[1] = self.factor_utils.cal_RankIC_mean(factor_dict_format); pbar.update(1)
                    values[2] = self.factor_utils.cal_excess_ic_mean(factor_dict_format); pbar.update(1)

                    # Get each column of indicators at one time with consistent definitions
                    ann_series   = self.factor_utils.cal_ann_return(factor_dict_format, N); pbar.update(1)
                    sharp_series = self.factor_utils.cal_sharp(factor_dict_format, N, rate=0.02); pbar.update(1)
                    win_series   = self.factor_utils.cal_winrate(factor_dict_format, N); pbar.update(1)
                    cum_series   = self.factor_utils.cal_cum_return(factor_dict_format, N); pbar.update(1)
                    mdd_series   = self.factor_utils.cal_MDD(factor_dict_format, N); pbar.update(1)

                    # long_excess
                    values[3]  = ann_series.get('long_excess', np.nan)
                    values[4]  = cum_series.get('long_excess', np.nan)
                    values[5]  = sharp_series.get('long_excess', np.nan)
                    values[6]  = win_series.get('long_excess', np.nan)
                    values[7]  = mdd_series.get('long_excess', np.nan)

                    # long_short
                    values[8]  = ann_series.get('long-short', np.nan)
                    values[9]  = cum_series.get('long-short', np.nan)
                    values[10] = sharp_series.get('long-short', np.nan)
                    values[11] = win_series.get('long-short', np.nan)
                    values[12] = mdd_series.get('long-short', np.nan)

                    # long_abs
                    values[13] = ann_series.get('long_abs', np.nan)
                    values[14] = cum_series.get('long_abs', np.nan)
                    values[15] = sharp_series.get('long_abs', np.nan)
                    values[16] = win_series.get('long_abs', np.nan)
                    values[17] = mdd_series.get('long_abs', np.nan)

                return values

            except Exception as e:
                print(f"{period_name} Calculation error: {e}")
                return [np.nan] * len(features)

        # Dynamically construct the result dictionary, which only contains non-None periods
        result_dict = {}

        # Calculate the indicators of each period, and only add non-None periods
        if train_period is not None:
            train_values = calculate_period_metrics(train_period, "Training period")
            result_dict['TRAIN'] = train_values

        if val_period is not None:
            val_values = calculate_period_metrics(val_period, "Validation period")
            result_dict['VALID'] = val_values

        if test_period is not None:
            test_values = calculate_period_metrics(test_period, "Test period")
            result_dict['TEST'] = test_values

        # Create result DataFrame
        result_df = pd.DataFrame(result_dict, index=features)

        # Print core indicators
        self.print_core_metrics(result_df, factor_name, expression, alphaforge_score)

        return result_df

    def print_core_metrics(self, result_df, factor_name="Factor", expression=None, alphaforge_score=None):
        """
        Print core indicators: excess IC, annualized_long_excess, sharpe_long_excess

        Parameters:
        - result_df: DataFrame, analysis results
        - factor_name: str, factor name
        - expression: str, factor expression
        - alphaforge_score: float, AlphaForge score
        """

        # Print factor information
        print(f"  Expr: {expression}")
        print(f"  Score: {alphaforge_score:.4f}\n")

        # Define the indicator to be printed (internal name)
        core_metrics_internal = ['ExcessIC', 'annualized_long_excess', 'sharpe_long_excess']
        # Define the indicator to be printed (display name)
        core_metrics_display = ['ExcessIC', 'excess return', 'Excess Sharpe']

        # Check available periods
        available_periods = []
        for period in ['TRAIN', 'VALID', 'TEST']:
            if period in result_df.columns:
                available_periods.append(period)

        if not available_periods:
            print("  ❌ No period data available")
            return

        # Print header (use tab alignment, standard ”field+tab+column name”)
        header = "Index \t" + "\t".join(available_periods)
        print("  " + header)
        # Simplify the dividing line
        print("  " + "-" * 30)

        # Print all period data for each indicator (using tab alignment)
        for i, metric in enumerate(core_metrics_internal):
            if metric in result_df.index:
                display_name = core_metrics_display[i]
                row_values = [display_name]
                for period in available_periods:
                    value = result_df.loc[metric, period]
                    if pd.notna(value):
                        row_values.append(f"{value:.4f}")
                    else:
                        row_values.append("N/A")
                row = "\t".join(row_values)
                print("  " + row)

    def plot_return(self, factor, N, save_dir=None, factor_name=None):
        """
        Drawing function, factor is the dict format, and N is the group number

        Parameters:
        - factor: dict, factor data
        - N: int, number of groups
        - save_dir: str, save directory
        - factor_name: str, factor name

        Returns:
        - None
        """
        if self.factor_utils is None:
            raise ValueError("Need to provide FactorUtils instance")

        group_return = self.factor_utils.cal_group_return(factor, N)
        group_cum_return = pd.DataFrame(group_return).T

        # Calculate excess returns (relative benchmark for each group)
        for i in range(1, N+1):
            group_cum_return['group'+str(i)] = group_cum_return['group'+str(i)] - group_cum_return['benchmark']

        # Clear all empty lines
        group_cum_return.dropna(how='all', inplace=True)

        # Calculate cumulative income (one day lag to prevent forward-looking)
        group_cum_return = group_cum_return.shift(1)
        group_cum_return.iloc[0] = 0
        group_cum_return = group_cum_return.cumsum()

        # Setup chart
        plt.figure(figsize=(12, 8))

        # Each group
        for i in range(1, N+1):
            col = 'group'+str(i)
            if col in group_cum_return.columns:
                plt.plot(group_cum_return.index, group_cum_return[col], label=f'Group {i}', alpha=0.7)

        # long-short
        if 'long-short' in group_cum_return.columns:
            plt.plot(group_cum_return.index, group_cum_return['long-short'], label='Long-Short', linewidth=2, color='orange')

        # long_excess
        if 'long_excess' in group_cum_return.columns:
            plt.plot(group_cum_return.index, group_cum_return['long_excess'], label='Long-Excess', linewidth=2, color='red')

        # long_abs
        if 'long_abs' in group_cum_return.columns:
            plt.plot(group_cum_return.index, group_cum_return['long_abs'], label='Long-Abs', linewidth=2, color='blue')

        # benchmark
        if 'benchmark' in group_cum_return.columns:
            plt.plot(group_cum_return.index, group_cum_return['benchmark'], label='Benchmark', linewidth=2, color='grey',alpha=0.8)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return / Excess', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save image
        if save_dir and factor_name:
            plots_dir = os.path.join(save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f"{factor_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Image saved: {save_path}")

        # Close the image to release memory
        plt.close()

        return None

    def save_backtest_results_by_period(self, results, analysis_dir, run_dir):
        """
        Save results of backtest_all_factors as three CSV files by period

        Parameters:
        - results: dict, backtest_all_factors result dictionary returned
        - analysis_dir: str, analysis directory path
        - run_dir: str, running directory name
        """

        # Read backtest/factor_list.csv as the basis
        factor_list_path = os.path.join(analysis_dir, "backtest", "factor_list.csv")
        if not os.path.exists(factor_list_path):
            print(f"❌ factor_list.csv does not exist: {factor_list_path}")
            return

        try:
            factor_list_df = pd.read_csv(factor_list_path)
            # Make sure the factor column is of type string to match results.keys()
            factor_list_df['factor'] = factor_list_df['factor'].astype(str)
            results_keys = [str(k) for k in results.keys()]
            factor_list_df = factor_list_df[factor_list_df['factor'].isin(results_keys)].copy()
            print(f"✅ Successfully matched {len(factor_list_df)} factors")
        except Exception as e:
            print(f"❌ Failed to read factor_list.csv: {e}")
            return

        # Define all possible feature columns
        features = [
            'IC', 'RankIC', 'ExcessIC',
            'annualized_long_excess','cumulative_long_excess','sharpe_long_excess','win_rate_long_excess','drawdown_long_excess',
            'annualized_long_short','cumulative_long_short','sharpe_long_short','win_rate_long_short','drawdown_long_short',
            'annualized_long_abs','cumulative_long_abs','sharpe_long_abs','win_rate_long_abs','drawdown_long_abs',
        ]

        # Get the actual period column in results
        if not results:
            print("❌ results is empty and cannot be saved")
            return

        # Obtain the results of the first factor to determine which period columns
        first_factor_key = list(results.keys())[0]
        first_factor_result = results[first_factor_key]
        available_periods = list(first_factor_result.columns)

        # Create backtest folder
        backtest_dir = os.path.join(analysis_dir, "backtest")
        os.makedirs(backtest_dir, exist_ok=True)

        # Period name mapping
        period_mapping = {
            'TRAIN': 'backtest_train.csv',
            'VALID': 'backtest_valid.csv',
            'TEST': 'backtest_test.csv'
        }

        # Create corresponding result files for each period
        for period in available_periods:
            if period in ['TRAIN', 'VALID', 'TEST']:
                # Create the result DataFrame of this period
                period_results = []

                for _, factor_row in factor_list_df.iterrows():
                    factor_name = factor_row['factor']

                    if factor_name in results:
                        # Get all indicator values of this factor in this period
                        factor_result = results[factor_name]
                        if period in factor_result.columns:
                            period_values = factor_result[period].values

                            # Create the result row for this factor
                            result_row = {
                                'factor': factor_name,
                                'expression': factor_row['expression'],
                                'alphaforge_score': factor_row['alphaforge_score']
                            }

                            # Add the values of all feature columns
                            for i, feature in enumerate(features):
                                if i < len(period_values):
                                    result_row[feature] = period_values[i]
                                else:
                                    result_row[feature] = np.nan

                            period_results.append(result_row)
                        else:
                            print(f"⚠️ factor {factor_name} Missing {period} Data")
                    else:
                        print(f"⚠️ factor {factor_name} does not exist in results")

                # Create DataFrame of this period and save it
                if period_results:
                    period_df = pd.DataFrame(period_results)

                    # Determine file name
                    filename = period_mapping.get(period, f'backtest_{period.lower()}.csv')

                    # Save the file to the backtest folder
                    save_path = os.path.join(backtest_dir, filename)
                    period_df.to_csv(save_path, index=False)
                    print(f"💾 Saved {period} Backtest results: {save_path}")
                else:
                    print(f"⚠️ {period} No valid data, skip saving")
            else:
                print(f"⚠️ Skip non-standard period columns: {period}")

    def analyze_backtest_statistics(self, analysis_dir, run_dir):
        """
        Statistically analyze the three CSV files in the backtest folder and generate statistical tables

        Parameters:
        - analysis_dir: str, analysis directory path
        - run_dir: str, running directory name

        Returns:
        - DataFrame: Statistical results
        """
        # Define feature columns (consistent with factor_analyse_simple)
        # Note: IC, RankIC, and excess IC use absolute values during batch statistics, and positive and negative values are still retained during single factor analysis.
        features = [
            'IC', 'RankIC', 'ExcessIC',
            'annualized_long_excess','cumulative_long_excess','sharpe_long_excess','win_rate_long_excess','drawdown_long_excess',
            'annualized_long_short','cumulative_long_short','sharpe_long_short','win_rate_long_short','drawdown_long_short',
            'annualized_long_abs','cumulative_long_abs','sharpe_long_abs','win_rate_long_abs','drawdown_long_abs',
        ]

        # Statistical indicators
        stats_metrics = ['mean', 'median', 'std', 'min', 'max']

        # Period list
        periods = ['TRAIN', 'VALID', 'TEST']

        # backtest folder path
        backtest_dir = os.path.join(analysis_dir, "backtest")

        if not os.path.exists(backtest_dir):
            print(f"❌ The backtest folder does not exist: {backtest_dir}")
            return None

        # Check which period files exist
        existing_periods = []
        for period in periods:
            filename = f"backtest_{period.lower()}.csv"
            file_path = os.path.join(backtest_dir, filename)
            if os.path.exists(file_path):
                existing_periods.append(period)

        if not existing_periods:
            print("❌ No backtest files found")
            return None

        # Create a multi-level column index (containing only the period of existence)
        columns = pd.MultiIndex.from_product([existing_periods, stats_metrics], names=['Period', 'Statistic'])

        # Initialization result DataFrame
        result_df = pd.DataFrame(index=features, columns=columns)

        # Process data for each existing period
        for period in existing_periods:
            filename = f"backtest_{period.lower()}.csv"
            file_path = os.path.join(backtest_dir, filename)

            try:
                # Read data
                period_df = normalize_metric_columns(pd.read_csv(file_path))

                # Perform statistical analysis on each feature column
                for feature in features:
                    if feature in period_df.columns:
                        # Obtain the data of this feature column and exclude the NaN value
                        feature_data = period_df[feature].dropna()

                        if len(feature_data) > 0:
                            # Use absolute value statistics for IC related indicators, and use original values for other indicators.
                            if feature in ['IC', 'RankIC', 'ExcessIC']:
                                # Use absolute values for statistics
                                abs_data = feature_data.abs()
                                result_df.loc[feature, (period, 'mean')] = round(abs_data.mean(), 4)
                                result_df.loc[feature, (period, 'median')] = round(abs_data.median(), 4)
                                result_df.loc[feature, (period, 'std')] = round(abs_data.std(), 4)
                                result_df.loc[feature, (period, 'min')] = round(abs_data.min(), 4)
                                result_df.loc[feature, (period, 'max')] = round(abs_data.max(), 4)
                            else:
                                # Other indicators use original value statistics
                                result_df.loc[feature, (period, 'mean')] = round(feature_data.mean(), 4)
                                result_df.loc[feature, (period, 'median')] = round(feature_data.median(), 4)
                                result_df.loc[feature, (period, 'std')] = round(feature_data.std(), 4)
                                result_df.loc[feature, (period, 'min')] = round(feature_data.min(), 4)
                                result_df.loc[feature, (period, 'max')] = round(feature_data.max(), 4)

            except Exception as e:
                print(f"❌ Processing {filename} An error occurred: {e}")
                continue

        # Save statistical results to the backtest folder
        stats_filename = "backtest_stats.csv"
        stats_path = os.path.join(backtest_dir, stats_filename)

        try:
            result_df.to_csv(stats_path)
            print(f"💾 Saved {existing_periods} Statistical results: {stats_path}")

        except Exception as e:
            print(f"❌ Failed to save statistical results: {e}")

        return result_df


# Convenience function
def analyze_factor(factor, factor_utils, train_period=None, val_period=None, test_period=None,
                  yjl=False, mar=False, win=True, std=True, N=5,
                  factor_name="Factor", expression=None, alphaforge_score=None):
    """
    Convenient factor analysis function

    Parameters:
    - factor: DataFrame, factor data
    - factor_utils: FactorUtils instance
    - train_period: tuple, training period
    - val_period: tuple, validation period
    - test_period: tuple, test period
    - yjl: bool, whether the premium rate is neutral
    - mar: bool, whether the market value is neutral
    - win: bool, whether to shrink the tail
    - std: bool, whether it is standardized
    - N: int, number of groups
    - factor_name: str, factor name
    - expression: str, factor expression
    - alphaforge_score: float, AlphaForge score

    Returns:
    - DataFrame: Analysis results
    """
    analyzer = FactorAnalysis(factor_utils)
    return analyzer.factor_analyse_simple(
        factor=factor,
        train_period=train_period,
        val_period=val_period,
        test_period=test_period,
        yjl=yjl,
        mar=mar,
        win=win,
        std=std,
        N=N,
        factor_name=factor_name,
        expression=expression,
        alphaforge_score=alphaforge_score
    )





# Test function
if __name__ == "__main__":
    print("🧪 Test factor analysis module...")

    # Test code can be added here
    print("✅ Factor analysis module loaded successfully!")
