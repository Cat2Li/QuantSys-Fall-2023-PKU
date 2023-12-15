import numpy as np
import pandas as pd
import xarray as xr
from IPython.core.display import display

# Function to transform the factor by ranking the values
rank_transform = lambda feature: feature.rank('stk_id')

# Function to transform the returns by keeping the same values
id_transform = lambda x: x

# Function to calculate the correlation score between two variables
corr_score = lambda x, y: np.corrcoef(x, y)[0][1]


def calc_relationship(factor, returns, transform_factor, transform_returns,
                      func):
    """
    Calculate the relationship between a factor and returns.

    Parameters:
    factor (xarray.DataArray): The factor data.
    returns (xarray.DataArray): The returns data.
    transform_factor (function): The function to transform the factor data.
    transform_returns (function): The function to transform the returns data.
    func (function): The function to calculate the relationship between the factor and returns.

    Returns:
    list: The list of calculated results for the relationship between the factor and returns.
    """
    result = []
    for date in returns.date:
        factor_ = factor.sel(date=date)
        return_ = returns.sel(date=date)

        if factor_.isnull().all():
            result.append(np.nan)
            continue

        factor_feat = transform_factor(factor_)
        return_feat = transform_returns(return_)

        mask = factor_feat.notnull() & return_feat.notnull()

        result.append(func(factor_feat[mask], return_feat[mask]))

    return result


def calc_rank_ic(factor, returns):
    """
    Calculate the rank IC (Information Coefficient) between a factor and returns.

    Parameters:
    factor (xarray.DataArray): The factor data.
    returns (xarray.DataArray): The returns data.

    Returns:
    list: The list of calculated rank IC values.
    """
    return calc_relationship(factor, returns, rank_transform, rank_transform,
                             corr_score)


def calc_pearson_corr(factor, returns):
    """
    Calculate the Pearson correlation between a factor and returns.

    Parameters:
    factor (xarray.DataArray): The factor data.
    returns (xarray.DataArray): The returns data.

    Returns:
    list: The list of calculated Pearson correlation values.
    """
    return calc_relationship(factor, returns, id_transform, id_transform,
                             corr_score)


def make_summary(portfolio_return_nofee,
                 portfolio_return,
                 market_return,
                 rank_ic,
                 pearson_corr,
                 test_samples=252):
    """
    Make a summary of the rank IC and Pearson correlation.

    Parameters:
    rank_ic (list): The list of calculated rank IC values.
    pearson_corr (list): The list of calculated Pearson correlation values.

    Returns:
    pandas.DataFrame: The summary of the rank IC and Pearson correlation.
    """
    display(
        pd.DataFrame(
            {
                '平均 Pearson 相关系数': np.nanmean(pearson_corr[-test_samples:]),
                '平均 Spearman 秩相关系数': np.nanmean(rank_ic[-test_samples:]),
            },
            index=["统计量 (test samples = %d)" % test_samples]))

    portfolio_nofee = np.cumsum(portfolio_return_nofee[-test_samples:])
    portfolio = np.cumsum(portfolio_return[-test_samples:])

    display(
        pd.DataFrame(
            {
                '年化收益率': [
                    np.nanmean(portfolio_return_nofee[-test_samples:]) * 252,
                    np.nanmean(portfolio_return[-test_samples:]) * 252
                ],
                '年化波动率': [
                    np.nanstd(portfolio_return_nofee[-test_samples:]) *
                    np.sqrt(252),
                    np.nanstd(portfolio_return[-test_samples:]) * np.sqrt(252)
                ],
                '年化夏普比率': [
                    np.nanmean(portfolio_return_nofee[-test_samples:]) /
                    np.nanstd(portfolio_return_nofee[-test_samples:]) *
                    np.sqrt(252),
                    np.nanmean(portfolio_return[-test_samples:]) /
                    np.nanstd(portfolio_return[-test_samples:]) * np.sqrt(252)
                ],
                '测试期年化超额收益':
                [(np.nanmean(portfolio_return_nofee[-test_samples:]) -
                  np.nanmean(market_return[-test_samples:])) * 252,
                 (np.nanmean(portfolio_return[-test_samples:]) -
                  np.nanmean(market_return[-test_samples:])) * 252],
                '测试期最大回撤': [
                    np.min(portfolio_nofee -
                           np.maximum.accumulate(portfolio_nofee)),
                    np.min(portfolio - np.maximum.accumulate(portfolio))
                ],
            },
            index=[
                "无手续费 (test samples = %d)" % test_samples,
                "有手续费 (test samples = %d)" % test_samples
            ]))
