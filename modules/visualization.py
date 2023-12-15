import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_visualization(portfolio_return_nofee,
                           portfolio_return,
                           rank_ic,
                           pearson_corr,
                           returns,
                           test_samples=252):
    """
    Generate visualization charts.

    Args:
    portfolio_return_nofee: Portfolio return without fee (numpy array)
    portfolio_return: Portfolio return with fee (numpy array)
    rank_ic: Rank IC (numpy array)
    pearson_corr: Pearson correlation coefficient (numpy array)
    returns: Return data (pandas DataFrame)
    test_samples: Number of test samples (default is 252)

    Returns:
    fig: Generated chart object
    """
    # Create a figure with 3 subplots
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    # Calculate cumulative returns
    portfolio_cumreturn_nofee = np.exp(
        np.cumsum(np.nan_to_num(portfolio_return_nofee[-test_samples:])))
    portfolio_cumreturn = np.exp(
        np.cumsum(np.nan_to_num(portfolio_return[-test_samples:])))

    # Plot cumulative returns
    ax[0].plot(returns.date[-test_samples:], portfolio_cumreturn_nofee)
    ax[0].plot(returns.date[-test_samples:], portfolio_cumreturn)
    ax[0].legend(['No Fee', 'With Fee'])
    ax[0].set_title('Cumulative Return')

    # Plot daily returns
    ax[1].plot(returns.date[-test_samples:],
               portfolio_return_nofee[-test_samples:])
    ax[1].plot(returns.date[-test_samples:], portfolio_return[-test_samples:])
    ax[1].legend(['No Fee', 'With Fee'])
    ax[1].set_title('Daily Return')

    # Plot rank IC and Pearson correlation
    ax[2].plot(returns.date[-test_samples:], rank_ic[-test_samples:])
    ax[2].plot(returns.date[-test_samples:], pearson_corr[-test_samples:])
    ax[2].legend(['Rank IC', 'Pearson Corr'])
    ax[2].set_title('Rank IC and Pearson Corr')

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Return the generated chart object
    return fig


def generate_heatmap(factors):
    """
    Generate a heatmap of correlation matrix.

    Args:
    factors: Factors data (pandas DataFrame)

    Returns:
    None
    """
    # Create a figure and set the size
    plt.figure(figsize=(15, 15))

    # Generate the heatmap using seaborn
    sns.heatmap(factors.to_dataframe().corr(),
                vmax=1,
                vmin=-1,
                center=0,
                cmap='RdBu_r',
                annot=True)

    # Adjust the layout of the heatmap
    plt.tight_layout()
