from ..config import INSTRUMENT_ID

import numpy as np


class BackTrader(object):
    """
    BackTrader class is used to perform backtesting operations.

    Args:
        strategy (object): Backtesting strategy object.
        fee (float, optional): Transaction fee, default is 0.001.

    Attributes:
        strategy (object): Backtesting strategy object.
        fee (float): Transaction fee.

    Methods:
        backtest(signals, returns): Perform backtesting operation and return portfolio returns.

    """

    def __init__(self, strategy, fee=0.0008):
        self.strategy = strategy
        self.fee = fee

    def backtest(self, signals, returns):
        """
            Perform backtesting operation and return portfolio returns.

            Args:
                signals (pandas.DataFrame): Trading signal data.
                returns (pandas.DataFrame): Stock returns data.

            Returns:
                list: List of portfolio returns.

            """
        portfolio_returns = []
        last_position = np.zeros(len(signals.stk_id))
        for date in signals.date:
            signal = signals.sel(date=date)
            if signal.isnull().all():
                portfolio_returns.append(np.nan)
                continue
            position = self.strategy.get_position(signal)
            _return = returns.sel(date=date)

            portfolio_return = float(
                (position * _return).sum(dim=INSTRUMENT_ID))
            additional_fee = np.abs(position - last_position).sum() * self.fee

            portfolio_returns.append(float(portfolio_return - additional_fee))
            last_position = position

        return portfolio_returns
