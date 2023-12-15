from abc import ABC, abstractmethod
from ..config import INSTRUMENT_ID
import numpy as np


class BaseStrategy(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_position(self, date, signal):
        pass


class TopKStrategy(BaseStrategy):
    """
    A strategy that selects the top K stocks based on a given signal.

    Parameters:
    - k (int): The number of top stocks to select.

    Methods:
    - get_position(signal): Calculates the position for each stock based on the signal.

    """

    def __init__(self, k=10):
        assert k >= 1 and k <= 99
        self.k = k

    def get_position(self, signal):
        """
        Calculates the position for each stock based on the signal.

        Parameters:
        - signal (pandas.DataFrame): The signal data for each stock.

        Returns:
        - pandas.DataFrame: The position data for each stock.

        """
        signal_rank = signal.rank(dim=INSTRUMENT_ID)
        signal_rank_thresh = np.percentile(
            signal_rank.dropna(dim=INSTRUMENT_ID), 100 - self.k)
        selected_stocks = (signal_rank >= signal_rank_thresh).astype(int)
        return selected_stocks / selected_stocks.sum(dim=INSTRUMENT_ID)
