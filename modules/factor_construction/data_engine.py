import pandas as pd

from typing import List
from numpy import ndarray
from xarray import Dataset


class DataEngine(object):
    """
    A class for loading and processing financial data.

    Parameters:
    - data_path (str): The path to the data file.
    - date_col (str): The name of the column containing the date information. Default is "date".
    - instrument_col (str): The name of the column containing the instrument information. Default is "stk_id".
    """

    def __init__(self, data_path: str, date_col: str,
                 instrument_col: str) -> None:
        self.data_path = data_path
        self.date_col = date_col
        self.instrument_col = instrument_col

        self.data = self.load_data(self.data_path)
        self.install_features()

    def load_data(self, path: str) -> Dataset:
        """
        Load the data from the specified path and preprocess it.

        Parameters:
        - path (str): The path to the data file.

        Returns:
        - xarray.Dataset: The preprocessed data in xarray format.
        """
        df = pd.read_feather(path)
        df = df.set_index([self.instrument_col, self.date_col])
        df = df.sort_index()
        for adj_col in ["open", "high", "low", "close"]:
            df[f"{adj_col}_adj"] = df[adj_col] * df["cumadj"]
        return df.to_xarray()

    @property
    def instruments(self) -> ndarray:
        """
        Get the list of instruments in the data.

        Returns:
        - numpy.ndarray: The array of instrument names.
        """
        return self.data[self.instrument_col].values

    @property
    def dates(self) -> ndarray:
        """
        Get the list of dates in the data.

        Returns:
        - numpy.ndarray: The array of dates.
        """
        return self.data[self.date_col].values

    @property
    def features(self) -> List[str]:
        """
        Get the list of available features in the data.

        Returns:
        - list: The list of feature names.
        """
        return list(self.data.data_vars.keys())

    def install_features(self) -> None:
        """
        Install the features as attributes of the DataEngine object.
        """
        for feature in self.features:
            setattr(self, feature, self.data[feature])
