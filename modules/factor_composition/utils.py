import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import make_scorer

corr_scorer = make_scorer(
    lambda x, y: np.mean(x * y) / np.sqrt(np.mean(x**2) * np.mean(y**2)),
    greater_is_better=True)


def train_test_split(factors, returns, test_size=252):
    """
    Split the factors and returns data into training and testing datasets.

    Parameters:
    factors (xarray.Dataset): Dataset containing the factors data.
    returns (xarray.Dataset): Dataset containing the returns data.
    test_size (int): Number of data points to be included in the testing dataset. Default is 252.

    Returns:
    train_dataset (pandas.DataFrame): Training dataset containing the factors and returns data.
    test_dataset (pandas.DataFrame): Testing dataset containing the factors and returns data.
    """

    # Calculate the total number of data points
    data_dates = factors.date.shape[0]

    # Calculate the index to split the data into training and testing datasets
    train_end = int(data_dates) - test_size

    # Split the factors and returns data into training and testing datasets
    factors_train, factors_test, returns_train, returns_test = (
        factors.isel(date=slice(None, train_end)),
        factors.isel(date=slice(train_end, None)),
        returns.isel(date=slice(None, train_end)),
        returns.isel(date=slice(train_end, None)),
    )

    # Convert the training datasets to pandas DataFrame and drop any rows with missing values
    train_dataset = factors_train.assign(
        RETURN=returns_train).to_dataframe().dropna()

    # Convert the testing datasets to pandas DataFrame and drop any rows with missing values
    test_dataset = factors_test.assign(
        RETURN=returns_test).to_dataframe().dropna()

    return train_dataset, test_dataset


def unpack_signals(regressor, train_dataset, test_dataset, returns):
    fitted = regressor.predict(train_dataset)
    pred = regressor.predict(test_dataset)

    train_dataset["PRED"] = fitted
    test_dataset["PRED"] = pred

    signals = pd.concat([train_dataset["PRED"],
                         test_dataset["PRED"]]).sort_index().to_xarray()

    dataset = xr.Dataset({"returns": returns, "signals": signals})

    return dataset["signals"]
