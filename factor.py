import joblib
from typing import Dict

import xarray as xr

from modules.config import *
from modules.factor_construction.data_engine import DataEngine
from modules.factor_construction.compute_engine import ComputeEngine


def factor_past_return(data_engine: DataEngine, compute_engine: ComputeEngine,
                       factors: Dict[str, xr.DataArray]) -> None:
    for i in [1, 2, 3, 5, 10, 15]:
        factors["PAST_RETURN_" + str(i)] = compute_engine.shift(
            compute_engine.ret(data_engine.close_adj, data_engine.date_col, i),
            data_engine.date_col, 1)
    return factors


def factor_past_return_corr(data_engine: DataEngine,
                            compute_engine: ComputeEngine,
                            factors: Dict[str, xr.DataArray]) -> None:
    for i in [5, 10, 15, 20]:
        factors["PAST_RETURN_VOL_CORR" + str(i)] = compute_engine.shift(
            compute_engine.rolling_corr(
                compute_engine.ret(data_engine.close_adj, data_engine.date_col,
                                   i), data_engine.volume,
                data_engine.date_col, i), data_engine.date_col, 1)
    return factors


def compute_factor(data_engine: DataEngine,
                   compute_engine: ComputeEngine) -> None:
    """
    Compute the specified factor and save it to disk.

    Parameters:
    - data_engine (DataEngine): The data engine.
    - compute_engine (ComputeEngine): The compute engine.
    """
    # Compute the returns using the compute_engine.ret() method
    returns = compute_engine.ret(data_engine.close_adj, data_engine.date_col,
                                 1)

    # Initialize an empty dictionary to store the factors
    factors = {}

    # Compute the factors for different time periods
    factors = factor_past_return(data_engine, compute_engine, factors)
    factors = factor_past_return_corr(data_engine, compute_engine, factors)

    # Convert the factors dictionary into an xarray Dataset
    factors = xr.Dataset(factors)

    # Save the factors to disk using the joblib.dump() method
    joblib.dump(factors, f"{FACTOR_PATH}/factors.joblib")

    # Save the returns to disk using the joblib.dump() method
    joblib.dump(returns, f"{FACTOR_PATH}/returns.joblib")


if __name__ == "__main__":
    data_engine = DataEngine(DATA_PATH, DATE_ID, INSTRUMENT_ID)
    compute_engine = ComputeEngine()

    compute_factor(data_engine, compute_engine)
