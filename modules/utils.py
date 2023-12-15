import joblib
import os

from modules.config import FACTOR_PATH


def load_factors():
    factors = joblib.load(os.path.join(FACTOR_PATH, 'factors.joblib'))
    returns = joblib.load(os.path.join(FACTOR_PATH, 'returns.joblib'))
    mkt_returns = returns.mean(axis=0)
    print(factors.data_vars)
    return factors, returns, mkt_returns
