import numpy as np
import xarray as xr


class ComputeEngine:

    def __init__(self):
        pass

    # Element-wise operations
    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def sin(x):
        return np.sin(x)

    @staticmethod
    def cos(x):
        return np.cos(x)

    # Pair-wise operations
    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def subtract(x, y):
        return x - y

    @staticmethod
    def multiply(x, y):
        return x * y

    @staticmethod
    def divide(x, y):
        return x / y

    @staticmethod
    def power(x, y):
        return x**y

    # MultiElement operations
    @staticmethod
    def max(xs):
        return np.max(xs)

    @staticmethod
    def min(xs):
        return np.min(xs)

    @staticmethod
    def mean(xs):
        return np.mean(xs)

    @staticmethod
    def std(xs):
        return np.std(xs)

    # Xarray operations
    @staticmethod
    def shift(x, shift_col, shift_num):
        return x.shift({shift_col: shift_num})

    @staticmethod
    def rolling_max(x, roll_col, roll_window):
        return x.rolling({roll_col: roll_window}).max()

    @staticmethod
    def rolling_min(x, roll_col, roll_window):
        return x.rolling({roll_col: roll_window}).min()

    @staticmethod
    def rolling_mean(x, roll_col, roll_window):
        return x.rolling({roll_col: roll_window}).mean()

    @staticmethod
    def rolling_std(x, roll_col, roll_window):
        return x.rolling({roll_col: roll_window}).std()

    # Group operations
    @staticmethod
    def rank(x, rank_col):
        return x.rank(rank_col)

    @staticmethod
    def scale(x, scale_col):
        return x / x.sum(scale_col)

    # Complex operations
    @classmethod
    def ret(cls, x, shift_col, shift_num):
        return cls.log(x) - cls.log(cls.shift(x, shift_col, shift_num))

    @classmethod
    def rolling_corr(cls, x, y, roll_col, roll_window):
        x = x.rolling({roll_col: roll_window}).construct("window_dim")
        y = y.rolling({roll_col: roll_window}).construct("window_dim")
        return xr.corr(x, y, "window_dim")
