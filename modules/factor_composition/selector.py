from abc import ABC, abstractmethod

import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class BaseSelector(ABC):
    """
    Abstract base class for feature selection.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select(self, **kwargs):
        pass


class LassoSelector(BaseSelector):
    """
    Feature selector using Lasso regression.
    """

    eps = 1e-8

    def __init__(self, scorer, k):
        """
        Initialize the LassoSelector.

        Args:
            scorer: Scoring function used for feature selection.
            k: Number of folds for cross-validation.
        """
        self.scorer = scorer
        self.k = k
        self.fitted_result = None

    def select(self, train_dataset):
        """
        Select features using Lasso regression.

        Args:
            train_dataset: Training dataset.

        Returns:
            Boolean array indicating selected features.
        """
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('model', LassoCV(tol=self.eps, n_jobs=-1,
                                           cv=self.k))])

        pipe.fit(train_dataset.drop(columns=['RETURN']),
                 train_dataset['RETURN'])

        self.fitted_result = pipe

        return np.abs(pipe.get_params()["model"].coef_) > self.eps
