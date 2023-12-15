from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class BaseRegressor(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class LassoRegressor(BaseRegressor):
    """
    LassoRegressor is a class that implements a Lasso regression model for prediction.

    Parameters:
    - scorer: The scoring function used for model evaluation.
    - k: The number of folds for cross-validation.

    Attributes:
    - eps: A small value used for numerical stability.
    - estimator: The fitted Lasso regression model.

    Methods:
    - fit(train_dataset): Fit the Lasso regression model to the training dataset.
    - predict(test_dataset): Make predictions using the fitted Lasso regression model on the test dataset.
    """

    eps = 1e-6

    def __init__(self, scorer, k):
        # Initialize the LassoRegressor object
        self.scorer = scorer
        self.k = k
        self.estimator = None

    def fit(self, train_dataset):
        """
        Fit the Lasso regression model to the training dataset.

        Parameters:
        - train_dataset: The training dataset containing features and target variable.

        Returns:
        None
        """
        # Create a pipeline with feature scaling and Lasso regression model
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('model', LassoCV(tol=self.eps, n_jobs=-1,
                                           cv=self.k))])

        # Perform grid search with cross-validation
        pipe.fit(train_dataset.drop(columns=['RETURN']),
                 train_dataset['RETURN'])

        # Save the fitted model
        self.estimator = pipe

    def predict(self, test_dataset):
        """
        Make predictions using the fitted Lasso regression model on the test dataset.

        Parameters:
        - test_dataset: The test dataset containing features.

        Returns:
        - predictions: The predicted target variable values.
        """
        return self.estimator.predict(test_dataset.drop(columns=['RETURN']))
