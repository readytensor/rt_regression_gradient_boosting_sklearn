import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the Gradient Boosting Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    MODEL_NAME = "Gradient Boosting Regressor"

    def __init__(
        self,
        n_estimators: Optional[int] = 300,
        learning_rate: Optional[float] = 0.2,
        min_samples_split: Optional[int] = 8,
        min_samples_leaf: Optional[int] = 4,
        **kwargs,
    ):
        """Construct a new Gradient Boosting Regressor.

        Args:
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 250.
            learning_rate (float, optional): Learning rate shrinks the contribution
                of each tree by learning_rate. There is a trade-off between 
                learning_rate and n_estimators.
                Defaults to 0.1.
            min_samples_split (int, optional): The minimum number of samples required
                to split an internal node.
                Defaults to 4.
            min_samples_leaf (int, optional): The minimum number of samples required
                to be at a leaf node.
                Defaults to 2.
        """
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> GradientBoostingRegressor:
        """Build a new regressor."""
        model = GradientBoostingRegressor(
            loss="huber",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The targets of the test data.
        Returns:
            float: The r-squared score of the regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.MODEL_NAME} ("
            f"n_estimators: {self.n_estimators}, "
            f"learning_rate: {self.learning_rate}, "
            f"min_samples_split: {self.min_samples_split}, "
            f"min_samples_leaf: {self.min_samples_leaf})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
