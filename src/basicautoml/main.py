# src/basicautoml/main.py

import numpy as np
import random

import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from .config import AutoMLConfig
from .utils.dataset_size import clasify_dataset_size


class TFM_AutoML:
    def __init__(self, config: AutoMLConfig):
        self.config = config

        # Initialize reproducibility
        np.random.seed(config.random_state)
        random.seed(config.random_state)

        # Create preprocessor
        # TODO Maybe allow to skip preprocessing?
        from .preprocessing import Preprocessor
        self.preprocessor = Preprocessor(**config.preprocessor_params)

        self.feature_selector = None

        # Placeholders for models and data
        self.searcher = None
        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.X_test = None
        self.y_test = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # TODO Add option to skip train-test split
        """
        Fit the entire AutoML pipeline: split data, preprocess, search, select best model.
        """

        dataset_size = clasify_dataset_size(X)

        # Split data
        if self.config.test_size != 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            self.X_test, self.y_test = X_test, y_test
        else:
            X_train, y_train = X, y
            self.X_test, self.y_test = None, None

        #X_train.to_csv("before_preprocessed_data.csv", index=True)

        # Preprocess
        if self.preprocessor is not None:
            X_train_prep = self.preprocessor.fit_transform(X_train)

        #X_train_prep.to_csv("preprocessed_data.csv", index=True)

        # Initialize searcher
        if self.config.search_type == 'bayesian':
            from .searches.bayesian_optimization import BayesianSearchAutoML as Searcher
        elif self.config.search_type == 'random':
            from .searches.random_search import RandomSearchAutoML as Searcher
        else:
            raise ValueError(f"Unsupported search type: {self.config.search_type}")

        self.searcher = Searcher(
            algorithms=self.config.algorithms,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            scoring=self.config.scoring,
            cv=self.config.cv,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            dataset_size=dataset_size
        )
        # Run search
        self.searcher.fit(X_train_prep, y_train)

        self.best_model = self.searcher.best_model
        self.best_score = self.searcher.best_score
        self.best_params = self.searcher.best_params

        # TODO Implement final training with best model
        """
        # Retrieve best configuration
        best_params = self.searcher.best_params
        best_estimator_class = type(self.searcher.best_model)

        # Fit model with best configuration
        self.best_model = best_estimator_class(**best_params)
        self.best_model.fit(X_train_prep, y_train)
        """



    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict using best model.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.preprocessor is not None:
            X_prep = self.preprocessor.transform(X)
        return self.best_model.predict(X_prep)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict probabilities.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.preprocessor is not None:
            X_prep = self.preprocessor.transform(X)
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X_prep)
        else:
            raise AttributeError(f"Model of type {type(self.best_model)} does not support predict_proba.")

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Score the best model on provided data.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.preprocessor is not None:
            X_prep = self.preprocessor.transform(X)
        scorer = get_scorer(self.config.scoring)
        return scorer(self.best_model, X_prep, y)
