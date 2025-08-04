import numpy as np
import random

import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from BasicAutoML.src.config import AutoMLConfig


class TFM_AutoML:
    def __init__(self, config: AutoMLConfig):
        self.config = config

        # Initialize reproducibility
        np.random.seed(config.random_state)
        random.seed(config.random_state)

        # Create preprocessor
        from BasicAutoML.src.preprocessing import Preprocessor
        self.preprocessor = Preprocessor(**config.preprocessor_params)

        # Placeholders for models and data
        self.searcher = None
        self.best_model = None
        self.best_score = None
        self.X_test = None
        self.y_test = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the entire AutoML pipeline: split data, preprocess, search, select best model.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        self.X_test, self.y_test = X_test, y_test

        # Preprocess
        X_train_prep = self.preprocessor.fit_transform(X_train)
        X_test_prep = self.preprocessor.transform(X_test)

        # Initialize searcher
        if self.config.search_type == 'bayesian':
            from BasicAutoML.src.searches.bayesian_optimization import BayesianSearchAutoML as Searcher
        elif self.config.search_type == 'random':
            from BasicAutoML.src.searches.random_search import RandomSearchAutoML as Searcher
        else:
            raise ValueError(f"Unsupported search type: {self.config.search_type}")

        self.searcher = Searcher(
            algorithms=self.config.algorithms,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            scoring=self.config.scoring,
            cv=self.config.cv,
            verbose=self.config.verbose,
            random_state=self.config.random_state
        )
        # Run search
        self.searcher.fit(X_train_prep, y_train)

        # Retrieve best
        self.best_model = self.searcher.best_model
        self.best_score = self.searcher.best_score

        # Refit best model on full training
        self.best_model.fit(
            np.vstack((X_train_prep, X_test_prep)),
            pd.concat([y_train, y_test])
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict using best model.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_prep = self.preprocessor.transform(X)
        return self.best_model.predict(X_prep)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict probabilities.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
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
        X_prep = self.preprocessor.transform(X)
        scorer = get_scorer(self.config.scoring)
        return scorer(self.best_model, X_prep, y)