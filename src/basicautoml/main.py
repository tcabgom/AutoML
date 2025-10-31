# src/basicautoml/main.py

import numpy as np
import random

import pandas as pd
from sklearn.metrics import get_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        self.skip_preprocessing = config.skip_preprocessing

        self.feature_selector = None

        # Placeholders for models and data
        self.searcher = None
        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.X_test = None
        self.y_test = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the entire AutoML pipeline: split data, preprocess, search, select best model.
        """

        dataset_size = clasify_dataset_size(X)

        # Encode target if categorical
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            self.label_encoder_y = LabelEncoder()
            y = self.label_encoder_y.fit_transform(y)
        else:
            self.label_encoder_y = None

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

        if self.config.validation_size != 0:
            X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_size,
            random_state=self.config.random_state
        )
        else:
            X_val, y_val = None, None

        #X_train.to_csv("before_preprocessed_data.csv", index=True)

        # Preprocess
        if self.preprocessor is not None and not self.skip_preprocessing:
            X_train_prep = self.preprocessor.fit_transform(X_train)
            X_val_prep = self.preprocessor.transform(X_val) if X_val is not None else None
        else:
            X_train_prep, X_val_prep = X_train, X_val

        #X_train_prep.to_csv("preprocessed_data.csv", index=True)

        # Initialize searcher
        if self.config.search_type == 'bayesian':
            from .searches.bayesian_optimization import BayesianSearchAutoML as Searcher
        elif self.config.search_type == 'random':
            from .searches.random_search import RandomSearchAutoML as Searcher
        elif self.config.search_type == 'stacking':
            from .searches.stacking_bayesian_search import StackingBayesianSearch as Searcher
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
        self.searcher.fit(X_train_prep, y_train, X_val_prep, y_val)

        self.best_model = self.searcher.best_model
        self.best_score = self.searcher.best_score
        self.best_params = self.searcher.best_params




    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict using best model.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.preprocessor is not None:
            X_prep = self.preprocessor.transform(X)
        else:
            X_prep = X

        preds = self.best_model.predict(X_prep)

        if hasattr(self, "label_encoder_y") and self.label_encoder_y is not None:
            preds = self.label_encoder_y.inverse_transform(preds.astype(int))

        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict probabilities.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.preprocessor is not None:
            X_prep = self.preprocessor.transform(X)
        else:
            X_prep = X

        if hasattr(self.best_model, 'predict_proba'):
            proba = self.best_model.predict_proba(X_prep)
        else:
            raise AttributeError(f"Model of type {type(self.best_model)} does not support predict_proba.")

        return proba

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Score the best model on provided data.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.preprocessor is not None:
            X_prep = self.preprocessor.transform(X)
        else:
            X_prep = X

        if hasattr(self, "label_encoder_y") and self.label_encoder_y is not None:
            # Solo transformar si y contiene etiquetas originales (no numericas 0..n-1)
            if np.array_equal(np.unique(y), np.unique(self.label_encoder_y.classes_)):
                y_input = self.label_encoder_y.transform(y)
            else:
                y_input = y
        else:
            y_input = y

        scorer = get_scorer(self.config.scoring)

        return scorer(self.best_model, X_prep, y_input)
