import numpy as np
import optuna
import random

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer


class BayesianSearchUnimodelAutoML:
    def __init__(self,
                 model_class: type,
                 param_distributions: dict,
                 n_trials: int = 120,
                 timeout: float = 60,
                 scoring: str = "roc_auc",
                 cv: int = 5,
                 verbose: bool = False,
                 random_state=None):

        self.model_class = model_class
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring = scoring
        self.verbose = verbose
        self.cv = cv
        self.random_state = random_state

        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.study = None

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        optuna.logging.set_verbosity(optuna.logging.WARNING)


    def __objective(self, trial: optuna.Trial, x_data: pd.DataFrame, y_data: pd.DataFrame) -> float:
        params = {}
        for param_name, distribution in self.param_distributions.items():
            if isinstance(distribution, list):  # Categorical
                params[param_name] = trial.suggest_categorical(param_name, distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                low, high = distribution
                if all(isinstance(v, int) for v in (low, high)):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            else:
                raise ValueError(f"Unsupported distribution type for param: {param_name}")

        model = self.model_class(**params)
        scorer = get_scorer(self.scoring)

        scores = cross_val_score(model, x_data, y_data, cv=self.cv, scoring=scorer)
        mean_score = float(np.mean(scores))

        if self.verbose:
            print(f"Trial {trial.number + 1} | Score: {mean_score:.4f} | Params: {params}")

        return mean_score


    def fit(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> None:
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        self.study.optimize(
            lambda trial: self.__objective(trial, x_data, y_data),
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(x_data, y_data)


    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        return self.best_model.predict(x_data)


    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(x_data)
        else:
            raise AttributeError(f"The model of type {type(self.best_model)} does not support `predict_proba`.")


    def score(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        if self.best_model is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        scorer = get_scorer(self.scoring)
        return scorer(self.best_model, x_data, y_data)
