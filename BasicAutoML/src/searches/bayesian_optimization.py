import numpy as np
import optuna
import random

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer


class BayesianSearchAutoML:
    def __init__(self,
                 algorithms: list,
                 n_trials: int = 120,
                 timeout: float = 60,
                 scoring: str = "roc_auc",
                 cv: int = 5,
                 verbose: bool = False,
                 random_state=None):

        self.algorithms = algorithms
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring = scoring
        self.verbose = verbose
        self.cv = cv
        self.random_state = random_state

        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.best_model_class = None
        self.best_algorithm = None
        self.study = None

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def __objective(self, trial: optuna.Trial, x_data: pd.DataFrame, y_data: pd.DataFrame) -> float:
        algo_choices = {algo.get_name(): algo for algo in self.algorithms}
        algo_name = trial.suggest_categorical("algorithm", list(algo_choices.keys()))
        algo_obj = algo_choices[algo_name]

        model_class = algo_obj.get_algorithm_class()
        param_space = algo_obj.get_algorithm_params()

        params = {}
        for param_name, distribution in param_space.items():
            full_name = f"{algo_name}__{param_name}"  # avoid collision in Optuna param names
            if isinstance(distribution, list):
                params[param_name] = trial.suggest_categorical(full_name, distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                low, high = distribution
                if all(isinstance(v, int) for v in (low, high)):
                    params[param_name] = trial.suggest_int(full_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(full_name, low, high)
            else:
                raise ValueError(f"Unsupported distribution type for param: {param_name}")

        model = model_class(**params)
        scorer = get_scorer(self.scoring)
        scores = cross_val_score(model, x_data, y_data, cv=self.cv, scoring=scorer)
        mean_score = float(np.mean(scores))

        if self.verbose:
            print(
                f"Trial {trial.number + 1} | Model: {model_class.__name__} | Score: {mean_score:.4f} | Params: {params}")

        trial.set_user_attr("algo_obj", algo_obj)

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
        self.best_algorithm = self.study.best_trial.user_attrs["algo_obj"]
        self.best_model_class = self.best_algorithm.get_algorithm_class()

        # Extract the parameters for the best algorithm
        prefix = self.best_algorithm.get_name() + "__"
        model_params = { k.replace(prefix, ""): v for k, v in self.best_params.items() if k.startswith(prefix) }

        self.best_model = self.best_model_class(**model_params)
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
