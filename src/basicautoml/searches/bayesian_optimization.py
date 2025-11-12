import numpy as np
import optuna
import random

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import get_scorer

from src.basicautoml.meta_learning import find_nearest_datasets


class BayesianSearchAutoML:
    def __init__(self,
                 algorithms: list,
                 n_trials: int = 120,
                 timeout: float = 60,
                 scoring: str = "roc_auc",
                 cv: int = 5,
                 random_state=None,
                 dataset_size: str = "medium",
                 n_jobs: int = 1,
                 dataset_meta_data: dict() = None,
                 verbose: bool = False):

        self.algorithms = algorithms
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring = scoring
        self.verbose = verbose
        self.cv = cv
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.n_jobs = n_jobs
        self.dataset_meta_data = dataset_meta_data

        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.best_model_class = None
        self.best_algorithm = None
        self.study = None
        self.trained_models = 0

        # Semillas para reproducibilidad
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Evita que los logs de Optuna saturen la salida
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def __objective(self, trial: optuna.Trial, x_data: pd.DataFrame, y_data: pd.Series) -> float:
        """
        Objective function for Optuna optimization.

        :param trial: Current Optuna trial object.
        :param x_data: DataFrame containing the features for training.
        :param y_data: DataFrame containing the target variable for training.

        :return: Score value for the current trial.
        """

        self.trained_models += 1

        # Choose model based on trial suggestions
        algo_choices = {algo.get_name(): algo for algo in self.algorithms}
        algo_name = trial.suggest_categorical("algorithm", list(algo_choices.keys()))
        algo_obj = algo_choices[algo_name]

        # Get model class and parameter space
        model_class = algo_obj.get_algorithm_class()
        param_space = algo_obj.get_algorithm_params(size=self.dataset_size)

        # Suggest parameters based on their distributions
        params = {}
        for param_name, distribution in param_space.items():
            full_name = f"{algo_name}__{param_name}"                                    # Avoid collision in Optuna param names
            if isinstance(distribution, list):                                          # Categorical values
                params[param_name] = trial.suggest_categorical(full_name, distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:            # Min/max values
                low, high = distribution
                if all(isinstance(v, int) for v in (low, high)):                        # Integer min/max values
                    params[param_name] = trial.suggest_int(full_name, low, high)
                else:                                                                   # Float min/max values
                    params[param_name] = trial.suggest_float(full_name, low, high)
            else:
                raise ValueError(f"Unsupported distribution type for param: {param_name}")

        # Create model and scorer
        model = model_class(**params)
        scorer = get_scorer(self.scoring)

        # Perform cross-validation and compute mean score
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, x_data, y_data, cv=cv, scoring=scorer)
        mean_score = float(np.mean(scores))

        if self.verbose:
            print(
                f"Trial {trial.number + 1} | Model: {model_class.__name__} | Score: {mean_score:.4f} | Params: {params}")

        # Store the algorithm in the trial for later retrieval
        trial.set_user_attr("algorithm_name", algo_obj.get_name())
        return mean_score

    def fit(self, x_data: pd.DataFrame, y_data: pd.DataFrame,
            x_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> None:
        """
        Fit the AutoML model using Bayesian optimization.

        :param x_data: DataFrame containing the features for training.
        :param y_data: DataFrame containing the target variable for training.
        """
        # Comprobar si existe estudio antes de borrarlo
        storage_url = "sqlite:///automl_bayesian_study.db"
        study_name = "Bayesian_AutoML_Study"

        # Obtener todos los estudios almacenados
        existing_studies = [s.study_name for s in optuna.get_all_study_summaries(storage=storage_url)]

        if study_name in existing_studies:
            optuna.delete_study(study_name=study_name, storage=storage_url)

        # Create the Optuna study with TPE sampler
        self.study = optuna.create_study(
            study_name="Bayesian_AutoML_Study",
            storage="sqlite:///automl_bayesian_study.db",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        # Enqueue initial points if provided for meta learning warm start
        if self.dataset_meta_data is not None:
            initial_points = find_nearest_datasets(
                query_meta_features=self.dataset_meta_data,
                n=5,
                dataset_size=self.dataset_size,
                allowed_algorithms=[a.get_algorithm_class().__name__ for a in self.algorithms]
            )
            if len(initial_points) > 0:
                print(f"Enqueuing {len(initial_points)} initial points from meta-learning warm start.")
            for point in initial_points:
                algo_name = point["algorithm"]
                hyperparams = point["hyperparameters"]
                formatted_params = {"algorithm": algo_name}
                for param, value in hyperparams.items():
                    formatted_params[f"{algo_name}__{param}"] = value
                self.study.enqueue_trial(formatted_params)

        # Optimize the study using the objective function
        self.study.optimize(
            lambda trial: self.__objective(trial, x_data, y_data),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )

        # Retrieve the best trial information
        self.best_score = self.study.best_value
        if self.verbose:
            print(f"\n ! Best trial score: {self.best_score:.4f}")
        algo_name = self.study.best_trial.user_attrs["algorithm_name"]
        self.best_algorithm = next(a for a in self.algorithms if a.get_name() == algo_name)
        self.best_model_class = self.best_algorithm.get_algorithm_class()

        # Extract the parameters for the best algorithm
        prefix = self.best_algorithm.get_name() + "__"
        model_params = { k.replace(prefix, ""): v for k, v in self.study.best_params.items() if k.startswith(prefix) }
        self.best_params = model_params

        # Train the best model on the entire dataset
        self.best_model = self.best_model_class(**model_params)
        self.best_model.fit(x_data, y_data)

        if x_val is not None and y_val is not None:
            scorer = get_scorer(self.scoring)
            val_score = scorer(self.best_model, x_val, y_val)
            self.best_score = val_score
            if self.verbose:
                print(f" ! Validation score of the best model: {val_score:.4f}\n")


    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """
        Predict using the best trained model.

        :param x_data: DataFrame containing the features for prediction.
        :return: Predicted values as a NumPy array.
        """
        if self.best_model is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        return self.best_model.predict(x_data)


    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the best trained model.

        :param x_data: DataFrame or ndarray containing the features for prediction.
        :return: Predicted probabilities as a NumPy array.
        """
        if self.best_model is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(x_data)
        else:
            raise AttributeError(f"The model of type {type(self.best_model)} does not support `predict_proba`.")


    def score(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """
        Evaluates the best trained model using the specified scoring metric.

        :param x_data: Ndarray or DataFrame containing the features for evaluation.
        :param y_data: Ndarray or DataFrame containing the true target variable for evaluation.

        :return: Score value as a float.
        """
        if self.best_model is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        scorer = get_scorer(self.scoring)
        return scorer(self.best_model, x_data, y_data)
