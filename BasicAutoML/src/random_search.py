import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer


class RandomSearchAutoML:
    def __init__(self,
                 model_class: type,
                 param_distributions: dict,
                 n_iter: int = 100,
                 scoring: str = "accuracy",
                 cv: int = 5,
                 verbose: bool = False,
                 random_state=None):

        self.model_class = model_class,
        self.param_distributions = param_distributions
        self.n_iter = n_iter,
        self.scoring = scoring,
        self.verbose = verbose,
        self.random_state = random_state,
        self.results = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    def __sample_params(self) -> dict:
        return {k: np.random.choice(v) for k, v in self.param_distributions.items()}

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        scorer = get_scorer(self.scoring)

        for i in range(self.n_iter):
            params = self.__sample_params()
            model = self.model_class(**params)

            scores = cross_val_score(model, x_data, y_data, cv=self.cv, scoring=scorer)
            mean_score = np.mean(scores)

            self.results.append({
                "params": params,
                "score": mean_score
            })

            if self.verbose:
                print(f"Iter {i + 1}/{self.n_iter} | Score: {mean_score:.4f} | Params: {params}")

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_model_ = model

        self.best_model_.fit(x_data, y_data)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        if self.best_model_ is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        return self.best_model_.predict(x_data)

    def score(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        if self.best_model_ is None:
            raise RuntimeError("The model has not been trained yet. Call `fit` first.")
        scorer = get_scorer(self.scoring)
        return scorer(self.best_model_, x_data, y_data)
