import numpy as np
from sklearn.linear_model import LogisticRegression

from src.basicautoml.searches.bayesian_optimization import BayesianSearchAutoML
from sklearn.ensemble import StackingClassifier

class StackingBayesianSearch:

    def __init__(self,
                algorithms: list,
                n_trials: int = 120,
                timeout: float = 60,
                scoring: str = "roc_auc",
                cv: int = 5,
                random_state=None,
                dataset_size: str = "medium",
                n_jobs: int = 1,
                verbose: bool = False):

        self.algorithms = algorithms
        self.n_trials_per_alg = n_trials
        self.timeout_per_alg = timeout
        self.scoring = scoring
        self.verbose = verbose
        self.cv = cv
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.n_jobs = n_jobs

        self.estimators = []
        self.best_model = None
        self.best_score = -float("inf")
        self.best_params = {}
        self.best_algorithm = "StackingClassifier"
        self.trained_models = 0

    def fit(self, X, y):

        self.estimators = []
        base_scores = []

        # Train each algorithm using Bayesian Search
        for alg in self.algorithms:
            bayes = BayesianSearchAutoML(
                algorithms=[alg],
                n_trials=self.n_trials_per_alg,
                timeout=self.timeout_per_alg,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
                dataset_size=self.dataset_size,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            bayes.fit(X, y)

            # Store the best model found for each algorithm
            print(f"Best model for {alg.get_name()}: {bayes.best_model} with score {bayes.best_score}")
            self.estimators.append((alg.get_name(), bayes.best_model))
            self.trained_models += bayes.trained_models
            self.best_params = {**self.best_params, **{alg.get_name(): bayes.best_params}}
            base_scores.append(bayes.best_score)

        final_estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)

        # Create and train the Stacking Classifier
        self.best_model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=final_estimator,
            cv=self.cv,
            n_jobs=self.n_jobs,
            stack_method="auto",
            passthrough=True    # Incluir o no las caracteristicas originales en el conjunto dsie entrenamiento del estimador final
        )

        print(f"Training Stacking model with {len(self.estimators)} base estimators and final estimator {final_estimator}")

        self.best_model.fit(X, y)
        self.best_score = float(np.mean(base_scores))

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet.")
        return self.best_model.predict_proba(X)

    def score(self, X, y):
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet.")
        return self.best_model.score(X, y)


