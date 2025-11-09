import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_predict

from src.basicautoml.algorithms.classification import LogisticRegression

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
                dataset_meta_data: dict() = None,
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
        self.dataset_meta_data = dataset_meta_data

        self.estimators = []
        self.best_model = None
        self.best_score = -float("inf")
        self.best_params = {}
        self.best_algorithm = "StackingClassifier"
        self.trained_models = 0

    def fit(self, X_data: pd.DataFrame, y_data: pd.DataFrame, x_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> None:

        self.estimators = []
        scores = {}
        models = {}
        params = {}

        # 1. Train each algorithm using Bayesian Search
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
                dataset_meta_data=self.dataset_meta_data,
                verbose=self.verbose
            )
            bayes.fit(X_data, y_data, x_val, y_val)

            # Store the best model found for each algorithm
            print(f" ! Best score for {alg.get_name()}: {bayes.best_score}\n")
            self.estimators.append((alg.get_name(), bayes.best_model))
            self.trained_models += bayes.trained_models

            scores[alg.get_name()] = bayes.best_score
            models[alg.get_name()] = bayes.best_model
            params[alg.get_name()] = bayes.best_params

        # 2. Generate meta-features to train the meta-model with bayesian optimization
        print("Generating meta-features dataset for stacking...\n")
        meta_features = np.zeros((len(X_data), len(self.estimators)))
        for i, (name, model) in enumerate(self.estimators):
            meta_features[:, i] = cross_val_predict(model, X_data, y_data, cv=self.cv, method="predict_proba", n_jobs=1)[:, 1]

        meta_X = pd.DataFrame(meta_features, columns=[name for name, _ in self.estimators])
        meta_y = y_data

        print("Meta-features dataset generated. Beggining training of stacking model\n")

        # 3. Train the meta-model using Bayesian Search
        meta_bayes = BayesianSearchAutoML(
            algorithms=[LogisticRegression.Algorithm_LR()],
            n_trials=self.n_trials_per_alg,
            timeout=self.timeout_per_alg,
            scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            dataset_size=self.dataset_size,
            dataset_meta_data=self.dataset_meta_data,
            verbose=self.verbose,
        )
        self.trained_models += meta_bayes.trained_models

        meta_bayes.fit(meta_X, meta_y)

        # 4. Retrieve the best meta-model parameters
        best_stacking_params = meta_bayes.best_params
        final_estimator_class = LogisticRegression.Algorithm_LR().get_algorithm_class()
        final_estimator = final_estimator_class(**best_stacking_params)

        # 5. Train an actual StackingClassifier model with the best meta-model parameters
        stacking_model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=final_estimator,
            cv=self.cv,
            n_jobs=self.n_jobs,
            stack_method="auto",
            passthrough=True    # Incluir o no las caracteristicas originales en el conjunto dsie entrenamiento del estimador final
        )


        print(f"\n ! Training Final Stacking Model...")
        stacking_model.fit(X_data, y_data)
        scorer = get_scorer(self.scoring)
        stacking_score = scorer(stacking_model, x_val, y_val)
        print(f" ! Stacking Model score: {stacking_score}")

        scores["StackingClassifier"] = stacking_score
        models["StackingClassifier"] = stacking_model
        params["StackingClassifier"] = best_stacking_params

        # 6. Retrieve the best model, not necessarily the stacking one
        best_model_name = max(scores, key=scores.get)
        self.best_model = models[best_model_name]
        self.best_params = params[best_model_name]
        self.best_score = scores[best_model_name]
        self.best_algorithm = best_model_name
        print(f" ! Best model overall: {best_model_name} with score {scores[best_model_name]}\n")

        #self.best_score = float(np.mean(base_scores))

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


