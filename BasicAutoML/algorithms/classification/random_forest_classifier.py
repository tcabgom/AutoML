from .. import parent_algorithm
from sklearn.ensemble import RandomForestClassifier

HYPERPARAMETER_LIMITS = {
    "n_estimators": [3, 19],
    "max_depth": [2, 20],
    "min_samples_split": [2, 25],
    "min_samples_leaf": [1, 20],
    "min_impurity_decrease": [0, 0.01]
}


class RFC_Algorithm (parent_algorithm.ParentAlgorithm):

    def __init__(self):

        self.model = None

        self.n_estimators = None
        self.max_depth = None
        self.min_samples_split = None
        self.min_samples_leaf = None
        self.min_impurity_decrease = None


    def get_hyperparameter_limits(self):
        return HYPERPARAMETER_LIMITS


    def load_hyperparameters(self, hyperparameters):
        self.n_estimators = round(hyperparameters["n_estimators"])
        self.max_depth = round(hyperparameters["max_depth"])
        self.min_samples_split = round(hyperparameters["min_samples_split"])
        self.min_samples_leaf = round(hyperparameters["min_samples_leaf"])
        self.min_impurity_decrease = hyperparameters["min_impurity_decrease"]


    def fit(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                min_impurity_decrease=self.min_impurity_decrease)
        return self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)


    def get_hyperparameters_map(self):
        hyperparameters_map = {"n_estimators": self.n_estimators,
                                "max_depth": self.max_depth,
                                "min_samples_split": self.min_samples_split,
                                "min_samples_leaf": self.min_samples_leaf,
                                "min_impurity_decrease": self.min_impurity_decrease}
        return hyperparameters_map
