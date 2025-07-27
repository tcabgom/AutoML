from .. import parent_algorithm
from sklearn.neighbors import KNeighborsClassifier


HYPERPARAMETER_LIMITS = {
    "n_neighbors": [3, 15],
    "weights": [0,1],
    "algorithm": [-0.49,2.49],
    "leaf_size": [10, 50],
    "p": [1, 2]
}

class KNC_Algorithm (parent_algorithm.ParentAlgorithm):

    def __init__(self):

        self.model = None

        self.n_neighbors = None
        self.weights = None
        self.algorithm = None
        self.leaf_size = None


    def get_hyperparameter_limits(self):
        return HYPERPARAMETER_LIMITS


    def load_hyperparameters(self, hyperparameters):
        self.n_neighbors = round(hyperparameters["n_neighbors"])
        self.weights = ["uniform", "distance"][round(hyperparameters["weights"])]
        self.algorithm = ["ball_tree", "kd_tree", "brute"][round(hyperparameters["algorithm"])]
        self.leaf_size = round(hyperparameters["leaf_size"])
        self.p = hyperparameters["p"]


    def fit(self, X_train, y_train):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                weights=self.weights,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size,
                                p=self.p)
        return self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)


    def get_hyperparameters_map(self):
        hyperparameters_map = {"n_neighbors": self.n_neighbors,
                                "weights": self.weights,
                                "algorithm": self.algorithm,
                                "leaf_size": self.leaf_size,
                                "p": self.p}
        return hyperparameters_map