from sklearn.ensemble import RandomForestClassifier
from .. import parent_algorithm

class Algorithm_RFC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Random Forest Classifier"

    def get_algorithm_class(self) -> type:
        return RandomForestClassifier

    def get_algorithm_params(self) -> dict:
        return {
            "n_estimators": (10, 100),
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": (3, 30),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 10),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced"],
        }