from sklearn.tree import ExtraTreeClassifier
from .. import parent_algorithm

class Algorithm_ETC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Extra Tree Classifier"

    def get_algorithm_class(self) -> type:
        return ExtraTreeClassifier

    def get_algorithm_params(self) -> dict:
        return {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": (1, 25),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 10),
            "max_features": ["sqrt", "log2", None],
            "max_leaf_nodes": (2, 100),
            "class_weight": [None, "balanced"]
        }
