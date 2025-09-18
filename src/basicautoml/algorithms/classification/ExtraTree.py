from sklearn.tree import ExtraTreeClassifier
from .. import parent_algorithm

class Algorithm_ETC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Extra Tree Classifier"

    def get_algorithm_class(self) -> type:
        return ExtraTreeClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "small":
            return {
                "criterion": ["gini", "entropy"],
                "max_depth": (2, 15),
                "min_samples_split": (2, 5),
                "min_samples_leaf": (1, 5),
                "max_features": [None, "sqrt"],
                "class_weight": [None, "balanced"],
            }

        elif size == "medium":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": (3, 25),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 10),
                "max_features": ["sqrt", "log2", None],
                "max_leaf_nodes": (2, 50),
                "class_weight": [None, "balanced"],
            }

        elif size == "large":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": (1, 40),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 15),
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": (2, 150),
                "min_impurity_decrease": (0.0, 0.01),
                "class_weight": [None, "balanced"],
                "ccp_alpha": (0.0, 0.05),
            }

        elif size == "xlarge":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": (1, 60),
                "min_samples_split": (2, 50),
                "min_samples_leaf": (1, 20),
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": (2, 200),
                "min_impurity_decrease": (0.0, 0.02),
                "min_weight_fraction_leaf": (0.0, 0.1),
                "class_weight": [None, "balanced"],
                "ccp_alpha": (0.0, 0.1),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'small', 'medium', 'large', or 'xlarge'.")