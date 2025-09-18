from sklearn.ensemble import RandomForestClassifier
from .. import parent_algorithm

class Algorithm_RFC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Random Forest Classifier"

    def get_algorithm_class(self) -> type:
        return RandomForestClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "small":
            return {
                "n_estimators": (10, 40),
                "criterion": ["gini", "entropy"],
                "max_depth": (2, 15),
                "min_samples_split": (2, 5),
                "min_samples_leaf": (1, 5),
                "max_features": [None,"sqrt"],
                "bootstrap": [True],
                "class_weight": [None, "balanced"],
            }

        elif size == "medium":
            return {
                "n_estimators": (40, 100),
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": (3, 25),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 10),
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
                "class_weight": [None, "balanced"],
            }

        elif size == "large":
            return {
                "n_estimators": (75, 300),
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": (5, 50),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 15),
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": (2, 200),
                "bootstrap": [True, False],
                "class_weight": [None, "balanced"],
                "ccp_alpha": (0.0, 0.05),
            }

        elif size == "xlarge":
            return {
                "n_estimators": (200, 500),
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": (5, 80),
                "min_samples_split": (2, 50),
                "min_samples_leaf": (1, 20),
                "min_weight_fraction_leaf": (0.0, 0.1),
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": (2, 300),
                "bootstrap": [True],
                "class_weight": [None, "balanced"],
                "ccp_alpha": (0.0, 0.1),
                "max_samples": (0.5, 1.0),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'small', 'medium', 'large', or 'xlarge'.")