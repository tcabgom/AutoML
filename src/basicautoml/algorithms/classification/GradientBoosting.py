from sklearn.ensemble import GradientBoostingClassifier
from .. import parent_algorithm

class Algorithm_XGBC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Gradient Boosting Classifier"

    def get_algorithm_class(self) -> type:
        return GradientBoostingClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "small":
            return {
                "loss": ["log_loss"],
                "learning_rate": (0.05, 0.2),
                "n_estimators": (10, 40),
                "min_samples_split": (2, 5),
                "min_samples_leaf": (1, 5),
                "max_depth": (2, 15),
                "max_features": [None, "sqrt"]
            }

        elif size == "medium":
            return {
                "loss": ["log_loss"],
                "learning_rate": (0.01, 0.2),
                "n_estimators": (40, 100),
                "subsample": (0.7, 1.0),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 10),
                "max_depth": (3, 25),
                "max_features": ["sqrt", "log2", None],
            }

        elif size == "large":
            return {
                "loss": ["log_loss"],
                "learning_rate": (0.01, 0.3),
                "n_estimators": (100, 300),
                "subsample": (0.5, 1.0),
                "criterion": ["friedman_mse", "squared_error"],
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 15),
                "max_depth": (3, 12),
                "max_features": [None, "sqrt", "log2"],
            }

        elif size == "xlarge":
            return {
                "loss": ["log_loss"],
                "learning_rate": (0.005, 0.3),
                "n_estimators": (200, 500),
                "subsample": (0.5, 1.0),
                "criterion": ["friedman_mse", "squared_error"],
                "min_samples_split": (2, 30),
                "min_samples_leaf": (1, 20),
                "max_depth": (3, 15),
                "max_features": [None, "sqrt", "log2"],
                "min_impurity_decrease": (0.0, 0.01),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'small', 'medium', 'large', or 'xlarge'.")