from sklearn.ensemble import GradientBoostingClassifier
from .. import parent_algorithm

class Algorithm_XGBC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Gradient Boosting Classifier"

    def get_algorithm_class(self) -> type:
        return GradientBoostingClassifier

    def get_algorithm_params(self) -> dict:
        return {
            "loss": ["log_loss"],#, "exponential"],
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 200),
            "subsample": (0.5, 1.0),
            "criterion": ["friedman_mse", "squared_error"],
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 10),
            "max_depth": (3, 10),
            "max_features": ["sqrt", "log2", None]
        }
