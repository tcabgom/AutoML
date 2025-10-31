from xgboost import XGBClassifier
from .. import parent_algorithm

class Algorithm_XGBC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Extreme Gradient Boosting Classifier"

    def get_algorithm_class(self) -> type:
        return XGBClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "tiny":
            return {
                "learning_rate": (0.01, 0.3),           # Tasa de aprendizaje
                "max_depth": (2, 10),                    # Profundidad máxima
                "min_child_weight": (1, 5),             # Peso minimo de las hojas
                "gamma": (0, 0.3),                      # Penalización por partición
                "subsample": (0.7, 1.0),                # Submuestreo de las filas
                "colsample_bytree": (0.7, 1.0),         # Submuestreo de las columnas
                "n_estimators": (5, 20),                # Numero de arboles
            }

        elif size == "small":
            return {
                "learning_rate": (0.01, 0.25),
                "max_depth": (3, 15),
                "min_child_weight": (1, 7),
                "gamma": (0, 0.4),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "n_estimators": (10, 40),
            }

        elif size == "medium":
            return {
                "learning_rate": (0.0075, 0.2),
                "max_depth": (3, 25),
                "min_child_weight": (1, 10),
                "gamma": (0, 0.5),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "n_estimators": (20, 80),
            }

        elif size == "large":
            return {
                "learning_rate": (0.005, 0.15),
                "max_depth": (5, 40),
                "min_child_weight": (1, 15),
                "gamma": (0, 0.6),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0),
                "n_estimators": (40, 160),
            }

        elif size == "xlarge":
            return {
                "learning_rate": (0.001, 0.1),
                "max_depth": (10, 60),
                "min_child_weight": (1, 20),
                "gamma": (0, 1.0),
                "subsample": (0.4, 1.0),
                "colsample_bytree": (0.4, 1.0),
                "n_estimators": (80, 320),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'tiny', 'small', 'medium', 'large', or 'xlarge'.")
