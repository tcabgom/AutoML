from sklearn.ensemble import HistGradientBoostingClassifier
from .. import parent_algorithm

class Algorithm_HistGBC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Histogram Gradient Boosting Classifier"

    def get_algorithm_class(self) -> type:
        return HistGradientBoostingClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "tiny":
            return {
                "learning_rate": (0.01, 0.3),          # Tasa de aprendizaje
                "max_iter": (5, 20),                   # Numero de arboles
                "max_leaf_nodes": (5, 31),             # Hojas maximas por arbol
                "min_samples_leaf": (1, 10),           # Min samples por hoja
                "l2_regularization": (0.0, 0.1),       # Regularizacion L2
                "max_features": (0.5, 1.0),            # Fraccion de features en cada split
            }

        elif size == "small":
            return {
                "learning_rate": (0.01, 0.2),
                "max_iter": (10, 40),
                "max_leaf_nodes": (7, 63),
                "min_samples_leaf": (1, 20),
                "l2_regularization": (0.0, 0.5),
                "max_features": (0.3, 1.0),
            }

        elif size == "medium":
            return {
                "learning_rate": (0.0075, 0.15),
                "max_iter": (20, 80),
                "max_leaf_nodes": (15, 127),
                "min_samples_leaf": (1, 30),
                "l2_regularization": (0.0, 1.0),
                "max_features": (0.1, 1.0),
            }

        elif size == "large":
            return {
                "learning_rate": (0.001, 0.1),
                "max_iter": (40, 160),
                "max_leaf_nodes": (31, 255),
                "min_samples_leaf": (1, 50),
                "l2_regularization": (0.0, 2.0),
                "max_features": (0.05, 1.0),
            }

        elif size == "xlarge":
            return {
                "learning_rate": (0.0005, 0.05),
                "max_iter": (80, 320),
                "max_leaf_nodes": (31, 1023),
                "min_samples_leaf": (1, 100),
                "l2_regularization": (0.0, 5.0),
                "max_features": (0.01, 1.0),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'tiny', 'small', 'medium', 'large', or 'xlarge'.")
