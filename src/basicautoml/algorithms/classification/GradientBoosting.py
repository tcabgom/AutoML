from sklearn.ensemble import GradientBoostingClassifier
from .. import parent_algorithm

class Algorithm_GBC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Gradient Boosting Classifier"

    def get_algorithm_class(self) -> type:
        return GradientBoostingClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "tiny":
            return {
                "learning_rate": (0.01, 0.3),                   # Tasa de aprendizaje
                "max_depth": (2, 10),                           # Profundidad maxima del arbol
                "min_samples_split": (2, 5),                    # Minimo de muestras para dividir un nodo
                "min_samples_leaf": (1, 5),                     # Minimo de muestras en un nodo hoja
                "n_estimators": (5, 20),                        # Numero de arboles en el bosque
                "subsample": (0.7, 1.0),                        # Submuestreo de las muestras
            }
        elif size == "small":
            return {
                "learning_rate": (0.01, 0.25),
                "max_depth": (3, 15),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 5),
                "n_estimators": (10, 40),
                "subsample": (0.7, 1.0),
            }

        elif size == "medium":
            return {
                "learning_rate": (0.0075, 0.2),
                "max_depth": (3, 25),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
                "n_estimators": (20, 80),
                "subsample": (0.7, 1.0),
            }

        elif size == "large":
            return {
                "learning_rate": (0.005, 0.2),
                "max_depth": (5, 40),
                "min_samples_split": (2, 30),
                "min_samples_leaf": (1, 15),
                "n_estimators": (40, 160),
                "subsample": (0.6, 1.0),
            }

        elif size == "xlarge":
            return {
                "learning_rate": (0.001, 0.15),
                "max_depth": (10, 60),
                "min_samples_split": (2, 50),
                "min_samples_leaf": (1, 20),
                "n_estimators": (80, 320),
                "subsample": (0.5, 1.0),

            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'tiny', 'small', 'medium', 'large', or 'xlarge'.")