from sklearn.ensemble import ExtraTreesClassifier

from .. import parent_algorithm

class Algorithm_ETC(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Extra Trees Classifier"

    def get_algorithm_class(self) -> type:
        return ExtraTreesClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "tiny":
            return {
                "criterion": ["gini", "entropy", "log_loss"],   # Funcion para medir la calidad de una division
                "class_weight": ["balanced"],                   # Manejo de clases desbalanceadas
                "max_depth": (2, 10),                           # Profundidad maxima del arbol
                "min_samples_split": (2, 5),                    # Minimo de muestras para dividir un nodo
                "min_samples_leaf": (1, 5),                     # Minimo de muestras en un nodo hoja
                "n_estimators": (5, 20),                        # Numero de arboles en el bosque
            }
        elif size == "small":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "class_weight": ["balanced"],
                "max_depth": (3, 15),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 5),
                "n_estimators": (10, 40),
            }
        elif size == "medium":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "class_weight": ["balanced"],
                "max_depth": (3, 25),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
                "n_estimators": (20, 80),
            }
        elif size == "large":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "class_weight": ["balanced"],
                "max_depth": (5, 40),
                "min_samples_split": (2, 30),
                "min_samples_leaf": (1, 15),
                "n_estimators": (40, 160),
            }
        elif size == "xlarge":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "class_weight": ["balanced"],
                "max_depth": (10, 60),
                "min_samples_split": (2, 50),
                "min_samples_leaf": (1, 20),
                "n_estimators": (80, 320),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'tiny', 'small', 'medium', 'large', or 'xlarge'.")