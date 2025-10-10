from sklearn.linear_model import LogisticRegression
from .. import parent_algorithm

class Algorithm_LR(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Logistic Regression"

    def get_algorithm_class(self) -> type:
        return LogisticRegression

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "tiny":
            return {
                "penalty": ["l1", "l2"],                  # Tipo de regularizacion
                "solver": ["liblinear"],                  # Optimizadores compatibles
                "class_weight": ["balanced"],             # Manejo de clases desbalanceadas
                "C": [10**x for x in range(-2, 3)],       # Inverso de la regularizacion
                "max_iter": (200, 500),                   # Iteraciones maximas
            }
        elif size == "small":
            return {
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "class_weight": ["balanced"],
                "C": [10**x for x in range(-2, 3)],
                "max_iter": (200, 800),
            }
        elif size == "medium":
            return {
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "class_weight": ["balanced"],
                "C": [10**x for x in range(-3, 4)],
                "max_iter": (200, 1000),
            }
        elif size == "large":
            return {
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "class_weight": ["balanced"],
                "C": [10**x for x in range(-4, 5)],
                "max_iter": (500, 1500),
            }
        elif size == "xlarge":
            return {
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "class_weight": ["balanced"],
                "C": [10**x for x in range(-4, 5)],
                "max_iter": (1000, 2000),
            }
        else:
            raise ValueError(f"Unknown size '{size}'. Use 'tiny', 'small', 'medium', 'large', or 'xlarge'.")
