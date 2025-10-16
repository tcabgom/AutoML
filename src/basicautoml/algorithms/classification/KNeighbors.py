from sklearn.neighbors import KNeighborsClassifier
from .. import parent_algorithm

class Algorithm_KNN(parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "K-Nearest Neighbors Classifier"

    def get_algorithm_class(self) -> type:
        return KNeighborsClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "tiny":
            return {
                "n_neighbors": (3, 12),              # Numero de vecinos
                "weights": ["uniform", "distance"],  # Ponderacion por distancia o igual
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algoritmo de busqueda
                "p": [1, 2],                         # Distancia de Minkowski (1=Manhattan, 2=Euclidiana)
            }

        elif size == "small":
            return {
                "n_neighbors": (5, 25),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],
            }

        elif size == "medium":
            return {
                "n_neighbors": (10, 50),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],
            }

        elif size == "large":
            return {
                "n_neighbors": (20, 100),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],
            }

        elif size == "xlarge":
            return {
                "n_neighbors": (40, 200),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],
            }