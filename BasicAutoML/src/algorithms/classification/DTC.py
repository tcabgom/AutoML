from sklearn.tree import DecisionTreeClassifier
from .. import parent_algorithm

class Algorithm_DTC (parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Decision Tree Classifier"

    def get_algorithm_class(self) -> type:
        return DecisionTreeClassifier

    def get_algorithm_params(self) -> dict:
        return {
            "criterion": ["gini", "entropy", "log_loss"],   # Evaluation function of division quality
            "splitter": ["best", "random"],                 # Division strategy
            "max_depth": (1, 25),                           # Tree max depth
            "class_weight": [None, "balanced"],
            "max_leaf_nodes": (2, 100),
        }