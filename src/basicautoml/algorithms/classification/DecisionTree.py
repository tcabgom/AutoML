from sklearn.tree import DecisionTreeClassifier
from .. import parent_algorithm

class Algorithm_DTC (parent_algorithm.ParentAlgorithm):

    def get_name(self) -> str:
        return "Decision Tree Classifier"

    def get_algorithm_class(self) -> type:
        return DecisionTreeClassifier

    def get_algorithm_params(self, size: str = "medium") -> dict:

        if size == "small":
            return {
                "criterion": ["gini", "entropy"],
                "splitter": ["best"],
                "max_depth": (2, 15),
                "class_weight": [None, "balanced"],
                "max_leaf_nodes": (2, 20),
            }
        elif size == "medium":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": (3, 25),
                "class_weight": [None, "balanced"],
                "max_leaf_nodes": (2, 50),
            }
        elif size == "large":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": (5, 40),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10),
                "max_features": [None, "sqrt", "log2"],
                "class_weight": [None, "balanced"],
                "max_leaf_nodes": (2, 100),
                "ccp_alpha": (0.0, 0.05),
            }
        elif size == "xlarge":
            return {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": (5, 60),
                "min_samples_split": (2, 50),
                "min_samples_leaf": (1, 20),
                "max_features": [None, "sqrt", "log2"],
                "class_weight": [None, "balanced"],
                "max_leaf_nodes": (2, 200),
                "min_impurity_decrease": (0.0, 0.01),
                "ccp_alpha": (0.0, 0.1),
            }

        else:
            raise ValueError(f"Unknown size '{size}'. Use 'small', 'medium', 'large', or 'xlarge'.")