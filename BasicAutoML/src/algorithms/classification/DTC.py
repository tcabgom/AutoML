from sklearn.tree import DecisionTreeClassifier

model_class = DecisionTreeClassifier
param_distribution = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": (1,25),
    "min_sample_split": (2,20),
    "min_sample_leaf": (1,10),
    "class_weight": [None, "balanced"],
    "max_leaf_nodes": (2,100),
    "cpp_alpha": (0.0, 0.05)
}