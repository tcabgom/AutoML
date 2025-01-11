
class ParentAlgorithm:

    def __init__(self):
        self.model = None

    def get_hyperparameter_limits(self):
        pass

    def load_hyperparameters(self, hyperparameters):
        pass

    def fit(self, X_train, y_train):
        pass

    def evaluate(self, X_test, y_test):
        pass

    def get_model(self):
        return self.model

    def get_hyperparameters_map(self):
        pass
