import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_preprocessing
from sklearn.model_selection import train_test_split
from hyperparameter_optimization import random_search, bayesian_optimization
import warnings
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class AutoML:

    def __init__(self):
        self.is_classification = None
        self.target_column = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_validate = None # TODO
        self.y_validate = None # TODO


        self.best_model = None
        self.best_hyperparameters = None
        self.best_score = None
        self.best_algorithm = None


    def __preprocess_data(self, data):
        data_preprocessor = data_preprocessing.DataPreprocessor(data)
        return data_preprocessor.preprocess_data()

    def get_best_model(self):
        return self.best_model

    def get_best_hyperparameters(self):
        return self.best_hyperparameters

    def get_best_score(self):
        return self.best_score

    def get_best_algorithm(self):
        return self.best_algorithm


    def load_csv(self, file, sep, target_column=None, is_classification=True):

        data = pd.read_csv(file, sep=sep, encoding="utf-8")
        if target_column is None:
            self.target_column = data.columns[-1]
        else:
            self.target_column = target_column
        self.is_classification = is_classification

        processed_data = self.__preprocess_data(data)
        print(processed_data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            processed_data.drop(target_column,
            axis=1),
            processed_data[target_column],
            test_size=0.3)


    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)



    def perform_random_search(self):

        rs_best_hyperparameters, rs_best_score, rs_best_model = random_search.random_search(self.X_train, self.y_train,
                                                                                   self.X_test, self.y_test,
                                                                                   self.is_classification)

        if self.best_score is None or rs_best_score > self.best_score:
            self.best_score = rs_best_score
            self.best_hyperparameters = rs_best_hyperparameters
            self.best_model = rs_best_model

    def perform_bayesian_optimization(self):

        bo_best_score, bo_best_hyperparameters = bayesian_optimization.bayesian_optimization(self.X_train, self.y_train,
                                                                                       self.X_test, self.y_test,
                                                                                       self.is_classification)

        if self.best_score is None or bo_best_score > self.best_score:
            self.best_score = bo_best_score
            self.best_hyperparameters = bo_best_hyperparameters


    def perform_feauture_engineering(self):
        pass


if __name__ == "__main__":
    automl = AutoML()
    automl.load_csv("datasets/classification/Iris.csv", sep=",", target_column="Species", is_classification=True)
    #automl.load_csv("datasets/classification/titanic.csv", sep=",", target_column="Survived", is_classification=True)
    automl.normalize_data()
    automl.perform_random_search()
    automl.perform_bayesian_optimization()
    print(automl.get_best_score())
    print(automl.get_best_hyperparameters())

