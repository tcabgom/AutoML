import pandas as pd
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


    def __preprocess_data(self, data):
        data_preprocessor = data_preprocessing.DataPreprocessor(data)
        return data_preprocessor.preprocess_data()


    def load_csv(self, file, sep, target_column=None, is_classification=True):

        data = pd.read_csv(file, sep=sep, encoding="utf-8")
        if target_column is None:
            self.target_column = data.columns[-1]
        else:
            self.target_column = target_column
        self.is_classification = is_classification

        target_column_name = data.columns[self.target_column]

        processed_data = self.__preprocess_data(data)
        print(processed_data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            processed_data.drop(target_column_name, axis=1), processed_data[target_column_name], test_size=0.3)

    def perform_random_search(self):
        #best_hyperparameters, best_score, best_model = random_search.random_search(self.X_train, self.y_train, self.X_test, self.y_test, self.is_classification)
        best_score, best_hyperparameters = bayesian_optimization.bayesian_optimization(self.X_train, self.y_train, self.X_test, self.y_test, self.is_classification)
        print(best_score)
        print(best_hyperparameters)
        #print(best_score)
        #print(best_hyperparameters)
        #print(best_model.predict(self.X_test))
        #print(best_model.predict(self.X_train))
        #plt.figure(figsize=(20, 10))
        #plot_tree(best_model, filled=True, feature_names=self.X_train.columns)
        #plt.show()

if __name__ == "__main__":
    automl = AutoML()
    #automl.load_csv("datasets/classification/Iris.csv", sep=",", target_column=5, is_classification=True)
    automl.load_csv("datasets/classification/titanic.csv", sep=",", target_column=1, is_classification=True)
    automl.perform_random_search()
