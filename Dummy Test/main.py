import autosklearn.classification
import sklearn.model_selection
from sklearn.datasets import load_iris
import sklearn.metrics



if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))