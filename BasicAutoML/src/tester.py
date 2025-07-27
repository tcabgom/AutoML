import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from BasicAutoML.src.preprocessing import Preprocessor
from sklearn.datasets import fetch_openml, make_classification

from BasicAutoML.src.searches.random_search_unimodel import RandomSearchUnimodelAutoML
if __name__ == "__main__":
    df = pd.read_csv("../OldAutoML/datasets/classification/titanic.csv", sep=",", encoding="utf-8")
    preprocessor = Preprocessor(target_column="Survived",
                                forced_dropped_columns=["PassengerId"],
                                too_many_lost_values_threshold=0.3,
                                too_many_categorical_value_threshold=0.05,
                                numerical_scaling="minmax",
                                verbose=True)

    preprocessed_df = preprocessor.fit_transform(df)
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )


    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_df.drop(columns=["Survived"]), preprocessed_df["Survived"], test_size=0.3, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_distributions = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": (2,10),
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"]
    }

    automl = RandomSearchUnimodelAutoML(
        model_class=DecisionTreeClassifier,
        param_distributions=param_distributions,
        n_trials=50,
        scoring="roc_auc",
        cv=5,
        verbose=True,
        random_state=42,
        timeout=30
    )

    automl.fit(X_train, y_train)

    print("\n--- Resultados ---")
    print("Mejores hiperparámetros:", automl.best_params)
    print("Mejor score CV:", automl.best_score)
    print("Score ROC AUC en test:", automl.score(X_test, y_test))