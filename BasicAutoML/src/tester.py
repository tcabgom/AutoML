import pandas as pd
from sklearn.model_selection import train_test_split

from BasicAutoML.src.data_loader import DataLoader
from BasicAutoML.src.preprocessing import Preprocessor

from BasicAutoML.src.searches.random_search import RandomSearchAutoML
from BasicAutoML.src.searches.bayesian_optimization import BayesianSearchAutoML
from algorithms.classification import DecisionTree, RandomForest, GradientBoosting, ExtraTree




if __name__ == "__main__":
    data = DataLoader("heart-statlog")
    (X,y) = data.load_data()
    df = pd.concat([X, y], axis=1)


    preprocessor = Preprocessor(target_column="class",
                                forced_dropped_columns=[],
                                too_many_lost_values_threshold=0.3,
                                too_many_categorical_value_threshold=0.05,
                                numerical_scaling="minmax",
                                verbose=True)

    preprocessed_df = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_df.drop(columns=["class"]),
        preprocessed_df["class"],
        test_size=0.2,
        random_state=42
    )


    automl = BayesianSearchAutoML(
        algorithms=[DecisionTree.Algorithm_DTC(), RandomForest.Algorithm_RFC(), ExtraTree.Algorithm_ETC(), GradientBoosting.Algorithm_XGBC()],
        n_trials=500,
        scoring="roc_auc",
        cv=5,
        verbose=True,
        random_state=42,
        timeout=300
    )

    automl.fit(X_train, y_train)

    print("\n--- Resultados ---")
    print("Mejores hiperparámetros:", automl.best_params)
    print("Mejor score CV:", automl.best_score)
    print("Score ROC AUC en test:", automl.score(X_test, y_test))