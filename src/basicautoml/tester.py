from src.basicautoml.algorithms.classification import DecisionTree, RandomForest, ExtraTree, GradientBoosting, \
    LogisticRegression, KNeighbors, ExtremeGradientBoosting, HistGradientBoosting
from src.basicautoml.main import TFM_AutoML
from src.basicautoml.config import AutoMLConfig
from src.basicautoml.utils.data_loader import DataLoader
from sklearn.metrics import classification_report
import logging

def test_automl_pipeline():

    dataset_name = "adult"
    loader = DataLoader(dataset_name)
    X, y = loader.load_data()

    y = y.astype(str)

    config = AutoMLConfig(
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc",
        verbose=True,
        n_trials=100,
        timeout=600,
        n_jobs=8,
        algorithms=[
            #DecisionTree.Algorithm_DTC(),
            RandomForest.Algorithm_RFC(),
            ExtraTree.Algorithm_ETC(),
            #GradientBoosting.Algorithm_GBC(),
            LogisticRegression.Algorithm_LR(),
            #KNeighbors.Algorithm_KNN(),
            #HistGradientBoosting.Algorithm_HistGBC(),
            ExtremeGradientBoosting.Algorithm_XGBC()
        ]
    )

    automl = TFM_AutoML(config)

    print("Training...")
    automl.fit(X, y)

    print("Evaluando el modelo...")
    score = automl.score(automl.X_test, automl.y_test)
    print(f"\nScore del modelo ({config.scoring}): {score:.4f}")

if __name__ == "__main__":
    test_automl_pipeline()
