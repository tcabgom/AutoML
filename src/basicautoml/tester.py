from src.basicautoml.main import TFM_AutoML
from src.basicautoml.config import AutoMLConfig
from src.basicautoml.utils.data_loader import DataLoader
from sklearn.metrics import classification_report
import logging

def test_automl_pipeline():
    # SMALL:
    # MEDIUM: kc2, credit-g, titanic
    # LARGE: adult, bank-marketing, PhishingWebsites
    # XLARGE: covertype

    # El dataset titanic es el GOAT para sustituir columnas con muchos datos por booleanos

    dataset_name = "titanic"
    loader = DataLoader(dataset_name)
    X, y = loader.load_data()

    y = y.astype(str)

    config = AutoMLConfig(
        test_size=0.2,
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc",
        verbose=True,
        n_trials=50,
        timeout=1200
    )

    automl = TFM_AutoML(config)

    print("Training...")
    automl.fit(X, y)

    print("Evaluando el modelo...")
    score = automl.score(automl.X_test, automl.y_test)
    print(f"\nScore del modelo ({config.scoring}): {score:.4f}")

if __name__ == "__main__":
    test_automl_pipeline()
