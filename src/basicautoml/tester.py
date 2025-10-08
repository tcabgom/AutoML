from src.basicautoml.main import TFM_AutoML
from src.basicautoml.config import AutoMLConfig
from src.basicautoml.utils.data_loader import DataLoader
from sklearn.metrics import classification_report
import logging

def test_automl_pipeline():

    ##### BINARY CLASSIFICATION:
    # TINY:
    # SMALL: kc2, credit-g, titanic (WARNING: Data leaking)
    # MEDIUM: adult, bank-marketing, PhishingWebsites
    # LARGE:
    # XLARGE:

    ##### MULTI-CLASS CLASSIFICATION:
    # TINY: iris, wine
    # SMALL:
    # MEDIUM:
    # LARGE: covertype
    # XLARGE:

    dataset_name = "adult" #"APSFailure"
    loader = DataLoader(dataset_name)
    X, y = loader.load_data()

    y = y.astype(str)

    config = AutoMLConfig(
        test_size=0.2,
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc",
        verbose=True,
        n_trials=10,
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
