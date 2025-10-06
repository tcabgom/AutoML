from src.basicautoml.config import AutoMLConfig
from src.basicautoml.main import TFM_AutoML
from src.benchmark.utils.data_storer import store_data
from utils.data_loader import load_benchmark_suite, load_task_dataset
from datetime import datetime
import time
import random
import numpy as np

def run():
    suite = load_benchmark_suite(271)

    for task_id in suite.tasks:
        # Obtener dataset
        task_id = 359983
        x, y, dataset, train_indices, test_indices = load_task_dataset(task_id)

        if len(np.unique(y)) > 2:
            print(f" ! Dataset {dataset.name} is not binary classification")
            continue

        for fold in range(10):
            # Realizar particion
            X_train, y_train = x.iloc[train_indices[fold]], y.iloc[train_indices[fold]]
            X_test, y_test = x.iloc[test_indices[fold]], y.iloc[test_indices[fold]]

            # Entrenar AutoML
            config = AutoMLConfig(
                test_size=0.0,
                random_state=int(time.time()),
                search_type="bayesian",

                n_trials=1000,
                timeout=3600,
                scoring="roc_auc",
                cv=5,
                verbose=True,
            )

            automl = TFM_AutoML(config)
            automl.fit(X_train, y_train)

            # Evaluar y almacenar los resultados
            score = automl.score(X_test, y_test)

            new_row = {}
            new_row["id"] = f"openml.org/t/{task_id}"
            new_row["task"] = dataset.name
            new_row["framework"] = "TFM_AutoML"
            new_row["constraint"] = None
            new_row["fold"] = fold
            new_row["type"] = "binary" if len(np.unique(y)) == 2 else "multiclass"
            new_row["result"] = score
            new_row["metric"] = config.scoring
            new_row["mode"] = None
            new_row["version"] = None
            new_row["params"] = str(automl.best_params)
            new_row["app_version"] = None
            new_row["utc"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            new_row["duration"] = None
            new_row["training_duration"] = None
            new_row["predict_duration"] = None
            new_row["models_count"] = None
            new_row["seed"] = config.random_state
            new_row["info"] = None
            new_row["acc"] = None
            new_row["balacc"] = None
            new_row["logloss"] = None
            new_row["models_ensemble_count"] = None
            new_row["auc"] = None

            store_data("results.csv", new_row)