import random

from src.basicautoml.config import AutoMLConfig
from src.basicautoml.main import TFM_AutoML
from src.benchmark.utils.data_storer import store_data
from utils.data_loader import load_benchmark_suite, load_task_dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from datetime import datetime
import time
import numpy as np

TASK_IDS = [
    3945, 146818, 146820, 167120, 168350, 168757, 168868, 168911, 189354,
    189356, 189922, 190137, 190392, 190410, 190411, 190412, 359955, 359956,
    359958, 359962, 359965, 359966, 359967, 359968, 359971, 359972, 359973,
    359975, 359979, 359980, 359982, 359983, 359988, 359989, 359990, 359991,
    359992, 359994, 360113, 360114, 360975
]

def run():
    #suite = load_benchmark_suite(271)

    random.shuffle(TASK_IDS)

    for task_id in [359955, 146818, 168757, 146820, 168350, 359956]: #suite.tasks:
        # Obtener dataset
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

                n_trials=250,
                timeout=1200,
                scoring="roc_auc",
                cv=5,
                verbose=True,
            )

            automl = TFM_AutoML(config)

            t0_train = time.time()
            automl.fit(X_train, y_train)
            training_duration = int((time.time() - t0_train)*10)/10

            # Evaluar y almacenar los resultados
            t0_pred = time.time()
            y_pred = automl.predict(X_test)
            y_proba = automl.predict_proba(X_test)
            score = automl.score(X_test, y_test)

            try:
                acc = accuracy_score(y_test, y_pred)
            except Exception:
                acc = None
            try:
                balacc = balanced_accuracy_score(y_test, y_pred)
            except Exception:
                balacc = None
            try:
                logloss = log_loss(y_test, y_proba)
            except Exception:
                logloss = None

            predict_duration = int((time.time() - t0_pred)*10)/10

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
            new_row["duration"] = training_duration + predict_duration
            new_row["training_duration"] = training_duration
            new_row["predict_duration"] = predict_duration
            new_row["models_count"] = automl.searcher.trained_models
            new_row["seed"] = config.random_state
            new_row["info"] = None
            new_row["acc"] = acc
            new_row["balacc"] = balacc
            new_row["logloss"] = logloss
            new_row["models_ensemble_count"] = None
            new_row["auc"] = score if len(np.unique(y)) == 2 else None

            store_data("results.csv", new_row)
            print(f"Result stored for dataset {dataset.name}, fold {fold}")
