from src.basicautoml.config import AutoMLConfig
from src.basicautoml.main import TFM_AutoML
from utils.data_loader import load_benchmark_suite, load_task_dataset


def run(folds: int = 10):
    suite = load_benchmark_suite(271)

    for task_id in suite.tasks:
        # Obtener dataset
        x, y, dataset, train_indices, test_indices = load_task_dataset(task_id)

        # Realizar particion
        X_train, y_train = x.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = x.iloc[test_indices], y.iloc[test_indices]

        # Entrenar AutoML
        config = AutoMLConfig(
            test_size=0.0,
            random_state=42,
            search_type="bayesian",
            scoring="roc_auc",
            verbose=True,
            n_trials=10,
            timeout=1200
        )

        automl = TFM_AutoML(config)
        automl.fit(X_train, y_train)



    # Evaluar y almacenar los resultados