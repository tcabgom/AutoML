import json
import os

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASIC_METAFEATURES = [
    'NumberOfInstances',
    'NumberOfFeatures',
    'Dimensionality'
]

STATISTICAL_METAFEATURES = [
    'ClassEntropy',
    'MeanMutualInformation',
    'MaxMutualInformation',
    'MinMutualInformation',
    'MajorityClassPercentage',
    'MinorityClassPercentage'
]

LANDMARKERS = [
    'REPTreeDepth1AUC',
    'REPTreeDepth2AUC',
    'REPTreeDepth3AUC',
    'RandomTreeDepth1AUC',
    'RandomTreeDepth2AUC',
    'RandomTreeDepth3AUC',
    'kNN1NAUC',
    'DecisionStumpAUC',
    'NaiveBayesAUC',
]

def obtain_metafeatures(dataset: object) -> dict:
    # En main.py
    qualities = dataset.qualities
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    meta_features = {}

    def get_or_compute(keys: list[str], compute_fn) -> dict[str, float]:
        if not any(k in qualities for k in keys):
            return compute_fn(X, y) if compute_fn.__code__.co_argcount > 1 else compute_fn(X)
        return {k: qualities[k] for k in keys if k in qualities}

    # Basic meta-features
    meta_features.update(get_or_compute(BASIC_METAFEATURES, compute_basic_meta_features))

    # Statistical meta-features
    meta_features.update(get_or_compute(STATISTICAL_METAFEATURES, compute_statistical_meta_features))

    # Landmarkers
    meta_features.update(get_or_compute(LANDMARKERS, compute_landmarkers))

    print("[INFO] Meta-features obtained:", meta_features)
    return meta_features


def compute_landmarkers(X: np.ndarray, y: np.ndarray) -> dict:
    # Auxiliar, en obtain_metafeatures
    print("[INFO] No landmarkers found in dataset. Computing them now...")
    results = {}
    y = LabelEncoder().fit_transform(y)

    models = {
        # Arboles deterministas
        'REPTreeDepth1AUC': DecisionTreeClassifier(max_depth=1, random_state=0),
        'REPTreeDepth2AUC': DecisionTreeClassifier(max_depth=2, random_state=0),
        'REPTreeDepth3AUC': DecisionTreeClassifier(max_depth=3, random_state=0),
        # Arboles con aleatoriedad en la seleccion de atributos
        'RandomTreeDepth1AUC': DecisionTreeClassifier(max_depth=1, max_features='sqrt', random_state=42),
        'RandomTreeDepth2AUC': DecisionTreeClassifier(max_depth=2, max_features='sqrt', random_state=42),
        'RandomTreeDepth3AUC': DecisionTreeClassifier(max_depth=3, max_features='sqrt', random_state=42),
        # Otros modelos simples
        'DecisionStumpAUC': DecisionTreeClassifier(max_depth=1, random_state=0),
        'NaiveBayesAUC': GaussianNB(),
        'kNN1NAUC': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    }

    for name, model in models.items():
        try:
            auc = np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))
        except Exception:
            auc = np.nan
        results[name] = auc

    return results


def compute_basic_meta_features(X: np.ndarray) -> dict:
    meta_features = {
        'NumberOfInstances': X.shape[0],
        'NumberOfFeatures': X.shape[1],
        'Dimensionality': X.shape[1] / X.shape[0] if X.shape[0] > 0 else np.nan
    }
    return meta_features


def compute_statistical_meta_features(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.feature_selection import mutual_info_classif
    from scipy.stats import entropy

    y = np.asarray(y)
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)

    mi = mutual_info_classif(X, y, discrete_features='auto')

    return {
        "ClassEntropy": entropy(class_probs),
        "MeanMutualInformation": np.mean(mi),
        "MaxMutualInformation": np.max(mi),
        "MinMutualInformation": np.min(mi),
        "MajorityClassPercentage": np.max(class_probs) * 100,
        "MinorityClassPercentage": np.min(class_probs) * 100,
    }


def save_meta_record(meta_features: dict, best_model_name: str, best_model_params: dict, best_model_score: float, dataset_size: str, file_path: str = "meta_database.json") -> None:
    # En main.py, al final de fit()
    record = {
        'meta_features': meta_features,
        'best_model': {
            'algorithm': best_model_name,
            'hyperparameters': best_model_params,
            'score': best_model_score
        },
        'dataset_size': dataset_size
    }

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    for existing_record in data:
        if existing_record['meta_features'] == record['meta_features'] and existing_record['best_model']['algorithm'] == record['best_model']['algorithm']:
            if existing_record['best_model']['score'] >= record['best_model']['score']:
                print("[INFO] Similar meta-record with equal or better score already exists. Skipping save.")
                return
            else:
                data.remove(existing_record)
                print("[INFO] Similar meta-record found with lower score. Replacing it.")

    data.append(record)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print("[INFO] Meta-record saved to meta-database.")


def load_meta_database(file_path: str = "meta_database.json") -> list:
    # Auxiliar, en find_nearest_datasets
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as f:
        return json.load(f)

def find_nearest_datasets(
                        query_meta_features: dict,
                        n: int,
                        allowed_algorithms: list(),
                        dataset_size: str,
                        file_path: str = "meta_database.json") -> list:
    # En main.py, en fit() antes de realizar la busqueda
    for k, v in query_meta_features.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            print(f"[WARN] {k} was NaN, replaced with 0.5 for distance calculation.")
            query_meta_features[k] = 0.5

    db = load_meta_database(file_path)
    if not db:
        return []

    filtered_db = [
        record for record in db
        if record.get("best_model", {}).get("algorithm") in allowed_algorithms and record.get("dataset_size") == dataset_size
    ]

    if not filtered_db:
        print("[WARN] No datasets in meta-database match the allowed algorithms.")
        return []

    feature_names = list(query_meta_features.keys())
    db_vectors = []

    for record in filtered_db:
        vec = [record['meta_features'].get(feature, np.nan) for feature in feature_names]
        db_vectors.append(vec)

    query_vector = np.array([list(query_meta_features.values())])
    db_matrix = np.array(db_vectors)

    distances = euclidean_distances(query_vector, db_matrix)[0]

    nearest_indices = np.argsort(distances)[:n]

    nearest_records = [filtered_db[i]["best_model"] for i in nearest_indices]
    return nearest_records
