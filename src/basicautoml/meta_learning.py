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
    missing_meta_features = []


    for k in BASIC_METAFEATURES + STATISTICAL_METAFEATURES + LANDMARKERS:
        if k not in qualities.keys() or qualities[k] is None:
            missing_meta_features.append(k)
        else:
            meta_features.update({k: qualities[k]})

    if len(missing_meta_features) != 0:
        calculated = compute_missing_meta_features(missing_meta_features, X, y)
        meta_features.update(calculated)

    print("[INFO] Meta-features obtained:", meta_features)
    return meta_features


def compute_missing_meta_features(missing_meta_features: list, X: np.ndarray, y: np.ndarray) -> dict:
    print("[INFO] Computing missing meta-features:", missing_meta_features)
    results = {}

    def obtain_landmarker_score(model) -> float:
        try:
            auc = np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))
        except Exception:
            auc = np.nan
        return auc

    # LANDMARKERS
    if 'REPTreeDepth1AUC' in missing_meta_features:
        results.update({'REPTreeDepth1AUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=1, random_state=0))})
    if 'REPTreeDepth2AUC' in missing_meta_features:
        results.update({'REPTreeDepth2AUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=2, random_state=0))})
    if 'REPTreeDepth3AUC' in missing_meta_features:
        results.update({'REPTreeDepth3AUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=3, random_state=0))})
    if 'RandomTreeDepth1AUC' in missing_meta_features:
        results.update({'RandomTreeDepth1AUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=1, splitter='random', random_state=0))})
    if 'RandomTreeDepth2AUC' in missing_meta_features:
        results.update({'RandomTreeDepth2AUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=2, splitter='random', random_state=0))})
    if 'RandomTreeDepth3AUC' in missing_meta_features:
        results.update({'RandomTreeDepth3AUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=3, splitter='random', random_state=0))})
    if 'kNN1NAUC' in missing_meta_features:
        results.update({'kNN1NAUC': obtain_landmarker_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)))})
    if 'DecisionStumpAUC' in missing_meta_features:
        results.update({'DecisionStumpAUC': obtain_landmarker_score(DecisionTreeClassifier(max_depth=1, random_state=0))})
    if 'NaiveBayesAUC' in missing_meta_features:
        results.update({'NaiveBayesAUC': obtain_landmarker_score(GaussianNB())})

    # BASIC METAFEATURES
    if 'NumberOfInstances' in missing_meta_features:
        results.update({'NumberOfInstances': X.shape[0]})
    if 'NumberOfFeatures' in missing_meta_features:
        results.update({'NumberOfFeatures': X.shape[1]})
    if 'Dimensionality' in missing_meta_features:
        results.update({'Dimensionality': X.shape[1] / X.shape[0] if X.shape[0] > 0 else np.nan})

    # STATISTICAL METAFEATURES
    if any(f in missing_meta_features for f in STATISTICAL_METAFEATURES):
        from sklearn.feature_selection import mutual_info_classif
        from scipy.stats import entropy

        y = np.asarray(y)
        if y.dtype == object or not np.issubdtype(y.dtype, np.integer):
            le = LabelEncoder()
            y = le.fit_transform(y)
        class_counts = np.bincount(y)
        class_probs = class_counts / len(y)

        mi = mutual_info_classif(X, y, discrete_features='auto')
        mi = np.nan_to_num(mi, nan=0.0)

        if 'ClassEntropy' in missing_meta_features:
            results.update({"ClassEntropy": entropy(class_probs)})
        if 'MeanMutualInformation' in missing_meta_features:
            results.update({"MeanMutualInformation": np.mean(mi)})
        if 'MaxMutualInformation' in missing_meta_features:
            results.update({"MaxMutualInformation": np.max(mi)})
        if 'MinMutualInformation' in missing_meta_features:
            results.update({"MinMutualInformation": np.min(mi)})
        if 'MajorityClassPercentage' in missing_meta_features:
            results.update({"MajorityClassPercentage": np.max(class_probs) * 100})
        if 'MinorityClassPercentage' in missing_meta_features:
            results.update({"MinorityClassPercentage": np.min(class_probs) * 100})

    return results


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
            db = json.load(f)
        if not isinstance(db, dict):
            db = {"data": db, "normalized": False, "normalization_stats": {}}
    else:
        db = {"data": [], "normalized": False, "normalization_stats": {}}

    data = db["data"]

    if db.get("normalized", False):
        stats = db.get("normalization_stats", {})
        norm_meta_features = {}

        for k, v in meta_features.items():
            if k in stats and isinstance(v, (int, float)):
                mean = stats[k]["mean"]
                std = stats[k]["std"]
                norm_meta_features[k] = (v - mean) / std
            else:
                norm_meta_features[k] = v

        record["meta_features"] = norm_meta_features

    for existing_record in data:
        if existing_record['meta_features'] == record['meta_features'] and existing_record['best_model']['algorithm'] == record['best_model']['algorithm']:
            if existing_record['best_model']['score'] >= record['best_model']['score']:
                print("[INFO] Similar meta-record with equal or better score already exists. Skipping save.")
                return
            else:
                data.remove(existing_record)
                print("[INFO] Similar meta-record found with lower score. Replacing it.")

    data.append(record)
    db["data"] = data

    with open(file_path, "w") as f:
        json.dump(db, f, indent=4)

    print("[INFO] Meta-record saved to meta-database.")


def load_meta_database(file_path: str = "meta_database.json") -> dict:
    # Auxiliar, en find_nearest_datasets
    if not os.path.exists(file_path):
        return {"data": [], "normalized": False, "normalization_stats": {}}

    with open(file_path, "r") as f:
        db = json.load(f)

    if isinstance(db, list):
        db = {"data": db, "normalized": False, "normalization_stats": {}}

    db.setdefault("data", [])
    db.setdefault("normalized", False)
    db.setdefault("normalization_stats", {})

    return db

def find_nearest_datasets(
                        query_meta_features: dict,
                        n: int,
                        allowed_algorithms: list(),
                        dataset_size: str,
                        file_path: str = "meta_database.json") -> list:
    # En main.py, en fit() antes de realizar la busqueda

    # Handle NaN values in query meta-features
    for k, v in query_meta_features.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            print(f"[WARN] {k} was NaN, replaced with 0.5 for distance calculation.")
            query_meta_features[k] = 0.5

    # Load meta-database
    db = load_meta_database(file_path)
    if not db:
        return []

    # Normalize query if database is normalized
    if db.get("normalized", False):
        stats = db.get("normalization_stats", {})
        norm_query = {}
        for k, v in query_meta_features.items():
            if k in stats and isinstance(v, (int, float)):
                mean = stats[k]["mean"]
                std = stats[k]["std"]
                norm_query[k] = (v - mean) / std
            else:
                norm_query[k] = v
        query_meta_features = norm_query

    # Filter database records by allowed algorithms and dataset size
    data = db.get("data", [])
    filtered_db = [
        record for record in data
        if record.get("best_model", {}).get("algorithm") in allowed_algorithms and record.get("dataset_size") == dataset_size
    ]

    # If no records match, return empty list
    if not filtered_db:
        print("[WARN] No datasets in meta-database match the allowed algorithms.")
        return []

    def impute_nan(value):
        if value is None:
            return 0.5
        if isinstance(value, float) and np.isnan(value):
            return 0.5
        return value

    # Compute distances
    feature_names = list(query_meta_features.keys())
    db_vectors = []
    for record in filtered_db:
        vec = [impute_nan(record['meta_features'].get(feature, np.nan))for feature in feature_names]
        db_vectors.append(vec)

    query_vector = np.array([list(query_meta_features.values())])
    db_matrix = np.array(db_vectors)

    distances = euclidean_distances(query_vector, db_matrix)[0]

    nearest_indices = np.argsort(distances)[:n]

    nearest_records = [filtered_db[i]["best_model"] for i in nearest_indices]
    return nearest_records

def fit_normalizer(file_path: str = "meta_database.json") -> dict:

    if not os.path.exists(file_path):
        print("[WARN] No meta-database found to fit normalizer.")
        return

    with open(file_path, "r") as f:
        db = json.load(f)

    data = db.get("data", db if isinstance(db, list) else [])
    if not data:
        print("[WARN] No data to fit normalizer.")
        return

    all_features = {}
    for record in data:
        for feature, value in record['meta_features'].items():
            if isinstance(value, (int, float)):
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    all_features.setdefault(feature, []).append(value)

    stats = {}
    for feature, values in all_features.items():
        if not values:
            # Fail save
            stats[feature] = {"mean": 0.0, "std": 1.0}
            continue

        arr = np.array(values, dtype=float)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        stats[feature] = {"mean": mean, "std": std if std > 0 else 1.0}

    db["normalized"] = False
    db["normalization_stats"] = stats

    with open(file_path, "w") as f:
        json.dump(db, f, indent=4)

    print("[INFO] Normalizer fitted and saved to meta-database.")


def transform_normalizer(file_path: str = "meta_database.json"):

    if not os.path.exists(file_path):
        print("[WARN] Meta-database not found. Nothing to normalize.")
        return

    with open(file_path, "r") as f:
        db = json.load(f)

    if db.get("normalized", False):
        print("[INFO] Meta-database is already normalized.")
        return

    stats = db.get("normalization_stats", None)
    if stats is None:
        print("[WARN] No normalizer stats found. Fit the normalizer first.")
        return

    data = db.get("data", db if isinstance(db, list) else [])
    for record in data:
        mf = record.get("meta_features", {})
        for k, v in mf.items():
            if k in stats and isinstance(v, (int, float)):
                mean, std = stats[k]["mean"], stats[k]["std"]
                mf[k] = (v - mean) / std

    db["data"] = data
    db["normalized"] = True

    with open(file_path, "w") as f:
        json.dump(db, f, indent=4)

    print("[INFO] Meta-database normalized and saved.")
