import random
from tqdm import tqdm
from BasicAutoML.algorithms.classification import decision_tree_classifier, random_forest_classifier, k_neighbors_classifier


ITERATIONS = 750

CLASSIFICATION_ALGORITHMS = [
                    decision_tree_classifier.DTC_Algorithm,
                    random_forest_classifier.RFC_Algorithm,
                    k_neighbors_classifier.KNC_Algorithm]

REGRESSION_ALGORITHMS = []


def select_random_hyperparameters(hyperparameter_space):

    hyperparameters = {}

    for key, value in hyperparameter_space.items():
        # When the value must be an integer it will be rounded in the algorithm class when loading all the values
        hyperparameters[key] = random.uniform(value[0], value[1])

    return hyperparameters


def random_search(X_train, y_train, X_test, y_test, classification=True):

    best_hyperparameters = None
    best_score = None
    best_model = None

    if classification:
        algorithm_list = CLASSIFICATION_ALGORITHMS
    else:
        algorithm_list = REGRESSION_ALGORITHMS

    for algorithm in algorithm_list:

        algBase = algorithm()
        hyperparameters_limits = algBase.get_hyperparameter_limits()

        for _ in tqdm(range(ITERATIONS), desc=f"Evaluating {algorithm.__name__}...", unit=" iteration"):

            alg = algorithm()   # Create an instance of the algorithm
            chosen_hyperparameters = select_random_hyperparameters(hyperparameters_limits)
            alg.load_hyperparameters(chosen_hyperparameters)
            alg.fit(X_train, y_train)
            score = alg.evaluate(X_test, y_test)

            if best_score is None or score > best_score:
                best_score = score
                best_hyperparameters = alg.get_hyperparameters_map()
                best_model = alg.get_model()

    return best_hyperparameters, best_score, best_model
