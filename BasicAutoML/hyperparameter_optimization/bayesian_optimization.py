from BasicAutoML.algorithms.classification import decision_tree_classifier, random_forest_classifier, k_neighbors_classifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


ITERATIONS = 100

CLASSIFICATION_ALGORITHMS = [
                    decision_tree_classifier.DTC_Algorithm,
                    random_forest_classifier.RFC_Algorithm,
                    k_neighbors_classifier.KNC_Algorithm]

REGRESSION_ALGORITHMS = []


def objective_function(X_train, y_train, X_test, y_test, algorithm, hyperparameters):
    alg = algorithm()
    alg.load_hyperparameters(hyperparameters)
    alg.fit(X_train, y_train)
    return -alg.evaluate(X_test, y_test)


def store_hyperparameters_in_map(values, keys):
    return {key: value for key, value in zip(keys, values)}


def bayesian_optimization(X_train, y_train, X_test, y_test, classification=True):

    if classification:
        algorithm_list = CLASSIFICATION_ALGORITHMS
    else:
        algorithm_list = REGRESSION_ALGORITHMS

    best_score = None
    best_model = None
    best_hyperparameters = None

    for algorithm in algorithm_list:

        algBase = algorithm()
        hyperparameters_limits = algBase.get_hyperparameter_limits()
        hyperparameter_keys = list(hyperparameters_limits.keys())

        bounds = [tuple(limit) for limit in hyperparameters_limits.values()]

        gp = GaussianProcessRegressor(kernel=Matern(), alpha=1e-6)

        # Definimos valores iniciales uniformemente distribuidos
        initial_points = np.random.uniform(
            [bound[0] for bound in bounds],
            [bound[1] for bound in bounds],
            (5, len(bounds))
        )

        evaluations = []

        for point in initial_points:
            hyperparameters = store_hyperparameters_in_map(point, hyperparameter_keys)
            evaluations.append(objective_function(X_train, y_train, X_test, y_test, algorithm, hyperparameters))

        X_sample = initial_points
        y_sample = np.array(evaluations)

        for i in range(ITERATIONS):
            gp.fit(X_sample, y_sample)

            def adquisition_function(x):
                mean, std = gp.predict(x.reshape(1, -1), return_std=True)
                best = np.min(y_sample)
                improvement = best - mean
                z = improvement / (std + 1e-9)  # Evitar dividir por 0
                return -((improvement * norm.cdf(z)) + (std * norm.pdf(z)))

            res = minimize(
                lambda x: adquisition_function(x),
                x0 = np.random.uniform([bound[0] for bound in bounds], [bound[1] for bound in bounds]),
                bounds=bounds,
                method="L-BFGS-B")

            new_point = res.x
            hyperparameters = store_hyperparameters_in_map(new_point, hyperparameter_keys)
            new_evaluation = objective_function(X_train, y_train, X_test, y_test, algorithm, hyperparameters)

            X_sample = np.vstack((X_sample, new_point))
            y_sample = np.append(y_sample, new_evaluation)

        current_best_index = np.argmin(y_sample)
        current_best_score = y_sample[current_best_index]
        current_best_hyperparameters = store_hyperparameters_in_map(X_sample[current_best_index], hyperparameter_keys)

        if best_score is None or current_best_score > best_score:
            print(best_score)
            best_score = current_best_score
            best_hyperparameters = current_best_hyperparameters

        return best_score, best_hyperparameters
            