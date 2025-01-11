from .. import parent_algorithm
#import xgboost


HYPERPARAMETER_LIMITS = {
    "n_estimators": [3, 19],
    "max_depth": [2, 20],
    "learning_rate": [0.01, 0.5],
    "gamma": [0, 0.5],
    "min_child_weight": [1, 5],
    "subsample": [0.5, 1],
    "colsample_bytree": [0.5, 1],
    "colsample_bylevel": [0.5, 1],
    "colsample_bynode": [0.5, 1],
    "reg_alpha": [0, 1],
    "reg_lambda": [0, 1]
}