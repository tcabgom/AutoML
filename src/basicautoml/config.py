# src/basicautoml/config.py
from .algorithms.classification import DecisionTree, RandomForest, GradientBoosting, ExtraTree, LogisticRegression
from dataclasses import dataclass, field

@dataclass
class AutoMLConfig:
    # Data loading and preprocessing
    test_size: float = 0.2
    validation_size: float = 0.0
    random_state: int = 42
    preprocessor_params: dict = field(default_factory=dict)
    skip_preprocessing: bool = False

    # Search settings
    search_type: str = "bayesian"
    algorithms: list = field(default_factory=lambda: [
        DecisionTree.Algorithm_DTC(),
        RandomForest.Algorithm_RFC(),
        ExtraTree.Algorithm_ETC(),
        GradientBoosting.Algorithm_GBC(),
        #LogisticRegression.Algorithm_LR(),
    ])
    n_trials: int = 120
    timeout: float = 60
    scoring: str = "roc_auc"
    cv: int = 5
    n_jobs: int = 1
    verbose: bool = False
