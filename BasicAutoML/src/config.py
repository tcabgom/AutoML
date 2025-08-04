from dataclasses import dataclass, field
from algorithms.classification import DecisionTree, RandomForest, GradientBoosting, ExtraTree

@dataclass
class AutoMLConfig:
    # Data loading and preprocessing
    test_size: float = 0.2
    random_state: int = 42
    preprocessor_params: dict = field(default_factory=dict)

    # Search settings
    search_type: str = "bayesian"
    algorithms: list = field(default_factory=lambda: [
        DecisionTree.Algorithm_DTC(),
        RandomForest.Algorithm_RFC(),
        ExtraTree.Algorithm_ETC(),
        GradientBoosting.Algorithm_XGBC()
    ])
    n_trials: int = 120
    timeout: float = 60
    scoring: str = "roc_auc"
    cv: int = 5
    verbose: bool = False
