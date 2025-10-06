import pandas as pd

def clasify_dataset_size(X: pd.DataFrame) -> str:

    n_samples, n_features = X.shape
    size_score = n_samples * n_features

    if size_score < 10_000:
        selected_size = "tiny"
    elif size_score < 100_000:
        selected_size = "small"
    elif size_score < 1_000_000:
        selected_size = "medium"
    elif size_score < 10_000_000:
        selected_size = "large"
    else:
        selected_size = "xlarge"

    print(f"Dataset size classified as '{selected_size}' with score {size_score} ({n_samples} samples, {n_features} features).")
    return selected_size
