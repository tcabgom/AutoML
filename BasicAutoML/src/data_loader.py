from sklearn.datasets import fetch_openml

class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_data(self) -> tuple:
        """
        Loads the dataset from OpenML.

        Returns:
            tuple: A tuple containing the features (X) and target (y) of the dataset.
        """
        data = fetch_openml(self.dataset_name, version=1, as_frame=True)
        X = data.data
        y = data.target
        return X, y
