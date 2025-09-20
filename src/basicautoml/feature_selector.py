import pandas as pd

class FeatureSelector:

    def __init__(self,
                 method: str ='pca',
                 params: dict = None,
                 verbose: bool = False):

        self.method = method
        self.params = params if params is not None else {}
        self.selector = None
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:

        if self.method == 'pca':
            from sklearn.decomposition import PCA
            self.selector = PCA(**self.params).fit(X)
            return self
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector is None:
            raise ValueError("The selector has not been fitted yet. Call 'fit' before 'transform'.")

        X_transformed = self.selector.transform(X)
        if self.verbose:
            explained = self.selector.explained_variance_ratio_
            print(f"[INFO] Varianza explicada acumulada: {explained.cumsum()}")
            print("[INFO] Ranking de features más influyentes por componente:")
            loadings = pd.DataFrame(
                self.selector.components_.T,
                index=X.columns,
                columns=[f'PC{i + 1}' for i in range(len(self.selector.components_))]
            )
            for pc in loadings.columns:
                top_features = loadings[pc].abs().sort_values(ascending=False).head(5)
                print(f"  {pc}: {', '.join(top_features.index)}")
        columns = [f'PC{i+1}' for i in range(X_transformed.shape[1])]
        return pd.DataFrame(X_transformed, columns=columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
