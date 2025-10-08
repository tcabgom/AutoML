import pandas as pd

class NumericScaler:

    def __init__(self, method: str = "minmax") -> None:
        if method not in {"standard", "minmax", "robustScaler", "none"}:
            raise ValueError("method must be one of 'standard', 'minmax', 'robustScaler' or 'none'")
        self.method = method

    def fit_column(self, column: pd.Series) -> dict():

        params = {}

        if self.method == "standard":
            params['mean'] = column.mean()
            params['std'] = column.std()
        elif self.method == "minmax":
            params['min'] = column.min()
            params['max'] = column.max()
        elif self.method == "robustScaler":
            params['median'] = column.median()
            params['iqr'] = column.quantile(0.75) - column.quantile(0.25)

        return params

    def transform_column(self, column: pd.Series, params: dict) -> pd.Series:

        if self.method == "standard":
            std = params.get('std')
            mean_p = params.get('mean')
            if mean_p is None or std is None:
                return column
            column = (column - mean_p) / std if std != 0 else column - mean_p

        elif self.method == "minmax":
            min_p = params.get('min')
            max_p = params.get('max')
            if min_p is None or max_p is None:
                return column
            if max_p == min_p:
                column = pd.Series(0.0, index=column.index)
            else:
                column = (column - min_p) / (max_p - min_p)
            column = column.clip(0, 1)

        elif self.method == "robustScaler":
            median = params.get('median').astype(float)
            iqr = params.get('iqr').astype(float)
            if median is None or iqr is None:
                return column
            if iqr == 0:
                column = column - median
            else:
                column = (column - median) / iqr

        return column
