import pandas as pd


class Preprocessor:
    def __init__(self,
                 target_column: str,
                 forced_dropped_columns: list = [],
                 # If the percentage of lost values in a column is greater than this threshold, the column will be removed
                 too_many_lost_values_threshold: float = 0.3,
                 # If the percentage of unique values in a categorical column is greater than this threshold, the column will be removed
                 too_many_categorical_value_threshold: float = 0.05,
                 numerical_scaling: str = "minmax",  # "standard", "min_max", "none"
                 verbose: bool = False):

        if numerical_scaling not in ["standard", "minmax", "none"]:
            raise ValueError("Invalid numerical scaling method. Choose 'standard', 'minmax' or 'none'.")

        self.target_column = target_column
        self.forced_dropped_columns = set(forced_dropped_columns)
        self.too_many_lost_values_threshold = too_many_lost_values_threshold
        self.too_many_categorical_value_threshold = too_many_categorical_value_threshold
        self.numerical_scaling = numerical_scaling
        self.verbose = verbose
        self.columns_to_drop = set()
        self.means = {}
        self.modes = {}
        self.encodings = {}
        self.numerical_scaling_params = {}

    ########################## AUXILIARY FUNCTIONS ##########################

    def __check_for_target_column(self, df: pd.DataFrame) -> None:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the DataFrame.")

    def __check_for_too_many_lost_values(self, df: pd.DataFrame, column: str) -> bool:
        result = df[column].isnull().sum() / df.shape[0] > self.too_many_lost_values_threshold
        if self.verbose and result:
            print(f" - Column '{column}': Has too many lost values. It will be removed.")
        return result

    def __check_for_one_unique_value(self, df: pd.DataFrame, column: str) -> bool:
        result = df[column].nunique() == 1
        if self.verbose and result:
            print(f" - Column '{column}': Has only one unique value. It will be removed.")
        return result

    def __check_for_column_in_forced_dropped_columns(self, column: str) -> bool:
        result = column in self.forced_dropped_columns
        if self.verbose and result:
            print(f" - Column '{column}': In the forced dropped columns list. It will be removed.")
        return result

    def __check_for_too_many_categorical_values(self, df: pd.DataFrame, column: str) -> bool:
        result = df[column].nunique() / df.shape[0] > self.too_many_categorical_value_threshold
        if self.verbose and result:
            print(f" - Column '{column}': Has too many categorical values. It will be removed.")
        return result

    ########################## DATA PREPROCESSING FUNCTIONS ##########################

    def fit(self, df: pd.DataFrame) -> None:

        self.__check_for_target_column(df)

        for column in df.columns:

            if column == self.target_column:
                continue

            if (
                    self.__check_for_too_many_lost_values(df, column) or
                    self.__check_for_one_unique_value(df, column) or
                    self.__check_for_column_in_forced_dropped_columns(column)
            ):
                self.columns_to_drop.add(column)
                continue

            if pd.api.types.is_numeric_dtype(df[column]):
                self.means[column] = df[column].mean()
                if self.numerical_scaling == "standard":
                    self.numerical_scaling_params[column] = {'mean': self.means[column], 'std': df[column].std()}
                    if self.verbose:
                        print(f" - Column '{column}': Numerical scaling set to standard. Mean: {self.means[column]}, Std: {df[column].std()}. Null values will be filled with {self.means[column]}.")
                elif self.numerical_scaling == "minmax":
                    self.numerical_scaling_params[column] = {'min': df[column].min(), 'max': df[column].max()}
                    if self.verbose:
                        print(f" - Column '{column}': Numerical scaling set to minmax. Min: {self.numerical_scaling_params[column]['min']}, Max: {self.numerical_scaling_params[column]['max']}. Null values will be filled with {self.means[column]}.")
                elif self.verbose:
                    print(f" - Column '{column}': Numerical scaling set to none. No scaling will be applied. Null values will be filled with {self.means[column]}.")

            else:
                if self.__check_for_too_many_categorical_values(df, column):
                    self.columns_to_drop.add(column)
                else:
                    self.modes[column] = df[column].mode()[0]
                    cat = df[column].astype("category")
                    self.encodings[column] = dict(enumerate(cat.cat.categories)) # Ordinal encoding
                    if self.verbose:
                        print(f" - Column '{column}': Ordinal encoding created with {len(self.encodings[column])} unique values. Null values will be filled with '{self.modes[column]}'.")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        # Remove dropped columns
        df = df.drop(columns=self.columns_to_drop, errors='ignore').copy()

        # Fill missing numerical values with means
        for col, val in self.means.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
                if self.numerical_scaling == 'minmax':
                    min_val = self.numerical_scaling_params[col]['min']
                    max_val = self.numerical_scaling_params[col]['max']
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.0
                elif self.numerical_scaling == 'standard':
                    mean_val = self.numerical_scaling_params[col]['mean']
                    std_val = self.numerical_scaling_params[col]['std']
                    if std_val != 0:
                        df[col] = (df[col] - mean_val) / std_val
                    else:
                        df[col] = 0.0

        # Fill missing categorical values with modes and apply ordinal encoding
        for col, val in self.modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
                inv_map = {v: k for k, v in self.encodings[col].items()}
                df[col] = df[col].astype(str).map(inv_map)

        # TODO: Normalize categorical columns if needed

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
