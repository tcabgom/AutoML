import pandas as pd


class Preprocessor:
    def __init__(self,
                 forced_dropped_columns: list = [],
                 # If the percentage of lost values in a column is greater than this threshold, the column will be removed
                 too_many_lost_values_threshold: float = 0.3,
                 # If the percentage of unique values in a categorical column is greater than this threshold, the column will be removed
                 too_many_categorical_value_threshold: float = 0.05,
                 numerical_scaling: str = "minmax",  # "standard", "min_max", "robustScaler", "none"
                 too_many_lost_values_bool_columns: bool = True,
                 verbose: bool = True):

        if numerical_scaling not in ["standard", "minmax", "robustScaler", "none"]:
            raise ValueError("Invalid numerical scaling method. Choose 'standard', 'minmax', 'robustScaler' or 'none'.")

        self.forced_dropped_columns = set(forced_dropped_columns)
        self.too_many_lost_values_threshold = too_many_lost_values_threshold
        self.too_many_categorical_value_threshold = too_many_categorical_value_threshold
        self.numerical_scaling = numerical_scaling
        self.too_many_lost_values_bool_columns = too_many_lost_values_bool_columns # TODO Implement
        self.verbose = verbose

        self.columns_to_drop = set()
        self.columns_to_replace_bool_lost_values = set()
        self.means = {}
        self.modes = {}
        self.encodings = {}
        self.numerical_scaling_params = {}

    ########################## AUXILIARY FUNCTIONS ##########################

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

        for column in df.columns:

            if (
                    self.__check_for_too_many_lost_values(df, column) or
                    self.__check_for_one_unique_value(df, column) or
                    self.__check_for_column_in_forced_dropped_columns(column)
            ):
                if (
                        self.too_many_lost_values_bool_columns and
                        self.__check_for_too_many_lost_values(df, column)
                ):
                    self.columns_to_replace_bool_lost_values.add(column)
                    if self.verbose:
                        print(f"    * A copy of the column '{column+'_missing'}' will be created to indicate the rows with missing values.")
                else:
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
                elif self.numerical_scaling == "robustScaler":
                    median_val = df[column].median()
                    q1_val = df[column].quantile(0.25)
                    q3_val = df[column].quantile(0.75)
                    iqr = q3_val - q1_val
                    self.numerical_scaling_params[column] = {'median': median_val, 'iqr': iqr}
                    if self.verbose:
                        print(f" - Column '{column}': Numerical scaling set to robustScaler. Median: {median_val}, IQR: {iqr}. Null values will be filled with {self.means[column]}.")
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

        if self.too_many_lost_values_bool_columns:
            for col in self.columns_to_replace_bool_lost_values:
                if col in df.columns:
                    df[col] = df[col].isnull().astype(int)
                    df.rename(columns={col: f"{col}_missing"}, inplace=True)

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
                elif self.numerical_scaling == 'robustScaler':
                    median_val = df[col].median()
                    q1_val = df[col].quantile(0.25)
                    q3_val = df[col].quantile(0.75)
                    iqr = q3_val - q1_val
                    if iqr != 0:
                        df[col] = (df[col] - median_val) / iqr
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
