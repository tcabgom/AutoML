import pandas as pd
import warnings
from typing import Dict, Set

from .utils.categorical_encoder import CategoricalEncoder
from .utils.numeric_scaler import NumericScaler

class Preprocessor:
    def __init__(self,
                 # Drop column variables
                 forced_dropped_columns: list = [],
                 too_many_lost_values_threshold: float = 0.3,
                 too_many_categorical_value_threshold: float = 0.05,

                 # Preprocessing variables
                 numerical_scaling: str = "standard",   # "standard", "minmax", "robustScaler", "none"
                 too_many_lost_values_bool_columns: bool = True,
                 categorical_encoding: str = "auto",  # "auto", "ordinal", "onehot"
                 max_one_hot_unique: int = 5,
                 rare_category_threshold: float = 0.01,

                 verbose: bool = True):
        """
        Preprocess the dataframe by handling missing values, encoding categorical variables, and scaling numerical features.

        :param forced_dropped_columns: List of columns to be forcibly dropped.
        :param too_many_lost_values_threshold: If the percentage of lost values in a column is greater than this threshold, the column will be removed.
        :param too_many_categorical_value_threshold: If the percentage of unique values in a categorical column is greater than this threshold, the column will be removed.
        :param numerical_scaling: The method to scale numerical features. Options are "standard", "minmax", "robustScaler", or "none".
        :param too_many_lost_values_bool_columns: Whether to replace columns with too many lost values with boolean indicators.
        :param categorical_encoding: The method to encode categorical variables. Options are "auto", "ordinal" or "onehot".
        :param max_one_hot_unique: Maximum number of unique values in a categorical column to apply one-hot encoding when categorical_encoding is set to "auto".
        :param rare_category_threshold: Categories with a frequency lower than this threshold will be grouped into a single 'rare' category when using one-hot encoding. Can be disabled by setting to 0.
        :param verbose: Whether to print detailed logs during preprocessing.
        """

        warnings.filterwarnings("ignore", message="Downcasting object dtype arrays")
        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

        if numerical_scaling not in ["standard", "minmax", "robustScaler", "none"]:
            raise ValueError("Invalid numerical scaling method. Choose 'standard', 'minmax', 'robustScaler' or 'none'.")

        if not (0.0 <= too_many_lost_values_threshold <= 1.0):
            raise ValueError("too_many_lost_values_threshold must be between 0 and 1.")

        if not (0.0 <= too_many_categorical_value_threshold <= 1.0):
            raise ValueError("too_many_categorical_value_threshold must be between 0 and 1.")

        if categorical_encoding not in ["auto", "ordinal", "onehot"]:
            raise ValueError("Invalid categorical encoding method. Choose 'auto', 'ordinal' or 'onehot'.")

        if not (0.0 <= rare_category_threshold <= 1.0):
            raise ValueError("rare_category_threshold must be between 0 and 1.")

        self.forced_dropped_columns: Set[str] = set(forced_dropped_columns)
        self.too_many_lost_values_threshold = too_many_lost_values_threshold
        self.too_many_categorical_value_threshold = too_many_categorical_value_threshold
        self.numerical_scaling = numerical_scaling
        self.too_many_lost_values_bool_columns = too_many_lost_values_bool_columns
        self.categorical_encoding = categorical_encoding
        self.max_one_hot_unique = max_one_hot_unique
        self.rare_category_threshold = rare_category_threshold
        self.verbose = verbose

        self.columns_to_drop: Set[str] = set()
        self.columns_to_replace_bool_lost_values: Set[str] = set()

        self.means: Dict[str, float] = {}
        self.numerical_scaling_params = {}
        self.numeric_scaler = NumericScaler(method=numerical_scaling)

        """self.modes: Dict[str, str] = {}
        self.categorical_strategy = {}
        self.encodings = {}
        self.onehot_columns = {}"""
        self.categorical_params: Dict[str, dict] = {}
        self.categorical_encoder = CategoricalEncoder(
            method=categorical_encoding,
            max_one_hot_unique=max_one_hot_unique,
            rare_category_threshold=rare_category_threshold,
            verbose=verbose
        )


    ########################## AUXILIARY FUNCTIONS ##########################

    def __check_for_too_many_lost_values(self, df: pd.DataFrame, column: str) -> bool:
        if df.shape[0] == 0:
            return False
        result = df[column].isnull().sum() / df.shape[0] > self.too_many_lost_values_threshold
        if self.verbose and result:
            print(f" - Column '{column}': Has too many lost values. It will be removed.")
        return result

    def __check_for_one_unique_value(self, df: pd.DataFrame, column: str) -> bool:
        result = (len(df[column].dropna().unique()) <= 1)
        if self.verbose and result:
            print(f" - Column '{column}': Has only one unique value. It will be removed.")
        return result

    def __check_for_column_in_forced_dropped_columns(self, column: str) -> bool:
        result = column in self.forced_dropped_columns
        if self.verbose and result:
            print(f" - Column '{column}': In the forced dropped columns list. It will be removed.")
        return result

    def __check_for_too_many_categorical_values(self, df: pd.DataFrame, column: str) -> bool:
        if df.shape[0] == 0:
            return False
        result = df[column].dropna().nunique() / df.shape[0] > self.too_many_categorical_value_threshold
        if self.verbose and result:
            print(f" - Column '{column}': Has too many categorical values. It will be removed.")
        return result

    ########################## DATA PREPROCESSING FUNCTIONS ##########################

    def fit(self, df: pd.DataFrame) -> None:

        for column in df.columns:

            if (
                    self.__check_for_column_in_forced_dropped_columns(column) or
                    self.__check_for_one_unique_value(df, column)
            ):
                self.columns_to_drop.add(column)
                continue

            if self.__check_for_too_many_lost_values(df, column):
                if self.too_many_lost_values_bool_columns:
                    self.columns_to_replace_bool_lost_values.add(column)
                    if self.verbose:
                        print(f"    * A copy of the column '{column+'_missing'}' will be created to indicate the rows with missing values.")
                else:
                    self.columns_to_drop.add(column)
                continue

            if pd.api.types.is_numeric_dtype(df[column]):
                # NUMERICAL COLUMN
                self.means[column] = df[column].mean()

                scaler = NumericScaler(self.numerical_scaling)
                params = scaler.fit_column(df[column])
                self.numerical_scaling_params[column] = params

                if self.verbose:
                    if self.numerical_scaling == "standard":
                        print(f" - Column '{column}': Numerical scaling set to standard. Mean: {params.get('mean')}, Std: {params.get('std')}. ({df[column].isna().sum()} null values will be filled with {self.means[column]}).")
                    elif self.numerical_scaling == "minmax":
                        print(f" - Column '{column}': Numerical scaling set to minmax. Min: {params.get('min')}, Max: {params.get('max')}. ({df[column].isna().sum()} null values will be filled with {self.means[column]}).")
                    elif self.numerical_scaling == "robustScaler":
                        print(f" - Column '{column}': Numerical scaling set to robustScaler. Median: {params.get('median')}, IQR: {params.get('iqr')}. ({df[column].isna().sum()} null values will be filled with {self.means[column]}).")
                    else:
                        print(f" - Column '{column}': Numerical scaling set to none. No scaling will be applied. ({df[column].isna().sum()} null values will be filled with {self.means[column]}).")

            else:
                # CATEGORICAL COLUMN
                if self.__check_for_too_many_categorical_values(df, column):
                    self.columns_to_drop.add(column)
                else:
                    params = self.categorical_encoder.fit_column(df[column], column_name=column)
                    self.categorical_params[column] = params




    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        if self.too_many_lost_values_bool_columns:
            for col in self.columns_to_replace_bool_lost_values:
                if col in df.columns:
                    df[col] = df[col].isnull().astype(int)
                    df.rename(columns={col: f"{col}_missing"}, inplace=True)

        # Remove dropped columns
        if len(self.columns_to_drop) > 0:
            df = df.drop(columns=self.columns_to_drop, errors='ignore').copy()

        # Fill missing numerical values with means
        for col, val in self.means.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
                scaler_params = self.numerical_scaling_params.get(col)
                df[col] = self.numeric_scaler.transform_column(df[col], scaler_params)

        # Fill missing categorical values with modes and apply encoding
        onehot_to_concat = []
        onehot_new_cols: Set[str] = set()

        for col, params in self.categorical_params.items():
            if params.get('strategy') == 'ordinal':
                if col not in df.columns:
                    continue
                df[col] = self.categorical_encoder.transform_column(df[col], params, column_name=col)

            else:
                expected_cols = params.get('onehot_columns', [])
                if col not in df.columns:
                    for c in expected_cols:
                        if c not in df.columns:
                            df[c] = 0
                    continue

                dummies_df = self.categorical_encoder.transform_column(df[col], params, column_name=col)
                df = df.drop(columns=[col])
                onehot_to_concat.append(dummies_df)
                onehot_new_cols.update(dummies_df.columns.tolist())

        if onehot_to_concat:
            df = pd.concat([df] + onehot_to_concat, axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
