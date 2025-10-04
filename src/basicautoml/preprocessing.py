import pandas as pd
import warnings

class Preprocessor:
    def __init__(self,
                 # Drop column variables
                 forced_dropped_columns: list = [],
                 too_many_lost_values_threshold: float = 0.3,
                 too_many_categorical_value_threshold: float = 0.05,

                 # Preprocessing variables
                 numerical_scaling: str = "robustScaler",   # "standard", "minmax", "robustScaler", "none"
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

        self.forced_dropped_columns = set(forced_dropped_columns)
        self.too_many_lost_values_threshold = too_many_lost_values_threshold
        self.too_many_categorical_value_threshold = too_many_categorical_value_threshold
        self.numerical_scaling = numerical_scaling
        self.too_many_lost_values_bool_columns = too_many_lost_values_bool_columns
        self.categorical_encoding = categorical_encoding
        self.max_one_hot_unique = max_one_hot_unique
        self.rare_category_threshold = rare_category_threshold
        self.verbose = verbose

        self.columns_to_drop = set()
        self.columns_to_replace_bool_lost_values = set()

        self.means = {}
        self.numerical_scaling_params = {}

        self.modes = {}
        self.categorical_strategy = {}
        self.encodings = {}
        self.onehot_columns = {}


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

    def __decide_categorical_encoding(self, df: pd.DataFrame, column: str) -> str:
        if self.categorical_encoding == "auto":
            if df[column].nunique() <= self.max_one_hot_unique:
                return "onehot"
            else:
                return "ordinal"
        else:
            return self.categorical_encoding


    def __fit_ordinal_encoding(self, df: pd.DataFrame, column: str) -> None:
        cat = df[column].astype("category")
        mapping = {str(cat_val): idx for idx, cat_val in enumerate(cat.cat.categories)}
        self.encodings[column] = mapping
        if self.verbose:
            print(f" - Column '{column}': Ordinal encoding created with {len(mapping)} unique values. Null values will be filled with '{self.modes[column]}'.")

    def __fit_onehot_encoding(self, df: pd.DataFrame, column: str) -> None:
        if self.rare_category_threshold > 0:
            freqs = df[column].value_counts(normalize=True, dropna=True)
            top = freqs[freqs >= self.rare_category_threshold].index.tolist()
            categories = top + (["OTHER"] if len(top) < freqs.shape[0] else [])
        else:
            categories = df[column].dropna().unique().tolist()

        self.onehot_columns[column] = [f"{column}__{str(cat)}" for cat in categories]
        self.encodings[column] = set(str(c) for c in categories)
        if self.verbose:
            print(f" - Column '{column}': One-hot encoding created with {len(categories)} unique values. Null values will be filled with '{self.modes[column]}'.")


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
                # CATEGORICAL COLUMN
                if self.__check_for_too_many_categorical_values(df, column):
                    self.columns_to_drop.add(column)
                else:
                    self.modes[column] = str(df[column].mode()[0])

                    categorical_strategy = self.__decide_categorical_encoding(df, column)
                    self.categorical_strategy[column] = categorical_strategy

                    if categorical_strategy == "ordinal":
                        self.__fit_ordinal_encoding(df, column)
                    else:
                        self.__fit_onehot_encoding(df, column)




    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        #print(self.onehot_columns) # TODO Remove
        #print(self.categorical_strategy)
        #print(self.encodings)
        #print("\n\n\n")
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
                    median_val = self.numerical_scaling_params[col]['median']
                    iqr = self.numerical_scaling_params[col]['iqr']
                    if iqr != 0: # TODO Check if this is the best approach
                        df[col] = (df[col] - median_val) / iqr
                    else:
                        df[col] = 0.0

        # Fill missing categorical values with modes and apply encoding
        for col, mapping in self.modes.items():
            if col not in df.columns:
                continue

            if self.categorical_strategy[col] == "ordinal":
                df[col] = df[col].fillna(self.modes[col]).astype(str)
                df[col] = df[col].map(self.encodings[col]).fillna(-1).astype("int64")  # Unseen categories mapped to -1


            else:
                for oh_col, dummies in self.onehot_columns.items():
                    if oh_col not in df.columns:
                        for d in dummies:
                            df[d] = 0
                        continue

                    series = df[oh_col].fillna(self.modes.get(oh_col)).astype(str)
                    allowed_categories = self.encodings[oh_col]
                    if self.rare_category_threshold > 0 and ("OTHER" in allowed_categories):
                        series = series.map(lambda x: x if x in allowed_categories else "OTHER")

                    dummies_df = pd.get_dummies(series, prefix=oh_col, prefix_sep="__", dummy_na=False).astype("int8")
                    for d in dummies:
                        if d not in dummies_df.columns:
                            dummies_df[d] = 0
                    dummies_df = dummies_df.reindex(columns=dummies)
                    df = df.drop(columns=[oh_col])
                    df = pd.concat([df, dummies_df], axis=1)


        # TODO: Normalize categorical columns if needed

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
