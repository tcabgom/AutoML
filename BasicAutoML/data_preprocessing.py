import pandas as pd
import numpy as np

# If the percentage of lost values in a column is greater than this threshold, the column will be removed
TOO_MANY_LOST_VALUES_THRESHOLD = 0.3
# If the percentage of unique values in a categorical column is greater than this threshold, the column will be removed
TOO_MANY_CATEGORIC_VALUE_THRESHOLD = 0.05


class DataPreprocessor:

    def __init__(self, data):
        self.data = data


    ########   DATA PREPROCESSING FUNCTIONS   ########

    def __column_is_numerical(self, column):
        return pd.api.types.is_numeric_dtype(self.data[column])

    def __replace_null_values_with_mean(self, column):
        self.data[column].fillna(self.data[column].mean(), inplace=True)

    def __replace_null_value_with_mode(self, column):
        self.data[column].fillna(self.data[column].mode()[0], inplace=True)

    def __column_matches_index(self, column):
        return False # TODO

    def __has_too_many_lost_values(self, column):
        return self.data[column].isnull().sum() / self.data.shape[0] > TOO_MANY_LOST_VALUES_THRESHOLD

    def __has_too_many_categoric_values(self, column):
        return self.data[column].nunique() / self.data.shape[0] > TOO_MANY_CATEGORIC_VALUE_THRESHOLD


    def __column_has_one_unique_value(self, column):
        return self.data[column].nunique() == 1


    def __ordinal_encode(self, column):
        self.data[column] = self.data[column].astype("category").cat.codes

    def __remove_column(self, column):
        self.data.drop(column, axis=1, inplace=True)

    def __column_preprocessing(self, column):
        if self.__has_too_many_lost_values(column) or self.__column_has_one_unique_value(column):
            print(f"The column {column} has too many lost values or only one unique value, and will be removed")
            self.__remove_column(column)
        else:
            if self.__column_is_numerical(column):
                if self.__column_matches_index(column):
                    print(f"The column {column} is just the index, and will be removed")
                    self.__remove_column(column)
                else:
                    self.__replace_null_values_with_mean(column)
            else:
                if self.__has_too_many_categoric_values(column):
                    print(f"The column {column} has too many categoric values, and will be removed")
                    self.__remove_column(column)
                else:
                    self.__replace_null_value_with_mode(column)
                    self.__ordinal_encode(column)

    def preprocess_data(self):
        """
        Preprocess the data

        :return: Preprocessed data
        """
        for column in self.data.columns:
            self.__column_preprocessing(column)
        return self.data

