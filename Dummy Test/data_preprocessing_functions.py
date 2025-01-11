import pandas as pd

def read_csv(file, sep):
    return pd.read_csv(file, sep=sep, encoding="utf-8")


def data_info(data):
    return data.info()


def show_head(data):
    return data.head()

def check_for_null(data):
    return data.isnull().sum()


def remove_row_with_null_values(data, column):
    return data.dropna(subset=[column])


def set_null_values_as_mean(data, column):
    data[column].fillna(data[column].mean(), inplace=True)
    return data


def set_null_values_as_most_common(data, column):
    return data.fillna(data[column].value_counts().idxmax(), inplace=True)


def one_hot_encode(data, column):
    return pd.get_dummies(data, columns=[column])


def ordinal_encode(data, column):
    return data[column].astype("category").cat.codes

