import pandas as pd
from BasicAutoML.src.preprocessing import Preprocessor



if __name__ == "__main__":
    df = pd.read_csv("../../datasets/classification/titanic.csv", sep=",", encoding="utf-8")
    preprocessor = Preprocessor(target_column="Survived",
                                forced_dropped_columns=["PassengerId"],
                                too_many_lost_values_threshold=0.3,
                                too_many_categorical_value_threshold=0.05,
                                numerical_scaling="minmax",
                                verbose=True)

    preprocessed_df = preprocessor.fit_transform(df)
    print(preprocessed_df.head())
    for column in preprocessed_df.columns:
        print(f"{column}: {preprocessed_df[column].unique()}")
