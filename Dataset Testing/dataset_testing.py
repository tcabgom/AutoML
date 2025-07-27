from sklearn.datasets import fetch_openml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from BasicAutoML.src.data_loader import DataLoader
from BasicAutoML.src.preprocessing import Preprocessor

DATASET_NAMES = ["adult", "titanic", "wine", "vehicle", "segment", "mushroom", "iris", "heart-statlog"]

def analyze_dataset():
    for name in DATASET_NAMES:
        print(f"\n\n\n###########   Fetching dataset: {name}   ###########")
        loader = DataLoader(name)
        X, y = loader.load_data()

        print(f"Shape: {X.shape}")
        print(f"Classes: {y.unique()}")

        df = X.copy()
        df['target'] = y

        preprocessor = Preprocessor(
            target_column='target',
            numerical_scaling='minmax',
            verbose=True
        )

        df_processed = preprocessor.fit_transform(df)

        X_processed = df_processed.drop(columns=['target'])
        y_processed = df_processed['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42
        )

        modelDTC = DecisionTreeClassifier()
        modelDTC.fit(X_train, y_train)
        y_predDTC = modelDTC.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_predDTC)
        if len(y_test.unique()) == 2:
            roc_auc_ovr = roc_auc_score(y_test, modelDTC.predict_proba(X_test)[:, 1])
        else:
            roc_auc_ovr = roc_auc_score(y_test, modelDTC.predict_proba(X_test), multi_class='ovr')
        print(f"\nBalanced accuracy on {name}: {bal_acc:.2f} for DTC")
        print(f"ROC AUC (OVR) on {name}: {roc_auc_ovr:.2f} for DTC\n")

        modelKNN = KNeighborsClassifier()
        modelKNN.fit(X_train, y_train)
        y_predKNN = modelKNN.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_predKNN)
        if len(y_test.unique()) == 2:
            roc_auc_ovr = roc_auc_score(y_test, modelKNN.predict_proba(X_test)[:, 1])
        else:
            roc_auc_ovr = roc_auc_score(y_test, modelKNN.predict_proba(X_test), multi_class='ovr')
        print(f"Balanced accuracy on {name}: {bal_acc:.2f} for KNN")
        print(f"ROC AUC (OVR) on {name}: {roc_auc_ovr:.2f} for KNN")

if __name__ == "__main__":
    analyze_dataset()

