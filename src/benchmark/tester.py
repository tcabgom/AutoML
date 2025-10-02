
from src.benchmark.utils.data_loader import load_benchmark_suite, load_task_dataset


def test():
    suite = load_benchmark_suite(271)
    print("Finished")
    for task_id in suite.tasks:
        x, y, dataset, train_indices, test_indices = load_task_dataset(task_id)

        X_train, y_train = x.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = x.iloc[test_indices], y.iloc[test_indices]

        print(len(X_train), len(X_test))

if __name__ == "__main__":
    test()
