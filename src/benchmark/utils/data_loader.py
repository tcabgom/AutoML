import openml


def load_benchmark_suite(suite_id: int) -> openml.study.OpenMLBenchmarkSuite:
    print(f"Loading benchmark suite with id {suite_id}...")
    return openml.study.get_suite(suite_id)

def load_task_dataset(task_id: int):
    print(f"Loading task with id {task_id}...")
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    all_train_indices = list()
    all_test_indices = list()
    for i in range(10):
        train_indices, test_indices = task.get_train_test_split_indices(i)
        all_train_indices.append(train_indices)
        all_test_indices.append(test_indices)
    return X, y, dataset, all_train_indices, all_test_indices

