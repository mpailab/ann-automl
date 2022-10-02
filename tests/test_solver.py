from ann_automl.core.nn_solver import solve, Task, set_log_dir, set_data_dir


def test_solve():
    set_data_dir('data')
    set_log_dir('log')
    # Создаём и решаем задачу создания модели нейронной сети
    task = Task("train", task_type="classification", obj_set={"cat", "dog"}, goal={'accuracy': 0.9})
    w = solve(task, debug_mode=True)


if __name__ == '__main__':
    test_solve()
