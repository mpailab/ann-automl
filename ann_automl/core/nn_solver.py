from .solver import Task, set_log_dir, solve
from .nn_rules import set_data_dir


def create_nn_task(category, task_type, obj_set, target_metric, target_value):
    """
    Создаёт задачу обучения нейронной сети.

    Parameters:
    ----------
    category: str
        Категория задачи: обучение модели "train", тестирование модели без предварительного обучения "test"
        "служебные" = {проверка наличия нужных классов,БД, моделей}
    task_type: str
        Тип задачи {classification, detection, segmentation}
    obj_set: list[str]
        Набор категорий объектов интереса
    target_metric: str
        Целевая метрика
    target_value: float
        Целевое значение целевой метрики

    Returns:
    -------
    task: Task
        Задача
    """
    return Task(category, task_type, obj_set, goal={target_metric: target_value})

