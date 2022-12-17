from typing import Any, Callable, Dict, List

from .solver import Task, set_log_dir, printlog, _log_dir, RecommendTask

TargetFunc = Callable[[List[float]], float]


def loss_target(scores: List[float]) -> float:
    """:meta private:"""
    return -scores[0]


def metric_target(scores: List[float]) -> float:
    """:meta private:"""
    return scores[1]


class NNTask(Task):
    """ Класс, задающий задачу на обучение нейронной сети """

    def __init__(self,
                 category='train',
                 objects=(),
                 type='classification',
                 func=metric_target,
                 # TODO: кажется, надо вернуть как было -- либо accuracy,
                 #       либо конкретный вид accuracy (categorial, binary, sparse_categorical, etc.)
                 target=0.9,
                 goals=None,
                 time_limit=60 * 60 * 24,
                 for_mobile=False,
                 ):
        """
        Инициализация задачи.

        Parameters:
        -----------
        category: str
            Категория задачи: пока поддерживается только 'train'
        type: str
            Тип задачи {classification, detection, segmentation}
        objects: list[str]
            Список категорий изображений
        func: TargetFunc
            Целевой функционал {loss_target, metric_target}
        target: str
            Значение целевого функционала {float from [0,1]}
        goal: dict[str, Any]
            Словарь дополнительных целей задачи (например {"maximize" :
            нужно ли максимизировать метрику выше целевого значения})
        time_limit: float
            Время на выполнение задачи в секундах
        for_mobile: bool
            Флаг, указывающий, что предполагается использование модели на мобильных устройствах
        """
        super().__init__(goals=goals or {})
        self._category = category  # str

        # Необязательные входные параметры задачи, нужны при категориях "train" и "test"
        self._type = type  # str - тип задачи {classification, detection, segmentation}
        self._objects = objects or []  # set of strings - набор категорий объектов интереса
        self.input_type = 'image'  # тип объектов задачи (image, video, audio, text)
        self.object_category = 'object'  # категория объектов задачи (object, symbol)
        self.func = func
        self.target = target
        self.log_name = ''
        self.time_limit = time_limit
        self.for_mobile = for_mobile

    @property
    def category(self):
        """ Возвращает категорию задачи """
        return self._category

    @property
    def type(self):
        """ Возвращает тип задачи """
        return self._type

    @property
    def objects(self):
        """ Возвращает набор категорий объектов интереса """
        return self._objects

    @property
    def metric(self):
        """ Возвращает название целевого функционала """
        return 'accuracy' if self.func.__name__ == 'metric_target' else self.func.__name__
        # TODO: это пока костыль для поиска по истории (не хорошо передавать имя функции в качестве метрики)

