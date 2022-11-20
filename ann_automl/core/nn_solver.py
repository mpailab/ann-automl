import os
import sys
import traceback
from copy import copy
from datetime import datetime
from pytz import timezone
from typing import Any, Callable, Dict, List

from .nnfuncs import nn_hparams, tune_hparams, get_hparams
from .solver import Task, set_log_dir, printlog, _log_dir, RecommendTask
from ..utils.process import pcall

TargetFunc = Callable[[List[float]], float]


def loss_target(scores: List[float]) -> float:
    return -scores[0]


def metric_target(scores: List[float]) -> float:
    return scores[1]


class NNTask(Task):
    """ Класс, задающий задачу на обучение нейронной сети """

    def __init__(self,
                 category: str = 'train',
                 objects: List[str] = (),
                 type: str = 'classification',
                 func: TargetFunc = metric_target,
                 # TODO: кажется, надо вернуть как было -- либо accuracy,
                 #       либо конкретный вид accuracy (categorial, binary, sparse_categorical, etc.)
                 target: float = 0.9,
                 goals: Dict[str, Any] = None):
        """
        Инициализация задачи.

        Parameters:
        ----------
        category: str
            Категория задачи:
            train - обучение модели,
            test - тестирование модели без предварительного обучения 
            "служебные" - {проверка наличия нужных классов,БД, моделей}
        type: str
            Тип задачи {classification, detection, segmentation}
        objects: list[str]
            Список категорий изображений
        func: TargetFunc
            Целевой функционал {loss_target, metric_target}
        target: str
            Значение целевого функционала {float from [0,1]}
        goal: dict[str, Any]
            Словарь целей задачи (например {"метрика" : желаемое значение метрики})
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


class SelectHParamsTask(RecommendTask):
    def __init__(self, nn_task: NNTask):
        super().__init__(goals={})
        self.nn_task = nn_task
        self.hparams = {param: nn_hparams[param]['default'] for param in nn_hparams}
        self.hparams['pipeline'] = None
        self.recommendations = {}

    def set_selected_options(self, options):
        self.hparams.update(options)


def recommend_hparams(task: NNTask, **kwargs) -> dict:
    """ Рекомендует гиперпараметры для задачи """
    htask = SelectHParamsTask(task)
    htask.solve(global_params=kwargs)
    return htask.hparams
