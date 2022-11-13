import abc
import itertools
import math
import sys
import time
from collections import defaultdict
from functools import reduce
import random
from copy import copy
from typing import Any

import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import os
from . import db_module
from datetime import datetime
from pytz import timezone

from .hw_devices import tf_devices_memory
from .nnfuncs import nn_hparams, nnDB, _data_dir, create_model, create_generators, fit_model, create_data_subset
from .solver import Rule, rule, Task, printlog, SolverState, Recommender
from ..utils.process import request, NoHandlerError
from .nn_solver import NNTask, SelectHParamsTask, metric_target


def find_zero_neighbor(center, table, radius=1):
    ranges = [range(max(0, center[i] - radius), min(szi, center[i] + radius + 1)) for i, szi in enumerate(table.shape)]
    for i in itertools.product(*ranges):
        if abs(table[i]) <= 1e-6:
            return list(i)
    return None


# class Recommender(Rule, abc.ABC):
#     def __init__(self):
#         self.key = self.__class__.__name__
#
#     def can_recommend(self, task) -> bool:
#         return True
#
#     def can_apply(self, task, state: SolverState) -> bool:
#         return self.key not in task.recommendations and self.can_recommend(task)


# Базовые приёмы для начальной рекомендации гиперпараметров
@rule(SelectHParamsTask)
class RecommendLoss(Recommender):
    def apply(self, task: SelectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        # TODO: проверить и добавить рекомендации для разных типов задач
        if len(task.nn_task.objects) == 2:
            prec['loss'] = 'binary_crossentropy'
        else:
            prec['loss'] = 'categorical_crossentropy'


@rule(SelectHParamsTask)
class RecommendOptimizer(Recommender):
    def apply(self, task: SelectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        # TODO: проверить и добавить рекомендации для разных типов задач
        prec['optimizer'] = 'SGD'
        prec['learning_rate'] = 0.01
        prec['nesterov'] = True


@rule(SelectHParamsTask)
class RecommendArch(Recommender):
    def apply(self, task: SelectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        prec['activation'] = 'relu'    # функция активации в базовой части. TODO: возможны ли другие рекомендации?
        prec['pipeline'] = 'ResNet18'  # TODO: давать рекоммендации в зависимости от типа задачи и размера входных данных
        last_layers = []
        if len(task.nn_task.objects) == 2:
            last_layers.append({'type': 'Dense', 'units': 1})
            last_layers.append({'type': 'Activation', 'activation': 'sigmoid'})
        else:
            last_layers.append({'type': 'Dense', 'units': len(task.nn_task.objects)})
            last_layers.append({'type': 'Activation', 'activation': 'softmax'})
        prec['last_layers'] = last_layers


@rule(SelectHParamsTask)
class RecommendAugmentation(Recommender):
    def apply(self, task: SelectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        aug = prec['augmen_params'] = {}
        if task.nn_task.input_type == 'image':
            aug.update({'preprocessing_function': 'keras.applications.resnet.preprocess_input',
                        'vertical_flip': None,
                        'width_shift_range': 0.4,
                        'height_shift_range': 0.4,
                        'horizontal_flip': (task.nn_task.object_category != 'symbol') or None,
                        })


# Могут добавляться или подгружаться извне и другие приёмы для рекомендации гиперпараметров.
# Например, на основе анализа данных, пробных запусков и т.д.


def estimate_batch_size(model: keras.Model, precision: int = 4) -> int:
    """ Оценка размера батча, который может поместиться в память """
    weight_multiplier = 3
    mem = min(tf_devices_memory())  # batch size is limited by the smallest device
    num_params = sum(np.prod(layer.output_shape[1:]) for layer in model.layers) + \
                 sum(keras.backend.count_params(x) for x in model.trainable_weights) + \
                 sum(keras.backend.count_params(x) for x in model.non_trainable_weights)
    max_size = int(mem / (precision * num_params * weight_multiplier))
    return int(2 ** math.floor(math.log(max_size, 2)))


@rule(SelectHParamsTask)
class RecommendBatchSize(Recommender):
    def can_recommend(self, task: SelectHParamsTask):
        return getattr(task.nn_task, 'model', None) is not None

    def apply(self, task: SelectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        prec['batch_size'] = estimate_batch_size(task.nn_task.model)  # recommends max possible batch size


def get_history(task: NNTask, exact_category_match=False):
    """ Получает рекомендации из истории запусков """
    ch_res = nnDB.get_models_by_filter({
        'min_metrics': {task.metric: task.target},
        'task_type': task.type,
        'categories': list(task.objects)},
        exact_category_match=exact_category_match)
    n = len(ch_res.index)
    candidates = []
    for i in range(n):
        hist_file = ch_res.iloc[i]['history_address']  # csv-файл с историей запусков
        if os.path.exists(hist_file):  # add all rows from history to candidates
            hist = pd.read_csv(hist_file)
            nrows = hist.shape[0]
            for row in range(nrows):
                candidates.append(hist.iloc[row].to_dict())
    return candidates


@rule(SelectHParamsTask)
class RecommendFromHistory(Recommender):
    """ Предлагает параметры, которые были использованы в предыдущих
        аналогичных запусках (может рекомендовать любые параметры)
    """
    def apply(self, task: SelectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        candidates = get_history(task.nn_task)
        if len(candidates) > 0:
            prec.update(random.choice(candidates))

