import abc
import itertools
import math
import sys
import time
from collections import defaultdict
from functools import reduce
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
from .nnfuncs import hyperparameters, nnDB, _data_dir, create_model, create_generators, fit_model, create_data_subset
from .solver import Rule, rule, Task, printlog, SolverState
from ..utils.process import request, NoHandlerError
from .nn_solver import NNTask


def find_zero_neighbor(center, table, radius=1):
    ranges = [range(max(0, center[i] - radius), min(szi, center[i] + radius + 1)) for i, szi in enumerate(table.shape)]
    for i in itertools.product(*ranges):
        if abs(table[i]) <= 1e-6:
            return list(i)
    return None


class SetectHParamsTask(Task):
    def __init__(self, nn_task: NNTask):
        super().__init__(goals={})
        self.nn_task = nn_task
        self.hparams = {param: hyperparameters[param]['default'] for param in hyperparameters}
        self.hparams['pipeline'] = None
        self.recommendations = {}


class Recommender(Rule, abc.ABC):
    def __init__(self):
        self.key = self.__class__.__name__

    def can_recommend(self, task) -> bool:
        return True

    def can_apply(self, task, state: SolverState) -> bool:
        return self.key not in task.recommendations and self.can_recommend(task)


# Базовые приёмы для начальной рекомендации гиперпараметров
@rule(SetectHParamsTask)
class RecommendLoss(Recommender):
    def apply(self, task: SetectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        # TODO: проверить и добавить рекомендации для разных типов задач
        if len(task.nn_task.objects) == 2:
            prec['loss'] = 'binary_crossentropy'
        else:
            prec['loss'] = 'categorical_crossentropy'


@rule(SetectHParamsTask)
class RecommendOptimizer(Recommender):
    def apply(self, task: SetectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        # TODO: проверить и добавить рекомендации для разных типов задач
        if 'accuracy' in task.nn_task.goals:
            prec['optimizer'] = 'Adam'
            prec['learning_rate'] = 0.001
        else:
            prec['optimizer'] = 'SGD'
            prec['learning_rate'] = 0.01
            prec['nesterov'] = True


@rule(SetectHParamsTask)
class RecommendArch(Recommender):
    def apply(self, task: SetectHParamsTask, state: SolverState):
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


@rule(SetectHParamsTask)
class RecommendAugmentation(Recommender):
    def apply(self, task: SetectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        aug = prec['augmen_params'] = {}
        if task.nn_task.input_type == 'image':
            aug.update({'preprocessing_function': 'keras.applications.resnet.preprocess_input',
                        'vertical_flip': None,
                        'width_shift_range': 0.4,
                        'height_shift_range': 0.4,
                        'horizontal_flip': (task.nn_task.object_category != 'symbol') or None,
                        })


@rule(SetectHParamsTask)
class RecommendFromHistory(Recommender):
    """ Предлагает параметры, которые были использованы в предыдущих
        аналогичных запусках (может рекомендовать любые параметры)
    """
    def apply(self, task: SetectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        # TODO: implement


# Могут добавляться или подгружаться извне и другие приёмы для рекомендации гиперпараметров.
# Например, на основе анализа данных, пробных запусков и т.д.


def estimate_batch_size(model: keras.Model, precision: int = 4) -> int:
    """ Оценка размера батча, который может поместиться в память """
    weight_multiplier = 5
    mem = min(tf_devices_memory())  # batch size is limited by the smallest device
    num_params = sum(np.prod(layer.output_shape[1:]) for layer in model.layers) + \
                 sum(keras.backend.count_params(x) for x in model.trainable_weights) + \
                 sum(keras.backend.count_params(x) for x in model.non_trainable_weights)
    max_size = int(mem / (precision * num_params * weight_multiplier))
    return int(2 ** math.floor(math.log(max_size, 2)))


@rule(SetectHParamsTask)
class RecommendBatchSize(Recommender):
    def can_recommend(self, task: SetectHParamsTask):
        return getattr(task.nn_task, 'model', None) is not None

    def apply(self, task: SetectHParamsTask, state: SolverState):
        prec = task.recommendations[self.key] = {}
        prec['batch_size'] = estimate_batch_size(task.nn_task.model)  # recommends max possible batch size


@rule(NNTask)
class CheckSuitableModelExistence(Rule):
    """ Приём для поиска готовой подходящей модели в базе данных """

    def can_apply(self, task: NNTask, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'FirstCheck'

    def apply(self, task: NNTask, solver_state: SolverState):
        ch_res = nnDB.get_models_by_filter({
            'min_metrics': task.goals,
            'task_type': task.task_type,
            'categories': list(task.objects)})
        # print(ch_res)

        s = []
        if ch_res.empty:
            printlog('Не найдено подходящих моделей')
            #######################################################################
            ############ ПРИМЕР ВЗАИМОДЕЙСТВИЯ С ПОЛЬЗОВАТЕЛЕМ ####################
            #######################################################################

            # Запрос у пользователя: как будем выбирать гиперпараметры?
            # Формат запроса:
            #     request('choose_hyperparameters', current_hyperparameters, history_available, params_to_choose)
            #     current_hyperparameters - текущие гиперпараметры
            #     history_available - доступна ли история обучения
            #     params_to_choose - список параметров, которые нужно задать
            # 1. Задать вручную:
            #    ожидается на выходе: ('manual', словарь с гиперпараметрами)
            # 2. Подобрать автоматически исходя из истории
            #    ожидается на выходе: ('from_history', None)
            # 3. Поиск по сетке
            #    ожидается на выходе: ('grid_search', словарь с параметрами поиска по сетке)
            try:
                params_to_choose = ['learning_rate', 'batch_size', 'epochs', 'optimizer', 'loss']
                decision, params = request('choose_hyperparameters', {}, True, params_to_choose)  # TODO: заполнить current_hyperparameters и history_available
                if decision == 'manual':
                    printlog('Выбрано задание гиперпараметров вручную')
                    printlog("Заданы следующие гиперпараметры:")
                    for key, value in params.items():
                        printlog(f"{key}: {value}")
                    task.cur_state = 'Done'  # TODO: Replace done with actual task
                elif decision == 'from_history':
                    printlog('Подбор гиперпараметров по истории (пока не реализовано)')
                    task.cur_state = 'Done'  # TODO: Replace done with actual task
                elif decision == 'grid_search':
                    printlog('Поиск по сетке')
                    task.cur_state = 'DB'  # TODO: Replace done with actual task
                else:
                    printlog('Неверный ввод', file=sys.stderr)
                    task.cur_state = 'Done'
            except NoHandlerError:  # Если запущено не в режиме взаимодействия с пользователем
                task.cur_state = 'DB'
            #######################################################################
        else:
            n = len(ch_res.index)
            for i in range(n):
                s.append({'model_address': ch_res.iloc[i]['model_address'],
                          list(task.goals.keys())[0]: ch_res.iloc[i]['metric_value']})

            task.cur_state = 'UserDec'
            task.message = 'There is a suitable model in our base of trained models. Would you like to train an another model? "No" - 0; "Yes" - 1. '
            task.actions = {'0': 'Done', '1': 'DB'}
        task.suitModels = s


@rule(NNTask)
class UserDec(Rule):
    """ Приём решения пользователя о необходимости обучения новой модели """
    def can_apply(self, task):
        return task.task_ct == "train" and task.cur_state == 'UserDec'

    def apply(self, task):
        printlog('\n' + task.message)
        answer = input()
        task.cur_state = task.actions[str(answer)]

        with open(task.log_name, 'a') as log_file:
            print('UserDec Rule', file=log_file)
            print(task.message + ' ' + answer, file=log_file)

        if task.cur_state == 'Done':

            # считаем, что сюда пришли только сразу после firstCheck => сохрани модели и всё
            # или после создания сетки, но до первого обучения [ошибки не предусмотрены] => сохрани историю,
            # или после хотя бы 1 обучения, т.е. существует best_model, если модель подходит до
            if hasattr(task, 'best_model'):
                loc = task.history.loc[task.history['Index'] == task.best_num]
                prefix = f"{_data_dir}/trainedNN/{loc['exp_name'].iloc[0]}/{loc['train_subdir'].iloc[0]}/{loc['train_subdir'].iloc[0]}"
                nnDB.add_model_record(task_type=task.task_type, categories=list(task.objects),
                                      model_address=prefix + '__Model.h5',
                                      metrics={list(task.goal.keys())[0]: loc['metric_test_value'].iloc[0]},
                                      history_address=prefix + 'History.h5')

        task.message = None
        task.actions = None


####################################################


@rule(NNTask)
class CreateDatabase(Rule):
    """ Приём для создания обучающей выборки """
    def can_apply(self, task: NNTask, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'DB'

    def apply(self, task: Any, solver_state: SolverState):
        with open(task.log_name, 'a') as log_file:
            print('CreateDatabase', file=log_file)

        task.data = create_data_subset(task.objects)
        task.cur_state = 'Training'


"""Hyper Tuning Grid Block"""


@rule(NNTask)
class SetGrid(Rule):
    """создать общую сетку параметров и инфраструктуру всего эксперимента"""

    def can_apply(self, task, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'Training'

    def apply(self, task, solver_state: SolverState):

        with open(task.log_name, 'a') as log_file:
            print('SetGrid', file=log_file)

        '''tmp'''
        task.fixed_hparams = {'pipeline': 'ResNet18',
                              'last_layers': [{'Type': 'Dense', 'units': 64},
                                              {'Type': 'Activation', 'activation': 'relu'},
                                              {'Type': 'Dense', 'units': 1},
                                              {'Type': 'Activation', 'activation': 'sigmoid'}],
                              'augmenParams': {'horizontal_flip': True,
                                               'preprocessing_function': 'keras.applications.resnet.preprocess_input',
                                               'vertical_flip': None,
                                               'width_shift_range': 0.4,
                                               'height_shift_range': 0.4},
                              'loss': 'binary_crossentropy',
                              'metrics': 'accuracy',
                              'epochs': 150,
                              'stop_criterion': 'stop_val_metr'}

        ''''''

        task.hyperParams = {'optimizer': ['Adam', 'RMSprop', 'SGD'],  # варианты оптимизаторов
                            'batch_size': [8, 16, 32, 64],  # варианты размера батча
                            'lr_A_R': [0.00025, 0.0005, 0.001, 0.002, 0.004],  # варианты скорости обучения для Adam и RMSprop
                            'lr_SGD': [0.0025, 0.005, 0.01, 0.02, 0.04]}   # варианты скорости обучения для SGD

        # создать сетку
        n_opt = len(task.hyperParams['optimizer'])
        n_batch = len(task.hyperParams['batch_size'])
        n_lr = len(task.hyperParams['lr_A_R'])
        assert n_lr == len(task.hyperParams['lr_SGD'])
        task.hp_grid = np.zeros((n_opt, n_lr, n_batch))  # (3, 5, 4)

        task.counter = 1

        # np.random.seed(1)
        # стартовая точка случайно
        opt_ini = np.random.randint(0, high=n_opt)
        lr_ini = np.random.randint(0, high=n_lr)
        bs_ini = np.random.randint(0, high=n_batch)

        # opt_ini = 0
        # lr_ini = 0
        # bs_ini = 3

        # print( opt_ini, lr_ini, bs_ini)
        task.cur_hp = {'optimizer': task.hyperParams['optimizer'][opt_ini],
                       'batch_size': task.hyperParams['batch_size'][bs_ini]}
        if task.cur_hp['optimizer'] == 'SGD':
            task.cur_hp['learning_rate'] = task.hyperParams['lr_SGD'][lr_ini]
        else:
            task.cur_hp['learning_rate'] = task.hyperParams['lr_A_R'][lr_ini]

        task.cur_C_hp = task.cur_hp.copy()

        task.cur_central_point = [opt_ini, lr_ini, bs_ini]
        task.cur_training_point = [opt_ini, lr_ini, bs_ini]

        # Experiment's directory
        MSK = timezone('Europe/Moscow')
        msk_time = datetime.now(MSK)
        tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')
        # print(tt)

        obj_codes = nnDB.get_cat_IDs_by_names(list(task.objects))
        obj_codes.sort()
        obj_str = '_'.join(map(str, obj_codes))
        # print( obj_str )

        task.exp_name = task.task_type + '_' + obj_str + '_DT_' + tt
        task.exp_path = f"{_data_dir}/trainedNN/{task.exp_name}"
        if not os.path.exists(task.exp_path):
            os.makedirs(task.exp_path)

        # Experiment History
        task.history = pd.DataFrame(
            {'task_type': task.task_type,
             'objects': [task.objects],
             'exp_name': task.exp_name,

             'pipeline': task.fixed_hparams['pipeline'],
             'last_layers': [task.fixed_hparams['last_layers']],
             'augmenParams': [task.fixed_hparams['augmenParams']],
             'loss': task.fixed_hparams['loss'],
             'metrics': task.fixed_hparams['metrics'],
             'epochs': task.fixed_hparams['epochs'],
             'stop_criterion': task.fixed_hparams['stop_criterion'],
             'data': [task.data],

             'optimizer': [task.hyperParams['optimizer']],
             'batch_size': [task.hyperParams['batch_size']],
             'learning_rate': [[[0.00025, 0.0005, 0.001, 0.002, 0.004], [0.00025, 0.0005, 0.001, 0.002, 0.004],
                     [0.0025, 0.005, 0.01, 0.02, 0.04]]],

             'metric_test_value': task.goal[str(list(task.goal.keys())[0])],
             'train_subdir': '',
             'time_stat': [[]],
             'total_time': 0,
             'Additional_params': [{}]})

        task.history.to_csv(task.exp_path + '/' + task.exp_name + '__History.csv', index_label='Index')
        # index=False)

        task.cur_state = 'Training_Gen'


@rule
class DataGenerator(Rule):
    """
    Прием для создания генераторов изображений по заданным в curStrategy параметрам аугментации
    В этот прием попадем как при первичном обучении, так и при смене параметров аугментации после обучения модели
    TODO: что такое curStrategy?
    """

    def can_apply(self, task):
        return task.task_ct == "train" and task.cur_state == 'Training_Gen'

    def apply(self, task, solver_state):
        with open(task.log_name, 'a') as log_file:
            print('DataGenerator', file=log_file)

        task.generators = create_generators(task.model, task.data, task.fixed_hparams['augmenParams'], task.cur_hp['batch_size'])
        task.cur_state = 'Training_MA'


@rule
class ModelAssembling(Rule):
    """
    Прием для сборки модели по заданным в curStrategy параметрам
    """
    def can_apply(self, task, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'Training_MA'

    def apply(self, task, solver_state: SolverState):
        printlog('ModelAssembling')

        task.model = create_model(task.fixed_hparams['pipeline'], task.fixed_hparams['last_layers'])

        # Директория для обучаемого экземпляра
        task.train_subdir = task.exp_name + '_' + str(task.counter)
        model_path = task.exp_path + '/' + task.train_subdir
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        prefix = model_path + '/' + task.train_subdir
        task.model.save(prefix + '__Model.h5')
        tf.keras.utils.plot_model(task.model, to_file=prefix + '__ModelPlot.png', rankdir='TB', show_shapes=True)

        task.cur_state = 'Training_FIT'


@rule
class FitModel(Rule):
    """ Прием для обучения модели """

    def can_apply(self, task: Any, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'Training_FIT'

    def apply(self, task: Any, solver_state: SolverState):
        # установить все гиперпараметры
        with open(task.log_name, 'a') as log_file:
            print(task.train_subdir, file=log_file)
            print(task.cur_hp, file=log_file)
        printlog(task.train_subdir)
        printlog(task.cur_hp)

        scores = fit_model(task.model, {**task.fixed_hparams, **task.cur_hp}, task.generators, task.train_subdir,
                           history=task.history,
                           stop_flag=getattr(task, 'stop_flag', None))

        task.hp_grid[tuple(task.cur_training_point)] = scores[1]

        with open(task.log_name, 'a') as log_file:
            print(scores, file=log_file)

        task.cur_state = 'GridStep'


@rule
class GridStep(Rule):
    """ Прием для перехода к следующей точке сетки """

    def can_apply(self, task):
        return task.task_ct == "train" and task.cur_state == 'GridStep'

    def apply(self, task):

        printlog('GRID')
        printlog('central hp', task.cur_C_hp)
        # print(task.cur_central_point, task.cur_training_point)
        # print(task.hp_grid)

        with open(task.log_name, 'a') as log_file:
            print('\n', file=log_file)
            print(task.hp_grid, file=log_file)
            print('\n', file=log_file)
            print('GRID', file=log_file)
            print('central hp', task.cur_C_hp, file=log_file)

        if not hasattr(task, 'best_model') or task.hp_grid[tuple(task.cur_training_point)] > \
                task.hp_grid[tuple(task.best_model)]:
            task.best_model = task.cur_training_point.copy()
            task.best_num = task.counter

        # пройтись по окрестности центральной точки, на наличие необученного соседа
        checkN, cur = neighborhood(task.cur_central_point, task.hp_grid)

        if not checkN:

            printlog('neighboor exists')

            with open(task.log_name, 'a') as log_file:
                print('neighboor exists', file=log_file)

            # если есть нулевой сосед
            task.cur_training_point = cur

            vC = task.cur_central_point
            vN = task.cur_training_point

            # print(vC, vN)

            if vC[0] != vN[0]:
                # print('total')

                # поменяла оптимизатор => меняем все три гиперпараметра по массивам
                task.cur_hp['optimizer'] = task.hyperParams['optimizer'][vN[0]]
                task.cur_hp['batch_size'] = task.hyperParams['batch_size'][vN[2]]

                if task.cur_hp['optimizer'] == 'SGD':
                    task.cur_hp['learning_rate'] = task.hyperParams['lr_SGD'][vN[1]]
                else:
                    task.cur_hp['learning_rate'] = task.hyperParams['lr_A_R'][vN[1]]

            elif vC[1] != vN[1]:
                # print('lr+bs')
                # поменяла learning rate => должен пропорционально поменяться batch size
                task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']

                if task.cur_hp['optimizer'] == 'SGD':
                    task.cur_hp['learning_rate'] = task.hyperParams['lr_SGD'][vN[1]]
                else:
                    task.cur_hp['learning_rate'] = task.hyperParams['lr_A_R'][vN[1]]

                task.cur_hp['batch_size'] = int(task.cur_C_hp['batch_size'] * 2 ** (
                            (vN[2] - vC[2]) + (vN[1] - vC[1])))  # изменение батча из-за learning rate см бумаги

            else:
                # print('only bs')
                task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']
                task.cur_hp['learning_rate'] = task.cur_C_hp['learning_rate']

                # если изменили только batch size, то просто умножили или поделили на два текущее значение
                # (предполагаю, что в центральной точке нормально ? в смысле без неправильных сдвигов)
                task.cur_hp['batch_size'] = int(task.cur_C_hp['batch_size'] * 2 ** (vN[2] - vC[2]))

            if task.cur_hp['batch_size'] > 64:
                task.cur_hp['batch_size'] = 64

            # print(task.cur_hp)
            # task.hp_grid[tuple(vN)]=0.5

        else:
            # все соседи уже обработаны

            with open(task.log_name, 'a') as log_file:
                print('checkN of our central point is equal -1 => Step', file=log_file)
                print(task.hp_grid, file=log_file)

            printlog('checkN of our central point is equal -1 => Step')
            printlog(task.hp_grid)

            if task.best_model == task.cur_central_point:

                with open(task.log_name, 'a') as log_file:
                    print('task.best_model == task.cur_central_point -- local maximum', file=log_file)

                printlog('task.best_model == task.cur_central_point -- local maximum')
                # осмотрела всю окрестность, но лучше результата нет => попали в локальный максимум => сохранить реузультаты и закончить работу
                # ЛУЧШИЙ РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ЗАПИСАТЬ В БД
                best = task.history['Index'] == task.best_num  # является ли текущий индекс лучшим
                loc = task.history.loc[best]
                prefix = f"{_data_dir}/trainedNN/{loc['exp_name'].iloc[0]}/{loc['train_subdir'].iloc[0]}/{loc['train_subdir'].iloc[0]}"
                nnDB.add_model_record(task_type=task.task_type,
                                      categories=list(task.objects),  # категории объектов
                                      model_address=prefix + '__Model.h5',  # путь к файлу, где лежит модель
                                      metrics={list(task.goal.keys())[0]: loc['metric_test_value'].iloc[0]},  # значения метрик
                                      history_address=prefix + '__History.h5')  # путь к файлу, где лежит история обучения
                task.cur_state = 'Done'
                return

            else:
                # поменяла центральную точку => надо осмотреть её окрестность

                with open(task.log_name, 'a') as log_file:
                    print('have found a new central point => change it anf it\'s hyperParams ', file=log_file)

                printlog('have found a new central point => change it anf it\'s hyperParams ')
                task.cur_central_point = task.best_model.copy()
                task.cur_C_hp['optimizer'] = task.hyperParams['optimizer'][task.cur_central_point[0]]
                task.cur_C_hp['batch_size'] = task.hyperParams['batch_size'][task.cur_central_point[2]]

                if task.cur_C_hp['optimizer'] == 'SGD':
                    task.cur_C_hp['learning_rate'] = task.hyperParams['lr_SGD'][task.cur_central_point[1]]
                else:
                    task.cur_C_hp['learning_rate'] = task.hyperParams['lr_A_R'][task.cur_central_point[1]]

                checkN, cur = neighborhood(task.cur_central_point, task.hp_grid)

                if not checkN:

                    with open(task.log_name, 'a') as log_file:
                        print('neighboor of a new central point exists', file=log_file)

                    printlog('neighboor of a new central point exists')

                    # если есть нулевой сосед
                    task.cur_training_point = cur

                    vC = task.cur_central_point
                    vN = task.cur_training_point

                    # print(vC, vN)

                    if vC[0] != vN[0]:
                        # print('total')

                        # поменяла оптимизатор => меняем все три гиперпараметра по массивам
                        task.cur_hp['optimizer'] = task.hyperParams['optimizer'][vN[0]]
                        task.cur_hp['batch_size'] = task.hyperParams['batch_size'][vN[2]]

                        if task.cur_hp['optimizer'] == 'SGD':
                            task.cur_hp['learning_rate'] = task.hyperParams['lr_SGD'][vN[1]]
                        else:
                            task.cur_hp['learning_rate'] = task.hyperParams['lr_A_R'][vN[1]]

                    elif vC[1] != vN[1]:
                        # print('lr+bs')
                        # поменяла learning rate => должен пропорционально поменяться batch size
                        task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']

                        if task.cur_hp['optimizer'] == 'SGD':
                            task.cur_hp['learning_rate'] = task.hyperParams['lr_SGD'][vN[1]]
                        else:
                            task.cur_hp['learning_rate'] = task.hyperParams['lr_A_R'][vN[1]]

                        task.cur_hp['batch_size'] = int(task.cur_C_hp['batch_size'] * 2 ** ((vN[2] - vC[2]) + (vN[1] - vC[1])))  # изменение батча из-за learning rate см бумаги

                    else:
                        # print('only bs')
                        task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']
                        task.cur_hp['learning_rate'] = task.cur_C_hp['learning_rate']

                        # если изменили только batch size, то просто умножили или поделили на два текущее значение
                        # (предполагаю, что в центральной точке нормально ? в смысле без неправильных сдвигов)
                        task.cur_hp['batch_size'] = int(task.cur_C_hp['batch_size'] * 2 ** (vN[2] - vC[2]))

                    if task.cur_hp['batch_size'] > 64:
                        task.cur_hp['batch_size'] = 64

                    # print(task.cur_hp)
                    # tate.task.hp_grid[tuple(vN)]=0.5

                else:

                    with open(task.log_name, 'a') as log_file:
                        print('set a new central point, but it\'s neighborhood has already filled => local maximum',
                              file=log_file)

                    printlog('set a new central point, but it\'s neighborhood has already filled => local maximum')
                    # осмотрела всю окрестность, но лучше результата нет => попали в локальный максимум => сохранить реузультаты и закончить работу
                    # ЛУЧШИЙ РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ЗАПИСАТЬ В БД
                    best = task.history['Index'] == task.best_num  # является ли текущий индекс лучшим
                    loc = task.history.loc[best]
                    prefix = f"{_data_dir}/trainedNN/{loc['exp_name'].iloc[0]}/{loc['train_subdir'].iloc[0]}/{loc['train_subdir'].iloc[0]}"
                    nnDB.add_model_record(task_type=task.task_type,
                                          categories=list(task.objects),
                                          model_address=prefix + '__Model.h5',
                                          metrics={list(task.goal.keys())[0]: loc['metric_test_value'].iloc[0]},
                                          history_address=prefix + 'History.h5')  # TODO: почему здесь "History.h5", а не "__History.h5"?

                    task.cur_state = 'Done'
                    return

        task.counter += 1
        task.cur_state = 'Training_Gen'
        if task.counter > 600:
            task.cur_state = 'Done'


