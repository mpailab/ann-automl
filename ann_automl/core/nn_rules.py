import itertools
import math
import sys
import time
import traceback
from collections import defaultdict

import ipywidgets
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import os
from . import db_module
from datetime import datetime
from pytz import timezone
from .solver import Rule, rule, Task, printlog, _log_dir, SolverState
from ..utils.process import request, NoHandlerError, pcall

myDB = db_module.dbModule(dbstring='sqlite:///tests.sqlite')  # TODO: уточнить путь к файлу базы данных


_data_dir = 'data'


def set_data_dir(data_dir):
    """
    Set the data directory. Data directory contains the following subdirectories:
    - architecures: contains the neural network architectures
    - datasets: contains the datasets
    - trainedNN: contains the trained neural networks
    - history: contains the training history of the neural networks
    """
    global _data_dir
    _data_dir = data_dir


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):  # TODO: для чего нужен параметр logs?
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):  # TODO: для чего нужен параметр logs?
        self.epoch_time_start = time.time()
        if epoch == 0:
            self.start_of_train = self.epoch_time_start

    def on_epoch_end(self, epoch, logs={}):  # TODO: для чего нужен параметр logs?
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs={}):  # TODO: для чего нужен параметр logs?
        self.total_time = (time.time() - self.start_of_train)


# функция осмотра окрестности
def neighborhood(cp, rr):
    """
    Finds next point in the neighborhood of the current point where there is zero in rr

    Parameters
    ----------
    cp: list[int]
        Current point
    rr: ndarray
        Array of function values

    Returns
    -------
    tuple
        (neighbor, point) where neighbor=0 if neighbor is found, -1 if neighbor is not found;
        point is the next point in the neighborhood
    """
    st = (-1, 0, 1)
    neighbor = -1
    cur = []

    printlog('\n')

    for i in range(3):
        # print('i ', i)
        if (cp[0] + st[i]) < 0 or (cp[0] + st[i]) >= rr.shape[0]:
            # print('cond  i')
            continue
        for j in range(3):
            # print('j ', j)
            if (cp[1] + st[j]) < 0 or (cp[1] + st[j]) >= rr.shape[1]:
                # print('cond j')
                continue

            for k in range(3):
                # print('k ', k)
                # print (cp, rr[cp[0]+st[i], cp[1]+st[j], cp[2]+st[k] ])
                # print('grid value',  abs(rr[cp[0]+st[i], cp[1]+st[j], cp[2]+st[k] ]))
                if 0 <= (cp[2] + st[k]) < rr.shape[2] and abs(rr[cp[0] + st[i], cp[1] + st[j], cp[2] + st[k]]) < 1e-6:
                    cur = [cp[0] + st[i], cp[1] + st[j], cp[2] + st[k]]
                    # print('cond k',  cur)
                    neighbor = 0
                    break
            if neighbor == 0:
                break
        if neighbor == 0:
            break

    return neighbor, cur


def find_zero_neighbor(center, table, radius=1):
    """
    Finds next point in the neighborhood of the current point where there is zero in table

    Parameters
    ----------
    center: list[int]
        Current point
    table: ndarray
        Array of function values
    radius: int
        Radius of the neighborhood

    Returns
    -------
    optional list[int]
        Next point in the neighborhood where there is zero in table or None if there is no such point
    """
    ranges = [range(max(0, center[i] - radius), min(szi, center[i] + radius + 1)) for i, szi in enumerate(table.shape)]
    for i in itertools.product(*ranges):
        if abs(table[i]) <= 1e-6:
            return list(i)
    return None


# Допустимая погрешность по достигнутой точности при выборе общей стратегии обучения
# eps = 0.1


@rule(NNTask)
class CheckSuitableModelExistence(Rule):
    """ Приём для поиска готовой подходящей модели в базе данных """

    def can_apply(self, task: NNTask, solver_state: SolverState) -> bool:
        return task.task_ct == "train" and task.cur_state == 'FirstCheck'

    def apply(self, task: NNTask, solver_state: SolverState):

        with open(task.log_name, 'a') as log_file:
            print('CheckSuitableModelExistence', file=log_file)

        ch_res = myDB.get_models_by_filter({
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
                    task.cur_state = 'Done'  # TODO: Replace done with actual state
                elif decision == 'from_history':
                    printlog('Подбор гиперпараметров по истории (пока не реализовано)')
                    task.cur_state = 'Done'  # TODO: Replace done with actual state
                elif decision == 'grid_search':
                    printlog('Поиск по сетке')
                    task.cur_state = 'DB'  # TODO: Replace done with actual state
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
                          list(task.goal.keys())[0]: ch_res.iloc[i]['metric_value']})

            task.cur_state = 'UserDec'
            task.message = 'There is a suitable model in our base of trained models. Would you like to train an another model? "No" - 0; "Yes" - 1. '
            task.actions = {'0': 'Done', '1': 'DB'}
        task.suitModels = s


@rule
class UserDec(Rule):
    """ Приём решения пользователя о необходимости обучения новой модели """
    def can_apply(self, task: NNTask, solver_state: SolverState) -> bool:
        return task.task_ct == "train" and task.cur_state == 'UserDec'

    def apply(self, task: NNTask, solver_state: SolverState):
        printlog('\n' + task.message)
        answer = input()
        task.cur_state = task.actions[str(answer)]

        with open(task.log_name, 'a') as log_file:
            print('UserDec Rule', file=log_file)
            print(task.message + ' ' + answer, file=log_file)

        if task.cur_state == 'Done':

            # считаем, что сюда пришли только сразу после firstCheck => сохрани модели и всё
            # или после создания сетки, но до первого обучения [ошибки не предусмотрены] => сохрани историю,
            # или после хотя бы 1 обучения, т.е. существует bestModel, если модель подходит до
            if hasattr(task, 'bestModel'):
                loc = task.history.loc[task.history['Index'] == task.best_num]
                prefix = f"{_data_dir}/trainedNN/{loc['exp_name'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}"
                myDB.add_model_record(task_type=task.task_type, categories=list(task.objects),
                                      model_address=prefix + '__Model.h5',
                                      metrics={list(task.goal.keys())[0]: loc['Metric_achieved_result'].iloc[0]},
                                      history_address=prefix + 'History.h5')

        task.message = None
        task.actions = None


####################################################

@rule
class CreateDatabase(Rule):
    """ Приём для создания обучающей выборки """
    def can_apply(self, task: NNTask, solver_state: SolverState) -> bool:
        return task.task_ct == "train" and task.cur_state == 'DB'

    def apply(self, task: NNTask, solver_state: SolverState):
        with open(task.log_name, 'a') as log_file:
            print('CreateDatabase', file=log_file)

        # crops
        # In next version there will be two databases methods\tricks - the first one to check are required categories exist
        # the second one - to create database

        tmpData = myDB.load_specific_categories_annotations(list(task.objects), normalizeCats=True,
                                                            splitPoints=[0.7, 0.85],
                                                            curExperimentFolder='./', crop_bbox=True,
                                                            cropped_dir='./crops/')

        task.data = {'train': tmpData[1]['train'],
                           'validate': tmpData[1]['validate'],
                           'test': tmpData[1]['test'],
                           'dim': (224, 224, 3),
                           'augmenConstr': {'vertical_flip': None}}
        task.cur_state = 'Training'


"""Hyper Tuning Grid Block"""


@rule
class SetGrid(Rule):
    """создать общую сетку параметров и инфраструктуру всего эксперимента"""

    def can_apply(self, task: NNTask, solver_state: SolverState) -> bool:
        return task.task_ct == "train" and task.cur_state == 'Training'

    def apply(self, task: NNTask, solver_state: SolverState):

        with open(task.log_name, 'a') as log_file:
            print('SetGrid', file=log_file)

        '''tmp'''
        task.fixedHyperParams = {'pipeline': 'ResNet18',
                                       'modelLastLayers': [{'Type': 'Dense', 'units': 64},
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
                                       'stoppingCriterion': 'stop_val_metr'}

        ''''''

        task.hyperParams = {'optimizer': ['Adam', 'RMSprop', 'SGD'],  # варианты оптимизаторов
                                  'batchSize': [8, 16, 32, 64],  # варианты размера батча
                                  'lr_A_R': [0.00025, 0.0005, 0.001, 0.002, 0.004],  # варианты скорости обучения для Adam и RMSprop
                                  'lr_SGD': [0.0025, 0.005, 0.01, 0.02, 0.04]}   # варианты скорости обучения для SGD

        # создать сетку
        n_opt = len(task.hyperParams['optimizer'])
        n_batch = len(task.hyperParams['batchSize'])
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
                             'batchSize': task.hyperParams['batchSize'][bs_ini]}
        if task.cur_hp['optimizer'] == 'SGD':
            task.cur_hp['lr'] = task.hyperParams['lr_SGD'][lr_ini]
        else:
            task.cur_hp['lr'] = task.hyperParams['lr_A_R'][lr_ini]

        task.cur_C_hp = task.cur_hp.copy()

        task.cur_central_point = [opt_ini, lr_ini, bs_ini]
        task.cur_training_point = [opt_ini, lr_ini, bs_ini]

        # Experiment's directory
        MSK = timezone('Europe/Moscow')
        msk_time = datetime.now(MSK)
        tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')
        # print(tt)

        obj_codes = myDB.get_cat_IDs_by_names(list(task.objects))
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

             'pipeline': task.fixedHyperParams['pipeline'],
             'modelLastLayers': [task.fixedHyperParams['modelLastLayers']],
             'augmenParams': [task.fixedHyperParams['augmenParams']],
             'loss': task.fixedHyperParams['loss'],
             'metrics': task.fixedHyperParams['metrics'],
             'epochs': task.fixedHyperParams['epochs'],
             'stoppingCriterion': task.fixedHyperParams['stoppingCriterion'],
             'data': [task.data],

             'optimizer': [task.hyperParams['optimizer']],
             'batchSize': [task.hyperParams['batchSize']],
             'lr': [[[0.00025, 0.0005, 0.001, 0.002, 0.004], [0.00025, 0.0005, 0.001, 0.002, 0.004],
                     [0.0025, 0.005, 0.01, 0.02, 0.04]]],

             'Metric_achieved_result': task.goal[str(list(task.goal.keys())[0])],
             'curTrainingSubfolder': '',
             'timeStat': [[]],
             'totalTime': 0,
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

    def can_apply(self, task: NNTask, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'Training_Gen'

    def apply(self, task: NNTask, solver_state: SolverState):
        with open(task.log_name, 'a') as log_file:
            print('DataGenerator', file=log_file)

        df_train = pd.read_csv(task.data['train'])
        df_validate = pd.read_csv(task.data['validate'])
        df_test = pd.read_csv(task.data['test'])

        dataGen = keras.preprocessing.image.ImageDataGenerator(task.fixedHyperParams['augmenParams'])
        # ,directory='./datasets/Kaggle_CatsVSDogs'  list(df_validate.columns)[0]

        train_generator = dataGen.flow_from_dataframe(df_train, x_col=list(df_train.columns)[0],
                                                      y_col=list(df_train.columns)[1],
                                                      target_size=task.data['dim'][0:2],
                                                      class_mode='raw',
                                                      batch_size=task.cur_hp['batchSize'])
        validate_generator = dataGen.flow_from_dataframe(df_validate, x_col=list(df_validate.columns)[0],
                                                         y_col=list(df_validate.columns)[1],
                                                         target_size=task.data['dim'][0:2],
                                                         class_mode='raw',
                                                         batch_size=task.cur_hp['batchSize'])
        test_generator = dataGen.flow_from_dataframe(df_test, x_col=list(df_test.columns)[0],
                                                     y_col=list(df_test.columns)[1],
                                                     target_size=task.data['dim'][0:2],
                                                     class_mode='raw',
                                                     batch_size=task.cur_hp['batchSize'])

        task.generators = [train_generator, validate_generator, test_generator]

        task.cur_state = 'Training_MA'


@rule
class ModelAssembling(Rule):
    """
    Прием для сборки модели по заданным в curStrategy параметрам
    """

    def can_apply(self, task: NNTask, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'Training_MA'

    def apply(self, task: NNTask, solver_state: SolverState):
        with open(task.log_name, 'a') as log_file:
            print('ModelAssembling', file=log_file)

        x = keras.layers.Input(shape=(task.data['dim']))
        y = keras.models.load_model(f'{_data_dir}/architectures/' + task.fixedHyperParams['pipeline'] + '.h5')(x)
        # y=keras.layers.Dense(units=64)(y)
        # y=keras.layers.Activation(activation='relu')(y)
        y = keras.layers.Dense(units=1)(y)
        y = keras.layers.Activation(activation='sigmoid')(y)
        task.model = keras.models.Model(inputs=x, outputs=y)

        # Директория для обучаемого экземпляра
        task.curTrainingSubfolder = task.exp_name + '_' + str(task.counter)
        model_path = task.exp_path + '/' + task.curTrainingSubfolder
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        prefix = model_path + '/' + task.curTrainingSubfolder
        task.model.save(prefix + '__Model.h5')
        tf.keras.utils.plot_model(task.model, to_file=prefix + '__ModelPlot.png', rankdir='TB', show_shapes=True)

        task.cur_state = 'Training_FIT'


@rule
class FitModel(Rule):
    """ Прием для обучения модели """

    def can_apply(self, task: NNTask, solver_state: SolverState):
        return task.task_ct == "train" and task.cur_state == 'Training_FIT'

    def apply(self, task: NNTask, solver_state: SolverState):
        # установить все гиперпараметры
        optimizer, lr = task.cur_hp['optimizer'], task.cur_hp['lr']
        if task.cur_hp['optimizer'] == 'SGD':
            task.model.compile(optimizer=eval(f'tf.keras.optimizers.{optimizer}(nesterov=True, learning_rate={lr})'),
                                     loss=task.fixedHyperParams['loss'],
                                     metrics=[task.fixedHyperParams['metrics']])
        else:
            task.model.compile(optimizer=eval(f'tf.keras.optimizers.{optimizer}(learning_rate={lr})'),
                                     loss=task.fixedHyperParams['loss'],
                                     metrics=[task.fixedHyperParams['metrics']])

        # Обучение нейронной сети
        # print(task.fixedHyperParams)

        with open(task.log_name, 'a') as log_file:
            print(task.curTrainingSubfolder, file=log_file)
            print(task.cur_hp, file=log_file)

        printlog('\n')
        printlog(task.curTrainingSubfolder)
        printlog('\n')
        printlog(task.cur_hp)

        model_path = f'{task.exp_path}/{task.curTrainingSubfolder}'
        model_history = model_path + '/' + task.curTrainingSubfolder + '__History.csv'

        C_Log = keras.callbacks.CSVLogger(model_history)  # устанавливаем файл для логов
        C_Ch = keras.callbacks.ModelCheckpoint(
            model_path + '/' + task.curTrainingSubfolder + '_weights' + '-{epoch:02d}.h5',  # устанавливаем имя файла для сохранения весов
            monitor='val_' + task.fixedHyperParams['metrics'],
            save_best_only=True, save_weights_only=False, mode='max', verbose=1)
        C_ES = keras.callbacks.EarlyStopping(monitor='val_' + task.fixedHyperParams['metrics'], min_delta=0.001,
                                             mode='max', patience=5)
        C_T = TimeHistory()

        task.model.fit(x=task.generators[0],
                             steps_per_epoch=len(task.generators[0].filenames) // task.cur_hp['batchSize'],
                             epochs=task.fixedHyperParams['epochs'],
                             validation_data=task.generators[1],
                             callbacks=[C_Log, C_Ch, C_ES, C_T],
                             validation_steps=len(task.generators[1].filenames) // task.cur_hp['batchSize'])

        scores = task.model.evaluate(task.generators[2], steps=None, verbose=1)  # оценка обученной модели
        task.hp_grid[tuple(task.cur_training_point)] = scores[1]

        with open(task.log_name, 'a') as log_file:
            print(scores, file=log_file)

        # сохранение результатов в историю эксперимента

        new_row = ({'Index': task.counter,  # номер эксперимента
                    'task_type': task.task_type,  # тип задачи
                    'objects': [task.objects],  # список объектов, на распознавание которых обучается модель (TODO: проверить!)
                    'exp_name': task.exp_name,  # название эксперимента

                    'pipeline': task.fixedHyperParams['pipeline'],  # TODO: что это?
                    'modelLastLayers': [task.fixedHyperParams['modelLastLayers']],  # TODO: что это?
                    'augmenParams': [task.fixedHyperParams['augmenParams']],  # параметры аугментации
                    'loss': task.fixedHyperParams['loss'],  # функция потерь
                    'metrics': task.fixedHyperParams['metrics'],  # метрика, по которой оценивается качество модели
                    'epochs': task.fixedHyperParams['epochs'],  # количество эпох обучения
                    'stoppingCriterion': task.fixedHyperParams['stoppingCriterion'],  # критерий остановки обучения
                    'data': [task.data],  # набор данных, на котором проводится обучение

                    'optimizer': task.cur_hp['optimizer'],  # оптимизатор
                    'batchSize': task.cur_hp['batchSize'],  # размер батча
                    'lr': task.cur_hp['lr'],  # скорость обучения

                    'Metric_achieved_result': scores[1],  # значение метрики на тестовой выборке
                    'curTrainingSubfolder': task.curTrainingSubfolder,  # папка, в которой хранятся результаты текущего обучения
                    'timeStat': [C_T.times],  # TODO: что сюда записывается, чем отличается от totalTime?
                    'totalTime': C_T.total_time,  # общее время обучения
                    'Additional_params': [{}]})  # дополнительные параметры

        task.history = task.history.append(new_row, ignore_index=True)

        task.history.to_csv(task.exp_path + '/' + task.exp_name + '__History.csv', index=False)

        task.cur_state = 'GridStep'


@rule
class GridStep(Rule):
    """ Прием для перехода к следующей точке сетки """

    def can_apply(self, state):
        return task.task_ct == "train" and task.cur_state == 'GridStep'

    def apply(self, state):

        printlog('GRID')
        printlog('central hp', task.cur_C_hp)
        # print(task.cur_central_point, task.cur_training_point)
        # print(task.hp_grid)

        with open(state.log_name, 'a') as log_file:
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

            with open(state.log_name, 'a') as log_file:
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
                task.cur_hp['batchSize'] = task.hyperParams['batchSize'][vN[2]]

                if task.cur_hp['optimizer'] == 'SGD':
                    task.cur_hp['lr'] = task.hyperParams['lr_SGD'][vN[1]]
                else:
                    task.cur_hp['lr'] = task.hyperParams['lr_A_R'][vN[1]]

            elif vC[1] != vN[1]:
                # print('lr+bs')
                # поменяла learning rate => должен пропорционально поменяться batch size
                task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']

                if task.cur_hp['optimizer'] == 'SGD':
                    task.cur_hp['lr'] = task.hyperParams['lr_SGD'][vN[1]]
                else:
                    task.cur_hp['lr'] = task.hyperParams['lr_A_R'][vN[1]]

                task.cur_hp['batchSize'] = int(task.cur_C_hp['batchSize'] * 2 ** (
                            (vN[2] - vC[2]) + (vN[1] - vC[1])))  # изменение батча из-за learning rate см бумаги

            else:
                # print('only bs')
                task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']
                task.cur_hp['lr'] = task.cur_C_hp['lr']

                # если изменили только batch size, то просто умножили или поделили на два текущее значение
                # (предполагаю, что в центральной точке нормально ? в смысле без неправильных сдвигов)
                task.cur_hp['batchSize'] = int(task.cur_C_hp['batchSize'] * 2 ** (vN[2] - vC[2]))

            if task.cur_hp['batchSize'] > 64:
                task.cur_hp['batchSize'] = 64

            # print(task.cur_hp)
            # task.hp_grid[tuple(vN)]=0.5

        else:
            # все соседи уже обработаны

            with open(state.log_name, 'a') as log_file:
                print('checkN of our central point is equal -1 => Step', file=log_file)
                print(task.hp_grid, file=log_file)

            printlog('checkN of our central point is equal -1 => Step')
            printlog(task.hp_grid)

            if task.best_model == task.cur_central_point:

                with open(state.log_name, 'a') as log_file:
                    print('task.best_model == task.cur_central_point -- local maximum', file=log_file)

                printlog('task.best_model == task.cur_central_point -- local maximum')
                # осмотрела всю окрестность, но лучше результата нет => попали в локальный максимум => сохранить реузультаты и закончить работу
                # ЛУЧШИЙ РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ЗАПИСАТЬ В БД
                best = task.history['Index'] == task.best_num  # является ли текущий индекс лучшим
                loc = task.history.loc[best]
                prefix = f"{_data_dir}/trainedNN/{loc['exp_name'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}"
                myDB.add_model_record(task_type=task.task_type,
                                      categories=list(task.objects),  # категории объектов
                                      model_address=prefix + '__Model.h5',  # путь к файлу, где лежит модель
                                      metrics={list(task.goal.keys())[0]: loc['Metric_achieved_result'].iloc[0]},  # значения метрик
                                      history_address=prefix + '__History.h5')  # путь к файлу, где лежит история обучения
                task.cur_state = 'Done'
                return

            else:
                # поменяла центральную точку => надо осмотреть её окрестность

                with open(state.log_name, 'a') as log_file:
                    print('have found a new central point => change it anf it\'s hyperParams ', file=log_file)

                printlog('have found a new central point => change it anf it\'s hyperParams ')
                task.cur_central_point = task.best_model.copy()
                task.cur_C_hp['optimizer'] = task.hyperParams['optimizer'][task.cur_central_point[0]]
                task.cur_C_hp['batchSize'] = task.hyperParams['batchSize'][task.cur_central_point[2]]

                if task.cur_C_hp['optimizer'] == 'SGD':
                    task.cur_C_hp['lr'] = task.hyperParams['lr_SGD'][task.cur_central_point[1]]
                else:
                    task.cur_C_hp['lr'] = task.hyperParams['lr_A_R'][task.cur_central_point[1]]

                checkN, cur = neighborhood(task.cur_central_point, task.hp_grid)

                if not checkN:

                    with open(state.log_name, 'a') as log_file:
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
                        task.cur_hp['batchSize'] = task.hyperParams['batchSize'][vN[2]]

                        if task.cur_hp['optimizer'] == 'SGD':
                            task.cur_hp['lr'] = task.hyperParams['lr_SGD'][vN[1]]
                        else:
                            task.cur_hp['lr'] = task.hyperParams['lr_A_R'][vN[1]]

                    elif vC[1] != vN[1]:
                        # print('lr+bs')
                        # поменяла learning rate => должен пропорционально поменяться batch size
                        task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']

                        if task.cur_hp['optimizer'] == 'SGD':
                            task.cur_hp['lr'] = task.hyperParams['lr_SGD'][vN[1]]
                        else:
                            task.cur_hp['lr'] = task.hyperParams['lr_A_R'][vN[1]]

                        task.cur_hp['batchSize'] = int(task.cur_C_hp['batchSize'] * 2 ** ((vN[2] - vC[2]) + (vN[1] - vC[1])))  # изменение батча из-за learning rate см бумаги

                    else:
                        # print('only bs')
                        task.cur_hp['optimizer'] = task.cur_C_hp['optimizer']
                        task.cur_hp['lr'] = task.cur_C_hp['lr']

                        # если изменили только batch size, то просто умножили или поделили на два текущее значение
                        # (предполагаю, что в центральной точке нормально ? в смысле без неправильных сдвигов)
                        task.cur_hp['batchSize'] = int(task.cur_C_hp['batchSize'] * 2 ** (vN[2] - vC[2]))

                    if task.cur_hp['batchSize'] > 64:
                        task.cur_hp['batchSize'] = 64

                    # print(task.cur_hp)
                    # tate.task.hp_grid[tuple(vN)]=0.5

                else:

                    with open(state.log_name, 'a') as log_file:
                        print('set a new central point, but it\'s neighborhood has already filled => local maximum',
                              file=log_file)

                    printlog('set a new central point, but it\'s neighborhood has already filled => local maximum')
                    # осмотрела всю окрестность, но лучше результата нет => попали в локальный максимум => сохранить реузультаты и закончить работу
                    # ЛУЧШИЙ РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ЗАПИСАТЬ В БД
                    best = task.history['Index'] == task.best_num  # является ли текущий индекс лучшим
                    loc = task.history.loc[best]
                    prefix = f"{_data_dir}/trainedNN/{loc['exp_name'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}"
                    myDB.add_model_record(task_type=task.task_type,
                                          categories=list(task.objects),
                                          model_address=prefix + '__Model.h5',
                                          metrics={list(task.goal.keys())[0]: loc['Metric_achieved_result'].iloc[0]},
                                          history_address=prefix + 'History.h5')  # TODO: почему здесь "History.h5", а не "__History.h5"?

                    task.cur_state = 'Done'
                    return

        task.counter += 1
        task.cur_state = 'Training_Gen'
        if task.counter > 600:
            task.cur_state = 'Done'
