import itertools
import time
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import os
import db_module
from datetime import datetime
from pytz import timezone
from .solver import Rule, rule, State, Task

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

    print('\n')

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

    return (neighbor, cur)


# Допустимая погрешность по достигнутой точности при выборе общей стратегии обучения
# eps = 0.1


#####################################################################


@rule
class CheckSuitableModelExistence(Rule):
    """ Приём для поиска готовой подходящей модели в базе данных """

    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'FirstCheck'

    def apply(self, state: State):

        with open(state.logName, 'a') as log_file:
            print('CheckSuitableModelExistence', file=log_file)

        ch_res = myDB.get_models_by_filter({
            'min_metrics': state.task.goal,
            'task_type': state.task.taskType,
            'categories': list(state.task.objects)})
        # print(ch_res)

        s = []
        if ch_res.empty:
            print('empty')
            state.curState = 'DB'
        else:
            n = len(ch_res.index)
            for i in range(n):
                s.append({'model_address': ch_res.iloc[i]['model_address'],
                          list(state.task.goal.keys())[0]: ch_res.iloc[i]['metric_value']})

            state.curState = 'UserDec'
            state.message = 'There is a suitable model in our base of trained models. Would you like to train an another model? "No" - 0; "Yes" - 1. '
            state.actions = {'0': 'Done', '1': 'DB'}
        state.task.suitModels = s


@rule
class UserDec(Rule):
    """ Приём решения пользователя о необходимости обучения новой модели """
    def can_apply(self, state):
        return state.task.taskCt == "train" and state.curState == 'UserDec'

    def apply(self, state):

        print('\n' + state.message)
        answer = input()
        state.curState = state.actions[str(answer)]

        with open(state.logName, 'a') as log_file:
            print('UserDec Rule', file=log_file)
            print(state.message + ' ' + answer, file=log_file)

        if state.curState == 'Done':

            # считаем, что сюда пришли только сразу после firstCheck => сохрани модели и всё
            # или после создания сетки, но до первого обучения [ошибки не предусмотрены] => сохрани историю,
            # или после хотя бы 1 обучения, т.е. существует bestModel, если модель подходит до
            if hasattr(state.task, 'bestModel'):
                loc = state.task.history.loc[state.task.history['Index'] == state.task.best_num]
                prefix = f"{_data_dir}/trainedNN/{loc['expName'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}"
                myDB.add_model_record(task_type=state.task.taskType, categories=list(state.task.objects),
                                      model_address=prefix + '__Model.h5',
                                      metrics={list(state.task.goal.keys())[0]: loc['Metric_achieved_result'].iloc[0]},
                                      history_address=prefix + 'History.h5')

        state.message = None
        state.actions = None


####################################################

@rule
class CreateDatabase(Rule):
    """ Приём для создания обучающей выборки """
    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'DB'

    def apply(self, state: State):
        with open(state.logName, 'a') as log_file:
            print('CreateDatabase', file=log_file)

        # crops
        # In next version there will be two databases methods\tricks - the first one to check are required categories exist
        # the second one - to create database

        tmpData = myDB.load_specific_categories_annotations(['cat', 'dog'], normalizeCats=True, splitPoints=[0.7, 0.85],
                                                            curExperimentFolder='./', crop_bbox=True,
                                                            cropped_dir='./crops/')

        state.task.data = {'train': tmpData[1]['train'], 'validate': tmpData[1]['validate'], 'test': tmpData[1]['test'],
                           'dim': (224, 224, 3), 'augmenConstr': {'vertical_flip': None}}
        state.curState = 'Training'


"""Hyper Tuning Grid Block"""


@rule
class SetGrid(Rule):
    """создать общую сетку параметров и инфраструктуру всего эксперимента"""

    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'Training'

    def apply(self, state: State):

        with open(state.logName, 'a') as log_file:
            print('SetGrid', file=log_file)

        '''tmp'''
        state.task.fixedHyperParams = {'pipeline': 'ResNet18',
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

        state.task.hyperParams = {'optimizer': ['Adam', 'RMSprop', 'SGD'],  # варианты оптимизаторов
                                  'batchSize': [8, 16, 32, 64],  # варианты размера батча
                                  'lr_A_R': [0.00025, 0.0005, 0.001, 0.002, 0.004],  # варианты скорости обучения для Adam и RMSprop
                                  'lr_SGD': [0.0025, 0.005, 0.01, 0.02, 0.04]}   # варианты скорости обучения для SGD

        # создать сетку
        n_opt = len(state.task.hyperParams['optimizer'])
        n_batch = len(state.task.hyperParams['batchSize'])
        n_lr = len(state.task.hyperParams['lr_A_R'])
        assert n_lr == len(state.task.hyperParams['lr_SGD'])
        state.task.hp_grid = np.zeros((n_opt, n_lr, n_batch))  # (3, 5, 4)

        state.task.counter = 1

        # np.random.seed(1)
        # стартовая точка случайно
        opt_ini = np.random.randint(0, high=n_opt)
        lr_ini = np.random.randint(0, high=n_lr)
        bs_ini = np.random.randint(0, high=n_batch)

        # opt_ini = 0
        # lr_ini = 0
        # bs_ini = 3

        # print( opt_ini, lr_ini, bs_ini)
        state.task.cur_hp = {'optimizer': state.task.hyperParams['optimizer'][opt_ini],
                             'batchSize': state.task.hyperParams['batchSize'][bs_ini]}
        if state.task.cur_hp['optimizer'] == 'SGD':
            state.task.cur_hp['lr'] = state.task.hyperParams['lr_SGD'][lr_ini]
        else:
            state.task.cur_hp['lr'] = state.task.hyperParams['lr_A_R'][lr_ini]

        state.task.cur_C_hp = state.task.cur_hp.copy()

        state.task.cur_central_point = [opt_ini, lr_ini, bs_ini]
        state.task.cur_training_point = [opt_ini, lr_ini, bs_ini]

        # Experiment's directory
        MSK = timezone('Europe/Moscow')
        msk_time = datetime.now(MSK)
        tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')
        # print(tt)

        obj_codes = myDB.get_cat_IDs_by_names(list(state.task.objects))
        obj_codes.sort()
        obj_str = '_'.join(map(str, obj_codes))
        # print( obj_str )

        state.task.expName = state.task.taskType + '_' + obj_str + '_DT_' + tt
        state.task.expPath = f"{_data_dir}/trainedNN/{state.task.expName}"
        if not os.path.exists(state.task.expPath):
            os.makedirs(state.task.expPath)

        # Experiment History
        state.task.history = pd.DataFrame(
            {'taskType': state.task.taskType, 'objects': [state.task.objects], 'expName': state.task.expName,

             'pipeline': state.task.fixedHyperParams['pipeline'],
             'modelLastLayers': [state.task.fixedHyperParams['modelLastLayers']],
             'augmenParams': [state.task.fixedHyperParams['augmenParams']],
             'loss': state.task.fixedHyperParams['loss'],
             'metrics': state.task.fixedHyperParams['metrics'],
             'epochs': state.task.fixedHyperParams['epochs'],
             'stoppingCriterion': state.task.fixedHyperParams['stoppingCriterion'],
             'data': [state.task.data],

             'optimizer': [state.task.hyperParams['optimizer']],
             'batchSize': [state.task.hyperParams['batchSize']],
             'lr': [[[0.00025, 0.0005, 0.001, 0.002, 0.004], [0.00025, 0.0005, 0.001, 0.002, 0.004],
                     [0.0025, 0.005, 0.01, 0.02, 0.04]]],

             'Metric_achieved_result': state.task.goal[str(list(state.task.goal.keys())[0])],
             'curTrainingSubfolder': '',
             'timeStat': [[]], 'totalTime': 0, 'Additional_params': [{}]})

        state.task.history.to_csv(state.task.expPath + '/' + state.task.expName + '__History.csv', index_label='Index')
        # index=False)

        state.curState = 'Training_Gen'


@rule
class DataGenerator(Rule):
    """
    Прием для создания генераторов изображений по заданным в curStrategy параметрам аугментации
    В этот прием попадем как при первичном обучении, так и при смене параметров аугментации после обучения модели
    TODO: что такое curStrategy?
    """

    def can_apply(self, state):
        return state.task.taskCt == "train" and state.curState == 'Training_Gen'

    def apply(self, state):
        with open(state.logName, 'a') as log_file:
            print('DataGenerator', file=log_file)

        df_train = pd.read_csv(state.task.data['train'])
        df_validate = pd.read_csv(state.task.data['validate'])
        df_test = pd.read_csv(state.task.data['test'])

        dataGen = keras.preprocessing.image.ImageDataGenerator(state.task.fixedHyperParams['augmenParams'])
        # ,directory='./datasets/Kaggle_CatsVSDogs'  list(df_validate.columns)[0]

        train_generator = dataGen.flow_from_dataframe(df_train, x_col=list(df_train.columns)[0],
                                                      y_col=list(df_train.columns)[1],
                                                      target_size=state.task.data['dim'][0:2],
                                                      class_mode='raw',
                                                      batch_size=state.task.cur_hp['batchSize'])
        validate_generator = dataGen.flow_from_dataframe(df_validate, x_col=list(df_validate.columns)[0],
                                                         y_col=list(df_validate.columns)[1],
                                                         target_size=state.task.data['dim'][0:2],
                                                         class_mode='raw',
                                                         batch_size=state.task.cur_hp['batchSize'])
        test_generator = dataGen.flow_from_dataframe(df_test, x_col=list(df_test.columns)[0],
                                                     y_col=list(df_test.columns)[1],
                                                     target_size=state.task.data['dim'][0:2],
                                                     class_mode='raw',
                                                     batch_size=state.task.cur_hp['batchSize'])

        state.task.generators = [train_generator, validate_generator, test_generator]

        state.curState = 'Training_MA'


@rule
class ModelAssembling(Rule):
    """
    Прием для сборки модели по заданным в curStrategy параметрам
    """

    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'Training_MA'

    def apply(self, state: State):
        with open(state.logName, 'a') as log_file:
            print('ModelAssembling', file=log_file)

        x = keras.layers.Input(shape=(state.task.data['dim']))
        y = keras.models.load_model(f'{_data_dir}/architectures/' + state.task.fixedHyperParams['pipeline'] + '.h5')(x)
        # y=keras.layers.Dense(units=64)(y)
        # y=keras.layers.Activation(activation='relu')(y)
        y = keras.layers.Dense(units=1)(y)
        y = keras.layers.Activation(activation='sigmoid')(y)
        state.task.model = keras.models.Model(inputs=x, outputs=y)

        # Директория для обучаемого экземпляра
        state.task.curTrainingSubfolder = state.task.expName + '_' + str(state.task.counter)
        model_path = state.task.expPath + '/' + state.task.curTrainingSubfolder
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        prefix = model_path + '/' + state.task.curTrainingSubfolder
        state.task.model.save(prefix + '__Model.h5')
        tf.keras.utils.plot_model(state.task.model, to_file=prefix + '__ModelPlot.png', rankdir='TB', show_shapes=True)

        state.curState = 'Training_FIT'


@rule
class FitModel(Rule):
    """ Прием для обучения модели """

    def can_apply(self, state):
        return state.task.taskCt == "train" and state.curState == 'Training_FIT'

    def apply(self, state):
        # установить все гиперпараметры
        optimizer, lr = state.task.cur_hp['optimizer'], state.task.cur_hp['lr']
        if state.task.cur_hp['optimizer'] == 'SGD':
            state.task.model.compile(optimizer=eval(f'tf.keras.optimizers.{optimizer}(nesterov=True, learning_rate={lr})'),
                                     loss=state.task.fixedHyperParams['loss'],
                                     metrics=[state.task.fixedHyperParams['metrics']])
        else:
            state.task.model.compile(optimizer=eval(f'tf.keras.optimizers.{optimizer}(learning_rate={lr})'),
                                     loss=state.task.fixedHyperParams['loss'],
                                     metrics=[state.task.fixedHyperParams['metrics']])

        # Обучение нейронной сети
        # print(state.task.fixedHyperParams)

        with open(state.logName, 'a') as log_file:
            print(state.task.curTrainingSubfolder, file=log_file)
            print(state.task.cur_hp, file=log_file)

        print('\n')
        print(state.task.curTrainingSubfolder)
        print('\n')
        print(state.task.cur_hp)

        model_path = f'{state.task.expPath}/{state.task.curTrainingSubfolder}'
        model_history = model_path + '/' + state.task.curTrainingSubfolder + '__History.csv'

        C_Log = keras.callbacks.CSVLogger(model_history)  # устанавливаем файл для логов
        C_Ch = keras.callbacks.ModelCheckpoint(
            model_path + '/' + state.task.curTrainingSubfolder + '_weights' + '-{epoch:02d}.h5',  # устанавливаем имя файла для сохранения весов
            monitor='val_' + state.task.fixedHyperParams['metrics'],
            save_best_only=True, save_weights_only=False, mode='max', verbose=1)
        C_ES = keras.callbacks.EarlyStopping(monitor='val_' + state.task.fixedHyperParams['metrics'], min_delta=0.001,
                                             mode='max', patience=5)
        C_T = TimeHistory()

        state.task.model.fit(x=state.task.generators[0],
                             steps_per_epoch=len(state.task.generators[0].filenames) // state.task.cur_hp['batchSize'],
                             epochs=state.task.fixedHyperParams['epochs'],
                             validation_data=state.task.generators[1],
                             callbacks=[C_Log, C_Ch, C_ES, C_T],
                             validation_steps=len(state.task.generators[1].filenames) // state.task.cur_hp['batchSize'])

        scores = state.task.model.evaluate(state.task.generators[2], steps=None, verbose=1)  # оценка обученной модели
        state.task.hp_grid[tuple(state.task.cur_training_point)] = scores[1]

        with open(state.logName, 'a') as log_file:
            print(scores, file=log_file)

        # сохранение результатов в историю эксперимента

        new_row = ({'Index': state.task.counter,  # номер эксперимента
                    'taskType': state.task.taskType,  # тип задачи
                    'objects': [state.task.objects],  # список объектов, на распознавание которых обучается модель (TODO: проверить!)
                    'expName': state.task.expName,  # название эксперимента

                    'pipeline': state.task.fixedHyperParams['pipeline'],  # TODO: что это?
                    'modelLastLayers': [state.task.fixedHyperParams['modelLastLayers']],  # TODO: что это?
                    'augmenParams': [state.task.fixedHyperParams['augmenParams']],  # параметры аугментации
                    'loss': state.task.fixedHyperParams['loss'],  # функция потерь
                    'metrics': state.task.fixedHyperParams['metrics'],  # метрика, по которой оценивается качество модели
                    'epochs': state.task.fixedHyperParams['epochs'],  # количество эпох обучения
                    'stoppingCriterion': state.task.fixedHyperParams['stoppingCriterion'],  # критерий остановки обучения
                    'data': [state.task.data],  # набор данных, на котором проводится обучение

                    'optimizer': state.task.cur_hp['optimizer'],  # оптимизатор
                    'batchSize': state.task.cur_hp['batchSize'],  # размер батча
                    'lr': state.task.cur_hp['lr'],  # скорость обучения

                    'Metric_achieved_result': scores[1],  # значение метрики на тестовой выборке
                    'curTrainingSubfolder': state.task.curTrainingSubfolder,  # папка, в которой хранятся результаты текущего обучения
                    'timeStat': [C_T.times],  # TODO: что сюда записывается, чем отличается от totalTime?
                    'totalTime': C_T.total_time,  # общее время обучения
                    'Additional_params': [{}]})  # дополнительные параметры

        state.task.history = state.task.history.append(new_row, ignore_index=True)

        state.task.history.to_csv(state.task.expPath + '/' + state.task.expName + '__History.csv', index=False)

        state.curState = 'GridStep'


@rule
class GridStep(Rule):
    """ Прием для перехода к следующей точке сетки """

    def can_apply(self, state):
        return state.task.taskCt == "train" and state.curState == 'GridStep'

    def apply(self, state):

        print('GRID')
        print('central hp', state.task.cur_C_hp)
        # print(state.task.cur_central_point, state.task.cur_training_point)
        # print(state.task.hp_grid)

        with open(state.logName, 'a') as log_file:
            print('\n', file=log_file)
            print(state.task.hp_grid, file=log_file)
            print('\n', file=log_file)
            print('GRID', file=log_file)
            print('central hp', state.task.cur_C_hp, file=log_file)

        if not hasattr(state.task, 'best_model') or state.task.hp_grid[tuple(state.task.cur_training_point)] > \
                state.task.hp_grid[tuple(state.task.best_model)]:
            state.task.best_model = state.task.cur_training_point.copy()
            state.task.best_num = state.task.counter

        # пройтись по окрестности центральной точки, на наличие необученного соседа
        checkN, cur = neighborhood(state.task.cur_central_point, state.task.hp_grid)

        if not checkN:

            print('neighboor exists')

            with open(state.logName, 'a') as log_file:
                print('neighboor exists', file=log_file)

            # если есть нулевой сосед
            state.task.cur_training_point = cur

            vC = state.task.cur_central_point
            vN = state.task.cur_training_point

            # print(vC, vN)

            if vC[0] != vN[0]:
                # print('total')

                # поменяла оптимизатор => меняем все три гиперпараметра по массивам
                state.task.cur_hp['optimizer'] = state.task.hyperParams['optimizer'][vN[0]]
                state.task.cur_hp['batchSize'] = state.task.hyperParams['batchSize'][vN[2]]

                if state.task.cur_hp['optimizer'] == 'SGD':
                    state.task.cur_hp['lr'] = state.task.hyperParams['lr_SGD'][vN[1]]
                else:
                    state.task.cur_hp['lr'] = state.task.hyperParams['lr_A_R'][vN[1]]

            elif vC[1] != vN[1]:
                # print('lr+bs')
                # поменяла learning rate => должен пропорционально поменяться batch size
                state.task.cur_hp['optimizer'] = state.task.cur_C_hp['optimizer']

                if state.task.cur_hp['optimizer'] == 'SGD':
                    state.task.cur_hp['lr'] = state.task.hyperParams['lr_SGD'][vN[1]]
                else:
                    state.task.cur_hp['lr'] = state.task.hyperParams['lr_A_R'][vN[1]]

                state.task.cur_hp['batchSize'] = int(state.task.cur_C_hp['batchSize'] * 2 ** (
                            (vN[2] - vC[2]) + (vN[1] - vC[1])))  # изменение батча из-за learning rate см бумаги

            else:
                # print('only bs')
                state.task.cur_hp['optimizer'] = state.task.cur_C_hp['optimizer']
                state.task.cur_hp['lr'] = state.task.cur_C_hp['lr']

                # если изменили только batch size, то просто умножили или поделили на два текущее значение
                # (предполагаю, что в центральной точке нормально ? в смысле без неправильных сдвигов)
                state.task.cur_hp['batchSize'] = int(state.task.cur_C_hp['batchSize'] * 2 ** (vN[2] - vC[2]))

            if state.task.cur_hp['batchSize'] > 64:
                state.task.cur_hp['batchSize'] = 64

            # print(state.task.cur_hp)
            # state.task.hp_grid[tuple(vN)]=0.5

        else:
            # все соседи уже обработаны

            with open(state.logName, 'a') as log_file:
                print('checkN of our central point is equal -1 => Step', file=log_file)
                print(state.task.hp_grid, file=log_file)

            print('checkN of our central point is equal -1 => Step')
            print(state.task.hp_grid)

            if state.task.best_model == state.task.cur_central_point:

                with open(state.logName, 'a') as log_file:
                    print('state.task.best_model == state.task.cur_central_point -- local maximum', file=log_file)

                print('state.task.best_model == state.task.cur_central_point -- local maximum')
                # осмотрела всю окрестность, но лучше результата нет => попали в локальный максимум => сохранить реузультаты и закончить работу
                # ЛУЧШИЙ РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ЗАПИСАТЬ В БД
                best = state.task.history['Index'] == state.task.best_num  # является ли текущий индекс лучшим
                loc = state.task.history.loc[best]
                prefix = f"{_data_dir}/trainedNN/{loc['expName'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}"
                myDB.add_model_record(task_type=state.task.taskType,
                                      categories=list(state.task.objects),  # категории объектов
                                      model_address=prefix + '__Model.h5',  # путь к файлу, где лежит модель
                                      metrics={list(state.task.goal.keys())[0]: loc['Metric_achieved_result'].iloc[0]},  # значения метрик
                                      history_address=prefix + '__History.h5')  # путь к файлу, где лежит история обучения
                state.curState = 'Done'
                return

            else:
                # поменяла центральную точку => надо осмотреть её окрестность

                with open(state.logName, 'a') as log_file:
                    print('have found a new central point => change it anf it\'s hyperParams ', file=log_file)

                print('have found a new central point => change it anf it\'s hyperParams ')
                state.task.cur_central_point = state.task.best_model.copy()
                state.task.cur_C_hp['optimizer'] = state.task.hyperParams['optimizer'][state.task.cur_central_point[0]]
                state.task.cur_C_hp['batchSize'] = state.task.hyperParams['batchSize'][state.task.cur_central_point[2]]

                if state.task.cur_C_hp['optimizer'] == 'SGD':
                    state.task.cur_C_hp['lr'] = state.task.hyperParams['lr_SGD'][state.task.cur_central_point[1]]
                else:
                    state.task.cur_C_hp['lr'] = state.task.hyperParams['lr_A_R'][state.task.cur_central_point[1]]

                checkN, cur = neighborhood(state.task.cur_central_point, state.task.hp_grid)

                if not checkN:

                    with open(state.logName, 'a') as log_file:
                        print('neighboor of a new central point exists', file=log_file)

                    print('neighboor of a new central point exists')

                    # если есть нулевой сосед
                    state.task.cur_training_point = cur

                    vC = state.task.cur_central_point
                    vN = state.task.cur_training_point

                    # print(vC, vN)

                    if vC[0] != vN[0]:
                        # print('total')

                        # поменяла оптимизатор => меняем все три гиперпараметра по массивам
                        state.task.cur_hp['optimizer'] = state.task.hyperParams['optimizer'][vN[0]]
                        state.task.cur_hp['batchSize'] = state.task.hyperParams['batchSize'][vN[2]]

                        if state.task.cur_hp['optimizer'] == 'SGD':
                            state.task.cur_hp['lr'] = state.task.hyperParams['lr_SGD'][vN[1]]
                        else:
                            state.task.cur_hp['lr'] = state.task.hyperParams['lr_A_R'][vN[1]]

                    elif vC[1] != vN[1]:
                        # print('lr+bs')
                        # поменяла learning rate => должен пропорционально поменяться batch size
                        state.task.cur_hp['optimizer'] = state.task.cur_C_hp['optimizer']

                        if state.task.cur_hp['optimizer'] == 'SGD':
                            state.task.cur_hp['lr'] = state.task.hyperParams['lr_SGD'][vN[1]]
                        else:
                            state.task.cur_hp['lr'] = state.task.hyperParams['lr_A_R'][vN[1]]

                        state.task.cur_hp['batchSize'] = int(state.task.cur_C_hp['batchSize'] * 2 ** ((vN[2] - vC[2]) + (vN[1] - vC[1])))  # изменение батча из-за learning rate см бумаги

                    else:
                        # print('only bs')
                        state.task.cur_hp['optimizer'] = state.task.cur_C_hp['optimizer']
                        state.task.cur_hp['lr'] = state.task.cur_C_hp['lr']

                        # если изменили только batch size, то просто умножили или поделили на два текущее значение
                        # (предполагаю, что в центральной точке нормально ? в смысле без неправильных сдвигов)
                        state.task.cur_hp['batchSize'] = int(state.task.cur_C_hp['batchSize'] * 2 ** (vN[2] - vC[2]))

                    if state.task.cur_hp['batchSize'] > 64:
                        state.task.cur_hp['batchSize'] = 64

                    # print(state.task.cur_hp)
                    # tate.task.hp_grid[tuple(vN)]=0.5

                else:

                    with open(state.logName, 'a') as log_file:
                        print('set a new central point, but it\'s neighborhood has already filled => local maximum',
                              file=log_file)

                    print('set a new central point, but it\'s neighborhood has already filled => local maximum')
                    # осмотрела всю окрестность, но лучше результата нет => попали в локальный максимум => сохранить реузультаты и закончить работу
                    # ЛУЧШИЙ РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ЗАПИСАТЬ В БД
                    best = state.task.history['Index'] == state.task.best_num  # является ли текущий индекс лучшим
                    loc = state.task.history.loc[best]
                    prefix = f"{_data_dir}/trainedNN/{loc['expName'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}/{loc['curTrainingSubfolder'].iloc[0]}"
                    myDB.add_model_record(task_type=state.task.taskType,
                                          categories=list(state.task.objects),
                                          model_address=prefix + '__Model.h5',
                                          metrics={list(state.task.goal.keys())[0]: loc['Metric_achieved_result'].iloc[0]},
                                          history_address=prefix + 'History.h5')  # TODO: почему здесь "History.h5", а не "__History.h5"?

                    state.curState = 'Done'
                    return

        state.task.counter += 1
        state.curState = 'Training_Gen'
        if state.task.counter > 600:
            state.curState = 'Done'

