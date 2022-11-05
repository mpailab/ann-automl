from datetime import datetime
import hashlib
import itertools
import math
import os
import time
import warnings
from collections import defaultdict

import keras
import numpy as np
import pandas as pd
from pytz import timezone

from . import db_module
from .solver import printlog
from ..utils.process import pcall

_data_dir = 'data'
_db_file = 'tests.sqlite'

nnDB = db_module.DBModule(dbstring=f'sqlite:///{_db_file}')  # TODO: уточнить путь к файлу базы данных


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


def set_db_file(db_file):
    """ Set the database file name """
    global _db_file
    _db_file = db_file


# !!! гиперпараметры и их значения сгенерированы автоматически !!!
# TODO: проверить их на корректность
augmen_params_list = {
    'rotation_range': {'type': 'float_range', 'range': [0, 180], 'default': None, 'name': 'угол поворота'},
    'width_shift_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'name': 'сдвиг по ширине'},
    'height_shift_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'name': 'сдвиг по высоте'},
    'shear_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'name': 'угол наклона'},
    'zoom_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'name': 'масштаб'},
    'channel_shift_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'name': 'сдвиг по цвету'},
    'fill_mode': {'type': 'str', 'values': ['constant', 'nearest', 'reflect', 'wrap'], 'default': 'constant',
                  'name': 'режим заполнения'},
    'cval': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'name': 'значение заполнения'},
    'horizontal_flip': {'type': 'bool', 'default': False, 'name': 'горизонтальное отражение'},
    'vertical_flip': {'type': 'bool', 'default': False, 'name': 'вертикальное отражение'},
    'rescale': {'type': 'float', 'range': [0, 1], 'default': None, 'name': 'масштабирование'},
    'preprocessing_function': {'type': 'str', 'values': ['auto', 'None', 'rescale', 'preprocess_input'],
                               'default': 'auto', 'name': 'функция предобработки'},
    'data_format': {'type': 'str', 'values': ['channels_last', 'channels_first'], 'default': 'channels_last',
                    'name': 'формат данных'},
}


nn_hparams = {
    'batch_size': {
        'type': 'int',
        'range': [1, 128],
        'default': 32,
        'step': 2,
        'scale': 'log',
        'name': "размер батча",
        'description': "Размер батча, используемый при обучении нейронной сети"
    },
    'epochs': {'type': 'int', 'range': [10, 1000], 'default': 150, 'step': 10, 'scale': 'lin', 'name': "количество эпох"},
    'optimizer': {
        'type': 'str',
        'values': {
            'Adam': {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon']},
            'SGD': {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum']},
            'RMSprop': {'params': ['rho', 'epsilon', 'momentum', 'centered']},
            'Adagrad': {'params': ['epsilon']},
            'Adadelta': {'params': ['rho', 'epsilon']},
            'Adamax': {'params': ['beta_1', 'beta_2', 'epsilon']},
            'Nadam': {'params': ['beta_1', 'beta_2', 'epsilon']},
        },
        'default': 'Adam',
        'name': "оптимизатор",
        'description': "Оптимизатор, используемый при обучении нейронной сети:\n"
    },
    'learning_rate': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'step': 2, 'scale': 'log',
                      'name': "скорость обучения"},
    'decay': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'step': 0.01, 'scale': 'lin',
              'name': 'декремент скорости обучения'},
    'activation': {'type': 'str', 'values': ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                             'sigmoid', 'hard_sigmoid', 'linear'],
                   'default': 'relu', 'name': 'функция активации'},
    'loss': {'type': 'str', 'values': ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                                       'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge',
                                       'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                                       'binary_crossentropy', 'kullback_leibler_divergence', 'poisson',
                                       'cosine_proximity'], 'default': 'mean_squared_error', 'name': 'функция потерь'},
    'metrics': {'type': 'str', 'values': ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                                          'sparse_categorical_accuracy', 'top_k_categorical_accuracy',
                                          'sparse_top_k_categorical_accuracy'],
                'default': 'accuracy', 'name': 'метрика'},
    'dropout': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'name': 'dropout'},
    # доля нейронов, которые отключаются при обучении
    'kernel_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                     'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                     'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                     'he_uniform'],
                           'default': 'glorot_uniform', 'name': 'инициализатор весов'},
    'bias_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                   'he_uniform'], 'default': 'zeros', 'name': 'инициализатор смещений'},
    'kernel_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                           'default': 'auto', 'name': 'регуляризатор весов'},
    'bias_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                         'default': 'auto', 'name': 'регуляризатор смещений'},
    'activity_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                             'default': 'auto', 'name': 'регуляризатор активации'},
    'kernel_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                          'default': 'auto', 'name': 'ограничение весов'},
    'bias_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                        'default': 'auto', 'name': 'ограничение смещений'},
    'augmen_params': {'type': 'dict', 'default': {},
                      'params': augmen_params_list,
                      'name': 'параметры аугментации'},

    # conditional parameters (for optimizers)
    'nesterov': {'type': 'bool', 'default': False, 'name': 'Nesterov momentum', 'cond': True},  # для SGD
    'centered': {'type': 'bool', 'default': False, 'name': 'centered', 'cond': True},  # для RMSprop
    'amsgrad': {'type': 'bool', 'default': False, 'name': 'amsgrad для Adam', 'cond': True},  # для Adam

    'momentum': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'step': 0.01, 'scale': 'lin',
                 'name': 'momentum', 'cond': True},  # момент для SGD
    'rho': {'type': 'float', 'range': [0.5, 0.99], 'default': 0.9, 'name': 'rho', 'cond': True,
            'step': 2**0.25, 'scale': '1-log'},  # коэффициент затухания для RMSprop
    'epsilon': {'type': 'float', 'range': [1e-8, 1e-1], 'default': 1e-7, 'step': 10, 'scale': 'log',
                'name': 'epsilon', 'cond': True},  # для RMSprop, Adagrad, Adadelta, Adamax, Nadam
    'beta_1': {'type': 'float', 'range': [0.5, 0.999], 'default': 0.9, 'name': 'beta_1 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
    'beta_2': {'type': 'float', 'range': [0.5, 0.9999], 'default': 0.999, 'name': 'beta_2 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
}


tune_hparams = {
    'method': {'type': 'str', 'values': {'grid': {'params': ['radius', 'metric', 'start']}}},
    # conditional parameters:
    'radius': {'type': 'int', 'range': [1, 5], 'default': 1},
    'grid_metric': {'type': 'str', 'values': ['l1', 'max'], 'default': 'l1'},
    'start_point': {'type': 'str', 'values': ['random', 'auto'], 'default': 'auto'},
}


def get_hparams(params_table, **kwargs):
    res = {key: value['default'] for key, value in params_table.items()}
    res.update(kwargs)
    cond_active = set()
    for key, value in kwargs.items():
        if 'values' in params_table[key]:
            if value not in params_table[key]['values']:
                raise ValueError(f'Значение {value} не входит в список возможных значений параметра {key}')
            if isinstance(params_table[key]['values'], dict):
                cond_active.update(params_table[key]['values'][value])

    for key, value in params_table.items():
        if 'cond' in value and value['cond']:
            if key not in cond_active:
                if key in kwargs:
                    warnings.warn(f'Зависимый параметр {key} не используется при заданных значениях других параметров')
                del res[key]
    return res


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
        self.start_of_train = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs=None):
        self.total_time = (time.time() - self.start_of_train)


def create_data_subset(objects, temp_dir='tmp', crop_bbox=True):
    """ Создание подвыборки данных для обучения

    Parameters
    ----------
    objects : list
        Список категорий, для которых необходимо создать подвыборку
    temp_dir : str
        Путь к папке, в которой будут созданы подвыборки
    crop_bbox : bool
        Если True, то изображения будут обрезаны по границам объектов
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    return nnDB.load_specific_categories_annotations(list(objects), normalizeCats=True,
                                                     splitPoints=[0.7, 0.85],
                                                     curExperimentFolder=temp_dir,
                                                     crop_bbox=crop_bbox,
                                                     cropped_dir=temp_dir + '/crops/')


def create_generators(model, data, augmen_params, batch_size):
    """
    Создание генераторов изображений по заданным в curStrategy параметрам аугментации
    В этот прием попадем как при первичном обучении, так и при смене параметров аугментации после обучения модели
    """
    df_train = pd.read_csv(data['train'])
    df_validate = pd.read_csv(data['validate'])
    df_test = pd.read_csv(data['test'])
    # Определяем размерность входных данных из модели
    flow_args = dict(target_size=model.input_shape[1:3], class_mode='raw', batch_size=batch_size)

    data_gen = keras.preprocessing.image.ImageDataGenerator(augmen_params)

    train_generator = data_gen.flow_from_dataframe(df_train, x_col=list(df_train.columns)[0],
                                                   y_col=list(df_train.columns)[1], **flow_args)
    val_generator = data_gen.flow_from_dataframe(df_validate, x_col=list(df_validate.columns)[0],
                                                 y_col=list(df_validate.columns)[1], **flow_args)
    test_generator = data_gen.flow_from_dataframe(df_test, x_col=list(df_test.columns)[0],
                                                  y_col=list(df_test.columns)[1], **flow_args)

    return train_generator, val_generator, test_generator


def create_layer(type, **kwargs):
    return getattr(keras.layers, type)(**kwargs)


def create_model(base, last_layers):
    y = keras.models.load_model(f'{_data_dir}/architectures/{base}.h5')
    input_shape = y.input_shape[1:]
    x = keras.layers.Input(shape=input_shape)
    y = y(x)
    for layer in last_layers:
        y = create_layer(**layer)(y)

    return keras.models.Model(inputs=x, outputs=y)


class ExperimentHistory:
    def __init__(self, task, exp_name, exp_path, data):
        self.experiment_number = 0
        self.exp_name = exp_name
        self.exp_path = exp_path
        self.data = data
        self.task_type = task.task_type
        self.objects = task.objects

        self.history = pd.DataFrame(columns=['Index', 'task_type', 'objects', 'exp_name', 'pipeline', 'last_layers',
                                             'augmen_params', 'loss', 'metrics', 'epochs', 'stop_criterion', 'data',
                                             'optimizer', 'batch_size', 'learning_rate', 'metric_test_value',
                                             'train_subdir', 'time_stat', 'total_time', 'additional_params'])

        self.save()

    def add_row(self, params, metric, train_subdir, time_stat, total_time, save=True):
        self.experiment_number += 1
        row = ({'Index': self.experiment_number,  # номер эксперимента
                'task_type': self.task_type,  # тип задачи
                'objects': [self.objects],  # список объектов, на распознавание которых обучается модель
                'exp_name': self.exp_name,  # название эксперимента

                'pipeline': params['pipeline'],  # базовая часть модели
                'last_layers': params['last_layers'],  # последние слои модели
                'augmen_params': params['augmen_params'],  # параметры аугментации
                'loss': params['loss'],  # функция потерь
                'metrics': params['metrics'],  # метрика, по которой оценивается качество модели
                'epochs': params['epochs'],  # количество эпох обучения
                'stop_criterion': params['stop_criterion'],  # критерий остановки обучения (TODO: не используется, исправить!!!)
                'data': params['data'],  # набор данных, на котором проводится обучение

                'optimizer': params['optimizer'],  # оптимизатор
                'batch_size': params['batch_size'],  # размер батча
                'learning_rate': params['learning_rate'],  # скорость обучения

                'metric_test_value': metric,  # значение метрики на тестовой выборке
                'train_subdir': train_subdir,  # папка, в которой хранятся результаты текущего обучения
                'time_stat': time_stat,  # список длительностей всех эпох обучения
                'total_time': total_time,  # общее время обучения
                'additional_params': [{}]})

        self.history.append(row, ignore_index=True)
        if save:
            self.save()

    def save(self):
        self.history.to_csv(self.exp_path + '/' + self.exp_name + '__History.csv', index=False)

    def get_best_model(self):
        best_model = self.history.loc[self.history['metric_test_value'].idxmax()]
        return best_model

    def get_best_model_path(self):
        best_model = self.get_best_model()
        return best_model['train_subdir'] + '/' + 'best_model.h5'

    def get_best_model_params(self):
        best_model = self.get_best_model()
        return {'optimizer': best_model['optimizer'],
                'batch_size': best_model['batch_size'],
                'learning_rate': best_model['learning_rate']}


class StopFlag:
    def __init__(self):
        self.flag = False

    def __call__(self):
        self.flag = True


class CheckStopCallback(keras.callbacks.Callback):
    def __init__(self, stop_flag):
        super().__init__()
        self.stop_flag = stop_flag

    def on_batch_end(self, batch, logs=None):
        if self.stop_flag.flag:
            self.model.stop_training = True


class NotifyCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        pcall('train_callback', 'batch', batch='batch', logs=logs, model=self.model)

    def on_epoch_end(self, epoch, logs=None):
        pcall('train_callback', 'epoch', epoch=epoch, logs=logs, model=self.model)

    def on_train_end(self, logs=None):
        pcall('train_callback', 'finish', logs=logs, model=self.model)

    def on_train_begin(self, logs=None):
        pcall('train_callback', 'start', logs=logs, model=self.model)


def fit_model(model, hparams, generators, cur_subdir, history=None, stop_flag=None):
    """
    Parameters
    ----------
    model: keras.models.Model
        модель, которую нужно обучить
    hparams: dict
        словарь с гиперпараметрами обучения
    generators: tuple
        кортеж из трех генераторов: train, val, test
    cur_subdir: str
        папка, в которой хранятся результаты текущего обучения
    history: Optional[ExperimentHistory]
        история экспериментов
    stop_flag: Optional[StopFlag]
        флаг, с помощью которого можно остановить обучение из другого потока
    Returns
    -------
    List[float]
        Достигнутые значения метрик на тестовой выборке во время обучения
    """

    optimizer, lr = hparams['optimizer'], hparams['learning_rate']
    opt_args = ['decay'] + hparams['optimizer']['values'][optimizer].get('params', [])
    kwargs = {arg: hparams[arg] for arg in opt_args if arg in hparams}
    optimizer = getattr(keras.optimizers, optimizer)(learning_rate=lr, **kwargs)
    model.compile(optimizer=optimizer, loss=hparams['loss'], metrics=[hparams['metrics']])

    # set up callbacks
    check_metric = 'val_' + hparams['metrics']
    c_log = keras.callbacks.CSVLogger(cur_subdir + '/Log.csv', separator=',', append=True)
    c_ch = keras.callbacks.ModelCheckpoint(cur_subdir + '/weights-{epoch:02d}.h5', monitor=check_metric, verbose=1,
                                           save_best_only=True, save_weights_only=False, mode='auto')
    c_es = keras.callbacks.EarlyStopping(monitor=check_metric, min_delta=0.001, mode='auto', patience=5)  # TODO: магические константы
    c_t = TimeHistory()
    callbacks = [c_log, c_ch, c_es, c_t, NotifyCallback()]
    if stop_flag is not None:
        callbacks.append(CheckStopCallback(stop_flag))

    # fit model
    model.fit(x=generators[0],
              steps_per_epoch=len(generators[0].filenames) // hparams['batch_size'],
              epochs=hparams['epochs'],
              validation_data=generators[1],
              callbacks=callbacks,
              validation_steps=len(generators[1].filenames) // hparams['batch_size'])

    # evaluate model
    scores = model.evaluate(generators[2], steps=None, verbose=1)

    # save results to history
    if history is not None:
        history.add_row(hparams, scores[1], cur_subdir, c_t.times, c_t.total_time, save=True)

    return scores


def create_and_train_model(hparams, data, cur_subdir, history=None, stop_flag=None):
    """
    Parameters
    ----------
    hparams: dict
        словарь с гиперпараметрами обучения
    data: tuple
        кортеж из трех генераторов: train, val, test
    cur_subdir: str
        папка, в которой хранятся результаты текущего обучения
    history: Optional[ExperimentHistory]
        история экспериментов
    stop_flag: Optional[StopFlag]
        флаг, с помощью которой можно остановить обучение из другого потока
    Returns
    -------
    List[float]
        Достигнутые значения метрик на тестовой выборке во время обучения
    """
    model = create_model(hparams['pipeline'], hparams['last_layers'])
    generators = create_generators(model, data, hparams['augmen_params'], hparams['batch_size'])
    return fit_model(model, hparams, generators, cur_subdir, history=history, stop_flag=stop_flag)


def train(nn_task, hparams, stop_flag=None, db_params=None):
    """
    Parameters
    ----------
    nn_task: NNTask
        задача обучения нейронной сети
    hparams: dict
        словарь с гиперпараметрами обучения
    stop_flag: Optional[StopFlag]
        флаг, с помощью которой можно остановить обучение из другого потока
    Returns
    -------
    List[float]
        Достигнутые значения метрик на тестовой выборке во время обучения
    """
    data = create_data_subset(nn_task.objects)
    exp_name, exp_dir = create_exp_dir('train', nn_task)
    history = ExperimentHistory(nn_task, exp_name, exp_dir, data)
    return create_and_train_model(hparams, data, exp_dir, history=history, stop_flag=stop_flag)


grid_hparams_space = {  # гиперпараметры, которые будем перебирать по сетке
    # TODO: объединить как-то с hyperparameters
    'optimizer': {'values': [
        ('Adam', {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon']}),
        ('SGD', {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum']}),
        ('RMSprop', {'params': ['rho', 'epsilon', 'momentum', 'centered']}),
    ]},
    # для каждого оптимизатора указывается, как другие гиперпараметры должны масштабироваться при смене оптимизатора
    'batch_size': {'range': [1, 1024], 'default': 1024, 'step': 2, 'scale': 'log', 'type': 'int'},
    'learning_rate': {'range': [0.000125, 0.064], 'default': 0.001, 'step': 2, 'scale': 'log', 'type': 'float'},
    'lr/batch_size': {'range': [0.00000125, 0.00128], 'default': 0.001, 'step': 2, 'scale': 'log', 'type': 'float'},
    # только один из двух параметров может быть задан: learning_rate или lr/batch_size
    'decay': {'type': 'float', 'range': [1/2**5, 1], 'default': 0.0, 'step': 2, 'scale': 'log', 'zero_point': 1},

    # conditonal params
    'amsgrad': {'values': [True, False], 'default': False, 'cond': True},   # для Adam
    'nesterov': {'values': [True, False], 'default': True, 'cond': True},   # для SGD
    'centered': {'values': [True, False], 'default': False, 'cond': True},  # для RMSprop

    'beta_1': {'range': [0.5, 0.999], 'default': 0.9, 'cond': True, 'step': 2, 'scale': '1-log'},     # для Adam
    'beta_2': {'range': [0.5, 0.9999], 'default': 0.999, 'cond': True, 'step': 2, 'scale': '1-log'},  # для Adam
    'rho': {'range': [0.5, 0.9999], 'default': 0.9, 'cond': True, 'step': 2, 'scale': '1-log'},  # для RMSprop
    'epsilon': {'range': [1e-8, 1], 'default': 1e-7, 'cond': True, 'step': 10, 'scale': 'log'},  # для Adam, RMSprop
    'momentum': {'range': [0, 1], 'default': 0.0, 'cond': True, 'step': 0.1, 'scale': 'lin'},    # для SGD, RMSprop
}


def param_values(range=None, default=None, values=None, step=None, scale=None, zero_point=None, type=None, **kwargs):
    if range is not None:
        if scale == 'log':
            back = round(math.log(range[0]/default, step))
            forward = round(math.log(range[1]/default, step))
            res = [default * step ** i for i in range(back, forward + 1)]
        elif scale == '1-log':
            back = round(math.log((1-range[1])/default, step))
            forward = round(math.log((1-range[0])/default, step))
            res = [1-default * step ** i for i in range(forward, back-1, -1)]
        elif scale == 'lin':
            back = round((range[0] - default) / step)
            forward = round((range[1] - default) / step)
            res = [default + step * i for i in range(back, forward + 1)]
        else:
            raise ValueError(f'Unknown scale {scale}')
        if type == 'int':
            res = [int(round(x)) for x in res]
        if zero_point:
            res = [0] + res
        return res
    elif values is not None:
        return list(values)
    else:
        raise ValueError('Either `range` or `values` should be specified')


class HyperParamGrid:
    def __init__(self, hparams, tuned_params):
        """
        Parameters
        ----------
        hparams: dict
            Словарь с текущими значениями гиперпараметров
        tuned_params: List
            Набор гиперпараметров, которые будут подбираться
        """
        self.hparams = hparams
        self.tuned_params = tuned_params
        self.fixed_params = [p for p in hparams if p not in tuned_params]
        self.param_control = {p: grid_hparams_space[p]['values'] for p in tuned_params
                              if 'values' in grid_hparams_space[p] and isinstance(grid_hparams_space[p]['values'], dict)}
        active = set()
        for p in self.fixed_params:
            v = hparams[p]
            active.update(self.param_control.get(p, {}).get(v, ()))
        self.active = {p for p in self.tuned_params if not grid_hparams_space[p].get('cond', False) or p in active}

        self.axis = [param_values(**grid_hparams_space[param]) for param in tuned_params]

    def remove_inactive(self, point):  # replaces inactive dependent params with None
        active = {*self.active}
        for p, i, ax in zip(self.tuned_params, point, self.axis):
            if p in active:
                active.update(self.param_control.get(p, {}).get(ax[i], ()))

        return tuple(x if p in active else None for p, x in zip(self.tuned_params, point))

    def __call__(self, point):
        key = self.remove_inactive(point)
        pt_params = {p: ax[v] for p, v, ax in zip(self.tuned_params, key, self.axis) if v is not None}
        res = {**self.hparams, **pt_params}
        if 'lr/batch_size' in res:
            res['learning_rate'] = res['lr/batch_size'] * res['batch_size']
            del res['lr/batch_size']

        # apply parameter scaling depending on other parameters
        for p, pp in self.param_control.items():
            if p in res:
                for ppp, s in pp.get(res[p], {}).items():
                    if ppp in res:
                        res[ppp] *= s
        return res


def neighborhood_gen(c, shape, cat_axis, r, metric):
    """
    Функция возвращает все точки в окрестности центра, которые не выходят за границы сетки
    :param c: центр окрестности
    :param shape: размеры сетки
    :param cat_axis: типы величин по осям (0 -- числовая, 1 -- категориальная)
    :param r: радиус окрестности
    :param metric: метрика расстояния на сетке ('max' -- максимум, 'l1' -- манхэттенское расстояние)
    :returns: генератор точек окрестности точки c
    """
    if metric == 'max':
        ranges = [range(shape[i]) if cat_axis[i] else range(max(0, c[i] - r), min(shape[i], c[i] + r + 1)) for i in range(len(c))]
        return (v for v in itertools.product(*ranges) if v != c)
    elif metric == 'l1':
        def yield_rec(i, ri):
            if len(c) == i or ri == 0:
                yield c[i:]
            else:
                if cat_axis[i]:
                    for j in range(shape[i]):
                        for v in yield_rec(i + 1, ri - (j != c[i])):
                            yield (j,) + v
                else:
                    for j in range(max(0, c[i] - ri), min(shape[i], c[i] + ri + 1)):
                        for v in yield_rec(i + 1, ri - abs(j - c[i])):
                            yield (j,) + v

        return (v for v in yield_rec(0, r) if v != c)
    else:
        raise ValueError(f'Unknown metric: {metric}')


def grid_search_gen(grid_size, cat_axis, func, gridmap, start_point='random', grid_metric='l1', radius=1):
    """
    Оптимизирует функцию на сетке жадным алгоритмом.
    На каждом шаге выбирается точка с максимальным значением функции в окрестности текущей точки.
    Если максимум достигается в центре окрестности, оптимизация на этом шаге завершается.

    Parameters
    ----------
    grid_size: tuple
        Размер сетки, на которой производится оптимизация. Каждый элемент кортежа - это количество точек в соответствующей оси.
    cat_axis: tuple
        Типы величин по осям (0 -- числовая, 1 -- категориальная)
    func: callable
        Функция, которую нужно оптимизировать.
    gridmap: callable
        Функция, которая преобразует точку сетки в кортеж (key, args, kwargs), где
        key - ключ для кэширования, args и kwargs - аргументы функции func.
    start_point: Union[tuple, str]
        Начальная точка. Если 'random', то начальная точка выбирается случайно.
    grid_metric: str
        Метрика, по которой определяется расстояние между точками сетки ('l1' или 'max').
    radius: int
        Радиус окрестности, в которой производится поиск лучшей точки.

    Returns
    -------
    Generator[tuple]
        Тройка, в которой первый элемент -- кортеж с координатами текущей точки,
        второй элемент -- значение функции в этой точке, третий элемент -- является ли точка локальным максимумом.
    """
    if start_point == 'random':
        start_point = tuple(np.random.randint(0, grid_size[i]) for i in range(len(grid_size)))
    else:
        start_point = tuple(start_point)

    cur_point = start_point
    key, args, kwargs = gridmap(cur_point)
    cur_value = func(*args, **kwargs)
    cache = {key: cur_value}
    yield cur_point, cur_value, True

    while True:
        printlog(f'Current point: {cur_point}, value: {cur_value}')
        best_point = cur_point
        best_value = cur_value
        for point in neighborhood_gen(cur_point, grid_size, cat_axis, radius, grid_metric):
            key, args, kwargs = gridmap(point)
            if key is None:  # если точка не входит в область определения функции
                continue
            if key not in cache:
                cache[key] = func(*args, **kwargs)
                yield point, cache[key], cache[key] > best_value
            if cache[key] > best_value:
                best_point = point
                best_value = cache[key]

        if best_point == cur_point:
            break

        cur_point = best_point
        cur_value = best_value


def hparams_grid_tune(nn_task, data, exp_name, exp_dir, hparams, tuned_params, stop_flag=None,
                      start_point='random', grid_metric='l1', radius=1):
    """
    Оптимизирует параметры нейронной сети на сетке.

    Parameters
    ----------
    nn_task: NNTask
        Задача, для которой оптимизируются параметры.
    data: tuple
        Кортеж, с генераторами для обучения, валидации и тестирования.
    exp_name: str
        Имя эксперимента.
    exp_dir: str
        Путь к директории, в которой сохраняются результаты оптимизации.
    hparams: dict
        Исходные гиперпараметры, часть из них будет оптимизироваться.
    tuned_params: list
        Параметры, которые будут оптимизироваться.
    stop_flag: optional StopFlag
        Флаг, который можно использовать для остановки оптимизации.
    start_point: str
        Начальная точка. Если 'random', то начальная точка выбирается случайно.
    grid_metric: str
        Метрика, по которой определяется расстояние между точками сетки ('l1' или 'max').
    radius: int
        Радиус окрестности, в которой производится поиск лучшей точки.
    """
    grid = HyperParamGrid(hparams, tuned_params)
    grid_size = list(map(len, grid.axis))
    cat_axis = ['values' in grid_hparams_space[p] for p in tuned_params]

    history = ExperimentHistory(nn_task, exp_name, exp_dir, data)

    def fit_and_get_score(params):
        scores = create_and_train_model(params, data, exp_dir, history=history, stop_flag=stop_flag)
        return scores[1]

    best_point, best_value = None, None
    for point, value, is_max in grid_search_gen(grid_size, cat_axis, fit_and_get_score,
                                                grid, start_point, grid_metric, radius):
        if stop_flag is not None and stop_flag.stop:
            break
        printlog(f"Evaluated point: {point}, value: {value}")
        pcall('tune_step', point, value)
        if is_max:
            best_point, best_value = point, value
            if not nn_task.goals.get('maximize', True) and best_value >= nn_task.goals['target']:
                break

    printlog(f"Best point: {best_point}, value: {best_value}")
    if best_value is not None and best_value >= nn_task.goals['target']:
        printlog("achieved target score")
    else:
        printlog("did not achieve target score")

    return best_point, best_value


def tune(nn_task, tuned_params, method, hparams=None, stop_flag=None, **kwargs):
    exp_name, exp_path = create_exp_dir(f'tune_{method}', nn_task)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    printlog(f"Experiment path: {exp_path}")
    if hparams is None:
        # взять дефолтные значения
        hparams = get_hparams(nn_hparams)

    data = create_data_subset(nn_task.objects)
    if method == 'grid':
        tune_func = hparams_grid_tune
    else:
        raise ValueError(f'Unknown tuning method: {method}')
    return tune_func(nn_task, data, exp_name, exp_path, hparams, tuned_params, stop_flag=stop_flag, **kwargs)


def create_exp_dir(prefix, nn_task):
    obj_set = sorted(nnDB.get_cat_IDs_by_names(list(nn_task.objects)))
    if len(obj_set) > 10:
        obj_set = obj_set[:10]+['etc']
    obj_str = '_'.join(map(str, obj_set))
    msk = timezone('Europe/Moscow')
    msk_time = datetime.now(msk)
    tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')
    exp_name = f'{prefix}_{obj_str}_DT_{tt}'
    exp_path = f"{_data_dir}/trainedNN/{exp_name}"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    return exp_name, exp_path
