import inspect
import json
import pickle
import random
import shutil
from datetime import datetime
import itertools
import math
import os
import time
import warnings
from typing import List, Tuple
import glob

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers
import keras_cv
from keras_cv import bounding_box
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
import resource
from keras_cv import visualization
import tqdm

from . import db_module
from .solver import printlog
from ..utils.process import pcall
from ..utils.thread_wrapper import ObjectWrapper
import atexit
from ast import literal_eval
import cv2
import imagesize

os.environ["SM_FRAMEWORK"] = "tf.keras" #before the import
import segmentation_models as sm


_data_dir = 'data'
_db_file = 'datasets.sqlite'

nnDB = db_module.DBModule(dbstring=f'sqlite:///{_db_file}')


def close_db():
    """ Закрывает соединение с базой данных """
    global nnDB
    if nnDB is not None:
        print('Close database')
        nnDB.close()
        nnDB = None


atexit.register(close_db)


_emulation = False  # флаг отладочного режима, когда не выполняются долгие операции


def set_emulation(emulation=True):
    """ Устанавливает или сбрасывает флаг эмуляции выполнения долгих операций """
    global _emulation
    _emulation = emulation


def set_db(db):
    """ Устанавливает объект базы данных (DBModule или ObjectWrapper над DBModule,
    если планируется использовать базу данных в многопоточном режиме) """
    global nnDB
    nnDB = db


class db_context:
    """ Контекстный менеджер для установки глоабльной базы данных """
    def __init__(self, db):
        global nnDB
        self._db = db

    def __enter__(self):
        global nnDB
        self._old_db = nnDB
        nnDB = self._db

    def __exit__(self, exc_type, exc_val, exc_tb):
        global nnDB
        nnDB.close()
        nnDB = self._old_db
        self._old_db = None


def cur_db():
    """ Возвращает текущий объект базы данных (DBModule) """
    global nnDB
    return nnDB


def set_multithreading_mode(mode=True):
    """ Set the multithreading mode """
    global nnDB
    if mode and not isinstance(nnDB, ObjectWrapper):
        nnDB.close()
        nnDB = ObjectWrapper(db_module.DBModule, dbstring=f'sqlite:///{_db_file}')
        print('Enter multithreading mode')
    elif not mode and isinstance(nnDB, ObjectWrapper):
        nnDB.close()
        nnDB.join_thread()
        nnDB = db_module.DBModule(dbstring=f'sqlite:///{_db_file}')
        print('Exit multithreading mode')


class multithreading_mode:
    """ Context manager for multithreading mode """
    def __enter__(self):
        set_multithreading_mode(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_multithreading_mode(False)


def set_data_dir(data_dir):
    """
    Set the data directory. Data directory contains the following subdirectories:
        - architecures: contains the neural network architectures
        - datasets: contains the datasets
        - trainedNN: contains the trained neural networks
    """
    global _data_dir
    _data_dir = data_dir


def tensorboard_logdir():
    """ Returns the directory for tensorboard logs """
    return f'{_data_dir}/tensorboard_logs'


def set_db_file(db_file):
    """ Set the database file name """
    global _db_file
    _db_file = db_file


pretrained_models = {
    'vgg16': tf.keras.applications.vgg16.VGG16,
    'vgg19': tf.keras.applications.vgg19.VGG19,
    'resnet50': tf.keras.applications.resnet50.ResNet50,
    'resnet101': tf.keras.applications.resnet.ResNet101,
    'resnet152': tf.keras.applications.resnet.ResNet152,
    'resnet50v2': tf.keras.applications.resnet_v2.ResNet50V2,
    'resnet101v2': tf.keras.applications.resnet_v2.ResNet101V2,
    'resnet152v2': tf.keras.applications.resnet_v2.ResNet152V2,
    'inceptionv3': tf.keras.applications.inception_v3.InceptionV3,
    'inceptionresnetv2': tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
    'mobilenet': tf.keras.applications.mobilenet.MobileNet,
    'densenet121': tf.keras.applications.densenet.DenseNet121,
    'densenet169': tf.keras.applications.densenet.DenseNet169,
    'densenet201': tf.keras.applications.densenet.DenseNet201,
    'nasnetlarge': tf.keras.applications.nasnet.NASNetLarge,  # error when loading
    'nasnetmobile': tf.keras.applications.nasnet.NASNetMobile,  # error when loading
    'xception': tf.keras.applications.xception.Xception,
    'mobilenetv2': tf.keras.applications.mobilenet_v2.MobileNetV2,
    'efficientnetb0': tf.keras.applications.efficientnet.EfficientNetB0,
    'efficientnetb1': tf.keras.applications.efficientnet.EfficientNetB1,
    'efficientnetb2': tf.keras.applications.efficientnet.EfficientNetB2,
    'efficientnetb3': tf.keras.applications.efficientnet.EfficientNetB3,
    'efficientnetb4': tf.keras.applications.efficientnet.EfficientNetB4,
    'efficientnetb5': tf.keras.applications.efficientnet.EfficientNetB5,
    'efficientnetb6': tf.keras.applications.efficientnet.EfficientNetB6,
    'efficientnetb7': tf.keras.applications.efficientnet.EfficientNetB7,
}


_model_num_weights = {
    'vgg16': 14714688,
    'vgg19': 20024384,
    'resnet50': 23587712,
    'resnet101': 42658176,
    'resnet152': 58370944,
    'resnet50v2': 23564800,
    'resnet101v2': 42626560,
    'resnet152v2': 58331648,
    'inceptionv3': 21802784,
    'inceptionresnetv2': 54336736,
    'mobilenet': 3228864,
    'densenet121': 7037504,
    'densenet169': 12642880,
    'densenet201': 18321984,
    'xception': 20861480,
    'mobilenetv2': 2257984,
    'efficientnetb0': 4049571,
    'efficientnetb1': 6575239,
    'efficientnetb2': 7768569,
    'efficientnetb3': 10783535,
    'efficientnetb4': 17673823,
    'efficientnetb5': 28513527,
    'efficientnetb6': 40960143,
    'efficientnetb7': 64097687}


_model_input_shape = {
    'vgg16': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet18': (224, 224, 3),
    'resnet34': (224, 224, 3),
    'resnet50': (224, 224, 3),
    'resnet101': (224, 224, 3),
    'resnet152': (224, 224, 3),
    'resnet50v2': (224, 224, 3),
    'resnet101v2': (224, 224, 3),
    'resnet152v2': (224, 224, 3),
    'inceptionv3': (299, 299, 3),
    'inceptionresnetv2': (299, 299, 3),
    'mobilenet': (224, 224, 3),
    'densenet121': (224, 224, 3),
    'densenet169': (224, 224, 3),
    'densenet201': (224, 224, 3),
    'xception': (299, 299, 3),
    'mobilenetv2': (224, 224, 3),
    'efficientnetb0': (224, 224, 3),
    'efficientnetb1': (240, 240, 3),
    'efficientnetb2': (260, 260, 3),
    'efficientnetb3': (300, 300, 3),
    'efficientnetb4': (380, 380, 3),
    'efficientnetb5': (456, 456, 3),
    'efficientnetb6': (528, 528, 3),
    'efficientnetb7': (600, 600, 3)}


def get_additionnal_models(dir_path):
    """ Get additionnal models from a directory """
    models = []
    if not os.path.exists(dir_path):
        return models
    for file in os.listdir(dir_path):
        if file.endswith('.h5'):
            model_name = file[:-3]
            model_path = os.path.join(dir_path, file)
            models.append(model_name)
    return models


_models_order_by_num_weights = sorted(_model_num_weights, key=_model_num_weights.get)
_additional_models = get_additionnal_models(os.path.join(_data_dir, 'architectures'))


# !!! гиперпараметры и их значения сгенерированы автоматически !!!
# TODO: проверить их на корректность
augmen_params_list = {
    'rotation_range': {'type': 'float_range', 'range': [0, 180], 'default': None, 'title': 'угол поворота'},
    'width_shift_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'title': 'сдвиг по ширине'},
    'height_shift_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'title': 'сдвиг по высоте'},
    'shear_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'title': 'угол наклона'},
    'zoom_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'title': 'масштаб'},
    'channel_shift_range': {'type': 'float_range', 'range': [0, 1], 'default': None, 'title': 'сдвиг по цвету'},
    'fill_mode': {'type': 'str', 'values': ['constant', 'nearest', 'reflect', 'wrap'], 'default': 'constant',
                  'title': 'режим заполнения'},
    'cval': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'title': 'значение заполнения'},
    'horizontal_flip': {'type': 'bool', 'default': False, 'title': 'горизонтальное отражение'},
    'vertical_flip': {'type': 'bool', 'default': False, 'title': 'вертикальное отражение'},
    'rescale': {'type': 'float', 'range': [0, 1], 'default': None, 'title': 'масштабирование'},
    'data_format': {'type': 'str', 'values': ['channels_last', 'channels_first'], 'default': 'channels_last',
                    'title': 'формат данных'},
}


db_hparams = {
    'crop_bbox': {'type': 'bool', 'default': True, 'title': 'обрезать изображения по границам объектов'},
    'val_frac': {'type': 'float', 'range': [0.05, 0.5], 'default': 0.2,
                 'scale': 'lin', 'step': 0.05,
                 'title': 'доля валидационной выборки'},
    'test_frac': {'type': 'float', 'range': [0.05, 0.5], 'default': 0.2,
                  'scale': 'lin', 'step': 0.05,
                  'title': 'доля тестовой выборки'},
    'balance_by_min_category': {'type': 'bool', 'default': False,
                                'title': 'cбалансировать выборки по минимальному классу'},
}


nn_hparams = {
    'batch_size': {
        'type': 'int',
        'range': [1, 128],
        'default': 32,
        'step': 2,
        'scale': 'log',
        'title': "размер батча",
        'description': "Размер батча, используемый при обучении нейронной сети"
    },
    'epochs': {
        'type': 'int',
        'range': [10, 1000],
        'default': 150,
        'step': 10,
        'scale': 'lin',
        'title': "количество эпох"
    },
    'optimizer': {
        'type': 'str',
        'values': {  # TODO: проверить описание оптимизаторов
            'Adam': {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon'],
                     'description': 'Оптимизатор использует оценки первого и второго момента градиента'},
            'SGD': {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum'],
                    'description': 'Метод стохастического градиентного спуска'},
            'RMSprop': {'params': ['rho', 'epsilon', 'momentum', 'centered'],
                        'description': 'Адаптивный метод градиентного спуска, '
                                       'основанный на оценках второго момента градиента'},
            'Adagrad': {'params': ['epsilon'],
                        'description': 'Адаптивный метод градиентного спуска, основанный на сумме квадратов градиента'},
            'Adadelta': {'params': ['rho', 'epsilon'],
                         'description': 'Адаптивный метод градиентного спуска, '
                                        'основанный на сумме квадратов градиента и '
                                        'градиента на предыдущем шаге'},
            'Adamax': {'params': ['beta_1', 'beta_2', 'epsilon'],
                       'description': 'Адаптивный метод градиентного спуска, '
                                      'основанный на оценках первого и максимального второго моментов градиента'},
            'Nadam': {'params': ['beta_1', 'beta_2', 'epsilon'],
                      'description': 'Адаптивный метод градиентного спуска, '
                                     'основанный на оценках первого и взвешенного второго моментов градиента'}
        },
        'default': 'Adam',
        'title': "оптимизатор",
        'description': "Оптимизатор, используемый при обучении нейронной сети"
    },
    'model_arch': {'type': 'str', 'values': _models_order_by_num_weights + _additional_models,
                   'default': 'resnet50', 'title': 'архитектура нейронной сети'},
    'transfer_learning': {'type': 'bool', 'default': False,
                          'title': 'дообучить предобученную нейронную сеть'},
    'learning_rate': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'step': 2, 'scale': 'log',
                      'title': "скорость обучения"},
    'fine_tune_lr_div': {'type': 'float', 'range': [1.0, 1e3], 'default': 10, 'step': 10, 'scale': 'log',
                         'title': "коэффициент для скорости fine-tune при дообучении"},
    # 'decay': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'step': 0.01, 'scale': 'lin',
    #           'title': 'декремент скорости обучения'},
    'activation': {'type': 'str', 'values': ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                             'sigmoid', 'hard_sigmoid', 'linear'],
                   'default': 'relu', 'title': 'функция активации'},
    'loss': {
        'type': 'str',
        'values': {
            # TODO: проверить и дополнить описания. Текущие описания переведены или сгенерированы автоматически
            'mean_squared_error': {
                'description': 'Среднеквадратичная ошибка. '
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие'},
            'mean_absolute_error': {
                'description': 'Средняя абсолютная ошибка. '
                               'Используется при выполнении регрессии, если не требуется, '
                               'чтобы выбросы играли большую роль. Это также может быть полезно, когда '
                               'распределение имеет несколько максимумов, '
                               'и желательно иметь прогнозы в окрестности одного из них, а не в среднем.'},
            'mean_absolute_percentage_error': {'description': ''},
            'mean_squared_logarithmic_error': {
                'description': 'Среднеквадратичная логарифмическая ошибка.'
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и не требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие, '
                               'в тех случаях, когда диапазон целевого значения велик.'},

            'squared_hinge': {
                'description': 'Используется в бинарных задачах классификации, и когда не важно знать, '
                               'насколько уверен классификатор в классификации. '
                               'Используется в сочетании с функцией активации tanh() на последнем слое.'},
            'hinge': {'description': ''},
            'categorical_hinge': {'description': ''},

            'binary_crossentropy': {
                'description': 'Используется в задачах бинарной классификации, когда целевая переменная '
                               'представляет собой бинарные метки, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации sigmoid() на последнем слое.'},
            'categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой вероятности, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'sparse_categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой индексы классов, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'kullback_leibler_divergence': {
                'description': 'Используется в задачах классификации, когда целевая переменная '
                               'представляет собой вероятности, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'poisson': {'description': 'Используется в задачах классификации, когда целевая переменная '
                                       'представляет собой счетчики, которые должны быть предсказаны. '
                                       'Используется в сочетании с функцией активации exp() на последнем слое.'},
            'logcosh': {'description': ''},
            'cosine_proximity': {'description': ''}
        },
        'default': 'mean_squared_error',
        'title': 'функция потерь'
    },
    'metrics': {
        'type': 'str',
        'values': ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                   'sparse_categorical_accuracy', 'top_k_categorical_accuracy',
                   'sparse_top_k_categorical_accuracy'],
        'default': 'accuracy',
        'title': 'метрика'
    },
    'dropout': {'type': 'float', 'range': [0, 1], 'step': 0.01, 'scale': 'lin', 'default': 0.0, 'title': 'dropout'},
    # доля нейронов, которые отключаются при обучении
    'kernel_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                     'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                     'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                     'he_uniform'],
                           'default': 'glorot_uniform', 'title': 'инициализатор весов'},
    'bias_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                   'he_uniform'], 'default': 'zeros', 'title': 'инициализатор смещений'},
    'kernel_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                           'default': 'auto',
                           'title': 'регуляризатор весов',
                           'description': ''},
    'bias_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                         'default': 'auto',
                         'title': 'регуляризатор смещений',
                         'description': ''},
    'activity_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                             'default': 'auto',
                             'title': 'регуляризатор активации',
                             'description': ''},
    'kernel_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                          'default': 'auto',
                          'title': 'ограничение весов',
                          'description': ''},
    'bias_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                        'default': 'auto',
                        'title': 'ограничение смещений',
                        'description': ''},
    'augmen_params': {'type': 'dict', 'default': {},
                      'params': augmen_params_list,
                      'title': 'параметры аугментации',
                      'description': 'Параметры аугментации изображений.'},

    # early stopping
    'early_stopping': {'type': 'bool', 'default': True, 'title': 'Ранняя остановка',
                       'description': 'Если в течение определенного количества эпох не происходит улучшения '
                                      'метрики, то обучение прекращается.'},
    'patience': {'type': 'int', 'range': [1, 100], 'step': 1, 'scale': 'lin', 'default': 5,
                 'title': 'Сколько эпох ждать',
                 'description': 'Количество эпох, в течение которых метрика не улучшается, '
                                'после чего обучение прекращается.'},
    'min_delta': {'type': 'float', 'range': [0, 0.1], 'step': 0.001, 'scale': 'lin', 'default': 0.001,
                  'title': 'Минимальное улучшение',
                  'description': 'Минимальное изменение метрики, которое считается улучшением.'},

    # dataset params
    **db_hparams,

    # conditional parameters (for optimizers)
    'nesterov': {'type': 'bool', 'default': False, 'title': 'Nesterov momentum', 'cond': True},  # для SGD
    'centered': {'type': 'bool', 'default': False, 'title': 'centered', 'cond': True},  # для RMSprop
    'amsgrad': {'type': 'bool', 'default': False, 'title': 'amsgrad для Adam', 'cond': True},  # для Adam

    'momentum': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'step': 0.01, 'scale': 'lin',
                 'title': 'momentum', 'cond': True},  # момент для SGD
    'rho': {'type': 'float', 'range': [0.5, 0.99], 'default': 0.9, 'title': 'rho', 'cond': True,
            'step': 2**0.25, 'scale': '1-log'},  # коэффициент затухания для RMSprop
    'epsilon': {'type': 'float', 'range': [1e-8, 1e-1], 'default': 1e-7, 'step': 10, 'scale': 'log',
                'title': 'epsilon', 'cond': True},  # для RMSprop, Adagrad, Adadelta, Adamax, Nadam
    'beta_1': {'type': 'float', 'range': [0.5, 0.999], 'default': 0.9, 'title': 'beta_1 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
    'beta_2': {'type': 'float', 'range': [0.5, 0.9999], 'default': 0.999, 'title': 'beta_2 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
}

dec_models = {
    'resnet50_imagenet',
    'resnet50_v2_imagenet',
    'densenet121_imagenet',
    'densenet169_imagenet',
    'densenet201_imagenet',
    'efficientnetv2_b0_imagenet',
    'efficientnetv2_b1_imagenet',
    'efficientnetv2_b2_imagenet',
    'efficientnetv2_s_imagenet',
    'efficientnetv2_s_imagenet',
    'mobilenet_v3_large_imagenet',
    'mobilenet_v3_small_imagenet',
    'yolo_v8_xs_backbone_coco',
    'yolo_v8_s_backbone_coco',
    'yolo_v8_m_backbone_coco',
    'yolo_v8_l_backbone_coco',
    'yolo_v8_xl_backbone_coco'}



nnd_hparams = {
    'model_arch': {
        'type': 'str',
        'values': ['Yolov8', 'RetinaNet'],
        'default': 'RetinaNet',
        'title': 'архитектура сети-детектора'
    },

    'backbone': {
        'type': 'str',
         'values': dec_models,
         'default': 'yolo_v8_s_backbone_coco',#'resnet50_imagenet',
         'title': 'архитектура бэкбона для нейронной сети'
    },
    
    'batch_size': {
        'type': 'int',
        'range': [1, 128],
        'default': 4,
        'step': 2,
        'scale': 'log',
        'title': "размер батча",
        'description': "Размер батча, используемый при обучении нейронной сети"
    },

    'metrics': {
        'type': 'str',
        'values': ['box_loss', 'classification_loss', 'loss',
                   'percent_boxes_matched_with_anchor'],
        'default': 'loss',
        'title': 'метрика'
    },

    'epochs': {
        'type': 'int',
        'range': [10, 1000],
        'default': 10,
        'step': 10,
        'scale': 'lin',
        'title': "количество эпох"
    },

    'optimizer': {
        'type': 'str',
        'values': {  # TODO: проверить описание оптимизаторов
            'SGD': {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum','global_clipnorm'],
                    'description': 'Метод стохастического градиентного спуска'},
            'Adam': {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon'],
                     'description': 'Оптимизатор использует оценки первого и второго момента градиента'},
            'RMSprop': {'params': ['rho', 'epsilon', 'momentum', 'centered'],
                        'description': 'Адаптивный метод градиентного спуска, '
                                       'основанный на оценках второго момента градиента'},
            'Adagrad': {'params': ['epsilon'],
                        'description': 'Адаптивный метод градиентного спуска, основанный на сумме квадратов градиента'},
            'Adadelta': {'params': ['rho', 'epsilon'],
                         'description': 'Адаптивный метод градиентного спуска, '
                                        'основанный на сумме квадратов градиента и '
                                        'градиента на предыдущем шаге'},
            'Adamax': {'params': ['beta_1', 'beta_2', 'epsilon'],
                       'description': 'Адаптивный метод градиентного спуска, '
                                      'основанный на оценках первого и максимального второго моментов градиента'},
            'Nadam': {'params': ['beta_1', 'beta_2', 'epsilon'],
                      'description': 'Адаптивный метод градиентного спуска, '
                                     'основанный на оценках первого и взвешенного второго моментов градиента'}
        },
        'default': 'SGD',
        'title': "оптимизатор",
        'description': "Оптимизатор, используемый при обучении нейронной сети"
    },
    
    'learning_rate': {
        'type': 'float',
        'range': [1e-5, 1e-1],
        'default': 0.005,
        'step': 2,
        'scale': 'log',
        'title': "скорость обучения"
    },

    'classification_loss': {
        'type': 'str',
        'values': {
            # TODO: проверить и дополнить описания. Текущие описания переведены или сгенерированы автоматически
            'focal': {
                'description': 'Focal Loss. '
                               'Используется для классификации в задачах детектирования'},
            'mean_squared_error': {
                'description': 'Среднеквадратичная ошибка. '
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие'},
            'mean_absolute_error': {
                'description': 'Средняя абсолютная ошибка. '
                               'Используется при выполнении регрессии, если не требуется, '
                               'чтобы выбросы играли большую роль. Это также может быть полезно, когда '
                               'распределение имеет несколько максимумов, '
                               'и желательно иметь прогнозы в окрестности одного из них, а не в среднем.'},
            'mean_absolute_percentage_error': {'description': ''},
            'mean_squared_logarithmic_error': {
                'description': 'Среднеквадратичная логарифмическая ошибка.'
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и не требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие, '
                               'в тех случаях, когда диапазон целевого значения велик.'},

            'squared_hinge': {
                'description': 'Используется в бинарных задачах классификации, и когда не важно знать, '
                               'насколько уверен классификатор в классификации. '
                               'Используется в сочетании с функцией активации tanh() на последнем слое.'},
            'hinge': {'description': ''},
            'categorical_hinge': {'description': ''},

            'binary_crossentropy': {
                'description': 'Используется в задачах бинарной классификации, когда целевая переменная '
                               'представляет собой бинарные метки, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации sigmoid() на последнем слое.'},
            'categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой вероятности, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'sparse_categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой индексы классов, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'kullback_leibler_divergence': {
                'description': 'Используется в задачах классификации, когда целевая переменная '
                               'представляет собой вероятности, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'poisson': {'description': 'Используется в задачах классификации, когда целевая переменная '
                                       'представляет собой счетчики, которые должны быть предсказаны. '
                                       'Используется в сочетании с функцией активации exp() на последнем слое.'},
            'logcosh': {'description': ''},
            'cosine_proximity': {'description': ''}
        },
        'default': 'binary_crossentropy',#'focal',
        'title': 'функция потерь классификации'
    },

    'box_loss': {
        'type': 'str',
        'values': {
            'smoothl1': {
                'description': 'Smooth l1 Loss'
                               'Используется для регрессии ограничивающих прямоугольников в задачах детектирования'},
            'mean_squared_error': {
                'description': 'Среднеквадратичная ошибка. '
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие'},
            'mean_absolute_error': {
                'description': 'Средняя абсолютная ошибка. '
                               'Используется при выполнении регрессии, если не требуется, '
                               'чтобы выбросы играли большую роль. Это также может быть полезно, когда '
                               'распределение имеет несколько максимумов, '
                               'и желательно иметь прогнозы в окрестности одного из них, а не в среднем.'},
            'mean_absolute_percentage_error': {'description': ''},
            'mean_squared_logarithmic_error': {
                'description': 'Среднеквадратичная логарифмическая ошибка.'
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и не требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие, '
                               'в тех случаях, когда диапазон целевого значения велик.'},

            'squared_hinge': {
                'description': 'Используется в бинарных задачах классификации, и когда не важно знать, '
                               'насколько уверен классификатор в классификации. '
                               'Используется в сочетании с функцией активации tanh() на последнем слое.'},
            'hinge': {'description': ''},
            'categorical_hinge': {'description': ''},

            'binary_crossentropy': {
                'description': 'Используется в задачах бинарной классификации, когда целевая переменная '
                               'представляет собой бинарные метки, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации sigmoid() на последнем слое.'},
            'categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой вероятности, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'sparse_categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой индексы классов, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'kullback_leibler_divergence': {
                'description': 'Используется в задачах классификации, когда целевая переменная '
                               'представляет собой вероятности, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'poisson': {'description': 'Используется в задачах классификации, когда целевая переменная '
                                       'представляет собой счетчики, которые должны быть предсказаны. '
                                       'Используется в сочетании с функцией активации exp() на последнем слое.'},
            'logcosh': {'description': ''},
            'cosine_proximity': {'description': ''}
        },
        'default': 'ciou',#'smoothl1',
        'title': 'функция потерь для ограничивающих прямоугольников'
    },

    # dataset params
    **db_hparams,

    #augmen_params
    'augmen_params_flip': {
        'type': 'str', 
        'default': "horizontal",
        'values': {"horizontal", "vertical", "horizontal_and_vertical"},
        'title': 'параметры аугментации отражения',
        'description': 'Параметры аугментации отражения.'},

    'bounding_box_format': {
        'type': 'str',
        'default': "xywh",
        'values': {"xywh", "xyxy", "rel_xyxy", "center_xywh"},
        'title': 'формат ограничивающих прямоугольников',
        'description': 'Параметры аугментации изображений.'},
    'resize_min': {
        'type': 'float',
        'default': 0.75,
        'step': 0.05,
        'range': [0,3],
        'title': 'scale_factor_min',
        'description': 'Параметры аугментации изображений.'},
    'resize_max': {
        'type': 'float',
        'default': 1.3,
        'step': 0.05,
        'range': [0,3],
        'title': 'scale_factor_max',
        'description': 'Параметры аугментации изображений.'},
    'pad_to_aspect_ratio': {
        'type': 'bool',
        'default': True,
        'values': {True, False},
        'title': 'pad_to_aspect_ratio',
        'description': 'Параметры аугментации изображений.'},  
    'drop_remainder': {
        'type': 'bool',
        'default': True,
        'values': {True, False},
        'title': 'drop_remainder',
        'description': 'drop_remainder.'}, 

    # early stopping
    'early_stopping': {'type': 'bool', 'default': False, 'title': 'Ранняя остановка',
                       'description': 'Если в течение определенного количества эпох не происходит улучшения '
                                      'метрики, то обучение прекращается.'},
    'patience': {'type': 'int', 'range': [1, 100], 'step': 1, 'scale': 'lin', 'default': 5,
                 'title': 'Сколько эпох ждать',
                 'description': 'Количество эпох, в течение которых метрика не улучшается, '
                                'после чего обучение прекращается.'},
    'min_delta': {'type': 'float', 'range': [0, 0.1], 'step': 0.001, 'scale': 'lin', 'default': 0.001,
                  'title': 'Минимальное улучшение',
                  'description': 'Минимальное изменение метрики, которое считается улучшением.'},


    # conditional parameters (for optimizers)
    'nesterov': {'type': 'bool', 'default': False, 'title': 'Nesterov momentum', 'cond': True},  # для SGD
    'centered': {'type': 'bool', 'default': False, 'title': 'centered', 'cond': True},  # для RMSprop
    'amsgrad': {'type': 'bool', 'default': False, 'title': 'amsgrad для Adam', 'cond': True},  # для Adam

    'momentum': {'type': 'float', 'range': [0, 1], 'default': 0.9, 'step': 0.01, 'scale': 'lin',
                 'title': 'momentum', 'cond': True},  # момент для SGD
    'rho': {'type': 'float', 'range': [0.5, 0.99], 'default': 0.9, 'title': 'rho', 'cond': True,
            'step': 2**0.25, 'scale': '1-log'},  # коэффициент затухания для RMSprop
    'epsilon': {'type': 'float', 'range': [1e-8, 1e-1], 'default': 1e-7, 'step': 10, 'scale': 'log',
                'title': 'epsilon', 'cond': True},  # для RMSprop, Adagrad, Adadelta, Adamax, Nadam
    'beta_1': {'type': 'float', 'range': [0.5, 0.999], 'default': 0.9, 'title': 'beta_1 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
    'beta_2': {'type': 'float', 'range': [0.5, 0.9999], 'default': 0.999, 'title': 'beta_2 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
    'global_clipnorm': {'type': 'float', 'range': [0, 15.0], 'step': 1, 'default': 10.0, 'title': 'global_clipnorm', 'cond': True},  # for allfor all


    #additional parameters
    'transfer_learning': {'type': 'bool', 'default': False,'title': 'дообучить предобученную нейронную сеть'},
    'activation': {'type': 'str', 'values': ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                             'sigmoid', 'hard_sigmoid', 'linear'],
                   'default': 'relu', 'title': 'функция активации'},
    'dropout': {'type': 'float', 'range': [0, 1], 'step': 0.01, 'scale': 'lin', 'default': 0.0, 'title': 'dropout'},
    # доля нейронов, которые отключаются при обучении
    'kernel_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                     'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                     'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                     'he_uniform'],
                           'default': 'glorot_uniform', 'title': 'инициализатор весов'},
    'bias_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                   'he_uniform'], 'default': 'zeros', 'title': 'инициализатор смещений'},
    'kernel_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                           'default': 'auto',
                           'title': 'регуляризатор весов',
                           'description': ''},
    'bias_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                         'default': 'auto',
                         'title': 'регуляризатор смещений',
                         'description': ''},
    'activity_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                             'default': 'auto',
                             'title': 'регуляризатор активации',
                             'description': ''},
    'kernel_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                          'default': 'auto',
                          'title': 'ограничение весов',
                          'description': ''},
    'bias_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                        'default': 'auto',
                        'title': 'ограничение смещений',
                        'description': ''},
}

nns_hparams = {
    'model_arch': {
        'type': 'str',
        'values': ['Unet', 'Linknet', 'PSPNet', 'FPN', 'local_model'],
        'default': 'Unet',
        'title': 'архитектура сегментационной нейронной сети'
    },

    'backbone': {
        'type': 'str',
         'values': ['vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152',
                    'seresnet18','seresnet34','seresnet50','seresnet101','seresnet152','resnext50' 'resnext101',
                    'seresnext50','seresnext101','senet154','densenet121','densenet169','densenet201',
                    'inceptionv3','inceptionresnetv2','mobilenet','mobilenetv2',
                    'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4','efficientnetb5','efficientnetb6','efficientnetb7'
                   ],
         'default': 'resnet18',
         'title': 'архитектура бэкбона нейронной сети'
    },

    'batch_size': {
        'type': 'int',
        'range': [1, 128],
        'default': 10,
        'step': 2,
        'scale': 'log',
        'title': "размер батча",
        'description': "Размер батча, используемый при обучении нейронной сети"
    },

    'epochs': {
        'type': 'int',
        'range': [10, 1000],
        'default': 10,
        'step': 10,
        'scale': 'lin',
        'title': "количество эпох"
    },

    'metrics': {
        'type': 'str',
        'values': ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                   'sparse_categorical_accuracy', 'top_k_categorical_accuracy',
                   'sparse_top_k_categorical_accuracy'],
        'default': 'accuracy',
        'title': 'метрика'
    },

    'loss': {
        'type': 'str',
        'values': {
            # TODO: проверить и дополнить описания. Текущие описания переведены или сгенерированы автоматически
            'mean_squared_error': {
                'description': 'Среднеквадратичная ошибка. '
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие'},
            'mean_absolute_error': {
                'description': 'Средняя абсолютная ошибка. '
                               'Используется при выполнении регрессии, если не требуется, '
                               'чтобы выбросы играли большую роль. Это также может быть полезно, когда '
                               'распределение имеет несколько максимумов, '
                               'и желательно иметь прогнозы в окрестности одного из них, а не в среднем.'},
            'mean_absolute_percentage_error': {'description': ''},
            'mean_squared_logarithmic_error': {
                'description': 'Среднеквадратичная логарифмическая ошибка.'
                               'Используется при выполнении регрессии, полагая, что цель, '
                               'обусловленная входными данными, нормально распределена, и не требуется, '
                               'чтобы большие ошибки наказывались значительно больше, чем маленькие, '
                               'в тех случаях, когда диапазон целевого значения велик.'},

            'squared_hinge': {
                'description': 'Используется в бинарных задачах классификации, и когда не важно знать, '
                               'насколько уверен классификатор в классификации. '
                               'Используется в сочетании с функцией активации tanh() на последнем слое.'},
            'hinge': {'description': ''},
            'categorical_hinge': {'description': ''},

            'binary_crossentropy': {
                'description': 'Используется в задачах бинарной классификации, когда целевая переменная '
                               'представляет собой бинарные метки, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации sigmoid() на последнем слое.'},
            'categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой вероятности, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'sparse_categorical_crossentropy': {
                'description': 'Используется в задачах классификации, '
                               'когда целевая переменная представляет собой индексы классов, '
                               'которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'kullback_leibler_divergence': {
                'description': 'Используется в задачах классификации, когда целевая переменная '
                               'представляет собой вероятности, которые должны быть предсказаны. '
                               'Используется в сочетании с функцией активации softmax() на последнем слое.'},
            'poisson': {'description': 'Используется в задачах классификации, когда целевая переменная '
                                       'представляет собой счетчики, которые должны быть предсказаны. '
                                       'Используется в сочетании с функцией активации exp() на последнем слое.'},
            'logcosh': {'description': ''},
            'cosine_proximity': {'description': ''}
        },
        'default': 'sparse_categorical_crossentropy',
        'title': 'функция потерь'
    },

    'optimizer': {
        'type': 'str',
        'values': {  # TODO: проверить описание оптимизаторов
            'SGD': {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum','global_clipnorm'],
                    'description': 'Метод стохастического градиентного спуска'},
            'Adam': {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon'],
                     'description': 'Оптимизатор использует оценки первого и второго момента градиента'},
            'RMSprop': {'params': ['rho', 'epsilon', 'momentum', 'centered'],
                        'description': 'Адаптивный метод градиентного спуска, '
                                       'основанный на оценках второго момента градиента'},
            'Adagrad': {'params': ['epsilon'],
                        'description': 'Адаптивный метод градиентного спуска, основанный на сумме квадратов градиента'},
            'Adadelta': {'params': ['rho', 'epsilon'],
                         'description': 'Адаптивный метод градиентного спуска, '
                                        'основанный на сумме квадратов градиента и '
                                        'градиента на предыдущем шаге'},
            'Adamax': {'params': ['beta_1', 'beta_2', 'epsilon'],
                       'description': 'Адаптивный метод градиентного спуска, '
                                      'основанный на оценках первого и максимального второго моментов градиента'},
            'Nadam': {'params': ['beta_1', 'beta_2', 'epsilon'],
                      'description': 'Адаптивный метод градиентного спуска, '
                                     'основанный на оценках первого и взвешенного второго моментов градиента'}
        },
        'default': 'Adam',
        'title': "оптимизатор",
        'description': "Оптимизатор, используемый при обучении нейронной сети"
    },
    
    'learning_rate': {
        'type': 'float',
        'range': [1e-5, 1e-1],
        'default': 1e-4,
        'step': 2,
        'scale': 'log',
        'title': "скорость обучения"
    },

    # dataset params
    **db_hparams,
    
    'ROTATION_FACTOR_min' :{
        'type': 'float',
        'default': -0.2,
        'step': 0.1,
        'range': [-2,2],
        'title': 'ROTATION_FACTOR',
        'description': 'Параметры аугментации изображений.'
    },

    'ROTATION_FACTOR_max':{
        'type': 'float',
        'default': 0.2,
        'step': 0.1,
        'range': [-2,2],
        'title': 'ROTATION_FACTOR_max',
        'description': 'Параметры аугментации изображений.'
    },

    'augmen_params_flip': {
        'type': 'str', 
        'default': "horizontal_and_vertical",
        'values': {"horizontal", "vertical", "horizontal_and_vertical"},
        'title': 'параметры аугментации отражения',
        'description': 'Параметры аугментации отражения.'},

    'bounding_box_format': {
        'type': 'str',
        'default': "xywh",
        'values': {"xywh", "xyxy", "rel_xyxy", "center_xywh"},
        'title': 'формат ограничивающих прямоугольников',
        'description': 'Параметры аугментации изображений.'
    },
    
    'resize_min': {
        'type': 'float',
        'default': 0.75,
        'step': 0.05,
        'range': [0,3],
        'title': 'scale_factor_min',
        'description': 'Параметры аугментации изображений.'
    },
    
    'resize_max': {
        'type': 'float',
        'default': 1.3,
        'step': 0.05,
        'range': [0,3],
        'title': 'scale_factor_max',
        'description': 'Параметры аугментации изображений.'
    },
    'pad_to_aspect_ratio': {
        'type': 'bool',
        'default': True,
        'values': {True, False},
        'title': 'pad_to_aspect_ratio',
        'description': 'Параметры аугментации изображений.'
    },  
    'drop_remainder': {
        'type': 'bool',
        'default': True,
        'values': {True, False},
        'title': 'drop_remainder',
        'description': 'drop_remainder.'
    }, 

    # early stopping
    'early_stopping': {'type': 'bool', 'default': False, 'title': 'Ранняя остановка',
                       'description': 'Если в течение определенного количества эпох не происходит улучшения '
                                      'метрики, то обучение прекращается.'},
    'patience': {'type': 'int', 'range': [1, 100], 'step': 1, 'scale': 'lin', 'default': 5,
                 'title': 'Сколько эпох ждать',
                 'description': 'Количество эпох, в течение которых метрика не улучшается, '
                                'после чего обучение прекращается.'},
    'min_delta': {'type': 'float', 'range': [0, 0.1], 'step': 0.001, 'scale': 'lin', 'default': 0.001,
                  'title': 'Минимальное улучшение',
                  'description': 'Минимальное изменение метрики, которое считается улучшением.'},


    # conditional parameters (for optimizers)
    'nesterov': {'type': 'bool', 'default': False, 'title': 'Nesterov momentum', 'cond': True},  # для SGD
    'centered': {'type': 'bool', 'default': False, 'title': 'centered', 'cond': True},  # для RMSprop
    'amsgrad': {'type': 'bool', 'default': False, 'title': 'amsgrad для Adam', 'cond': True},  # для Adam

    'momentum': {'type': 'float', 'range': [0, 1], 'default': 0.9, 'step': 0.01, 'scale': 'lin',
                 'title': 'momentum', 'cond': True},  # момент для SGD
    'rho': {'type': 'float', 'range': [0.5, 0.99], 'default': 0.9, 'title': 'rho', 'cond': True,
            'step': 2**0.25, 'scale': '1-log'},  # коэффициент затухания для RMSprop
    'epsilon': {'type': 'float', 'range': [1e-8, 1e-1], 'default': 1e-7, 'step': 10, 'scale': 'log',
                'title': 'epsilon', 'cond': True},  # для RMSprop, Adagrad, Adadelta, Adamax, Nadam
    'beta_1': {'type': 'float', 'range': [0.5, 0.999], 'default': 0.9, 'title': 'beta_1 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
    'beta_2': {'type': 'float', 'range': [0.5, 0.9999], 'default': 0.999, 'title': 'beta_2 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': '1-log'},  # для Adam, Nadam, Adamax
    'global_clipnorm': {'type': 'float', 'range': [0, 15.0], 'step': 1, 'default': 10.0, 'title': 'global_clipnorm', 'cond': True},  # for allfor all


    #additional parameters
    'transfer_learning': {'type': 'bool', 'default': False,'title': 'дообучить предобученную нейронную сеть'},
    'activation': {'type': 'str', 'values': ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                             'sigmoid', 'hard_sigmoid', 'linear'],
                   'default': 'relu', 'title': 'функция активации'},
    'dropout': {'type': 'float', 'range': [0, 1], 'step': 0.01, 'scale': 'lin', 'default': 0.0, 'title': 'dropout'},
    # доля нейронов, которые отключаются при обучении
    'kernel_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                     'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                     'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                     'he_uniform'],
                           'default': 'glorot_uniform', 'title': 'инициализатор весов'},
    'bias_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                   'he_uniform'], 'default': 'zeros', 'title': 'инициализатор смещений'},
    'kernel_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                           'default': 'auto',
                           'title': 'регуляризатор весов',
                           'description': ''},
    'bias_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                         'default': 'auto',
                         'title': 'регуляризатор смещений',
                         'description': ''},
    'activity_regularizer': {'type': 'str', 'values': ['auto', 'l1', 'l2', 'l1_l2'],
                             'default': 'auto',
                             'title': 'регуляризатор активации',
                             'description': ''},
    'kernel_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                          'default': 'auto',
                          'title': 'ограничение весов',
                          'description': ''},
    'bias_constraint': {'type': 'str', 'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                        'default': 'auto',
                        'title': 'ограничение смещений',
                        'description': ''},
}
    

grid_hparams_space = {  # гиперпараметры, которые будем перебирать по сетке
    # TODO: объединить как-то с hyperparameters
    'model_arch': {'values': _models_order_by_num_weights + _additional_models},
    'transfer_learning': {'values': [True, False], 'default': True},
    'optimizer': {'values': {
        'Adam': {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon']},
        'SGD': {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum']},
        'RMSprop': {'params': ['rho', 'epsilon', 'momentum', 'centered']},
    }},
    # для каждого оптимизатора указывается, как другие гиперпараметры должны масштабироваться при смене оптимизатора
    'batch_size': {'range': [8, 32], 'default': 32, 'step': 2, 'scale': 'log', 'type': 'int'},
    'learning_rate': {'range': [0.000125, 0.064], 'default': 0.001, 'step': 2, 'scale': 'log', 'type': 'float'},
    'lr/batch_size': {'range': [0.00000125, 0.00128], 'default': 0.001, 'step': 2, 'scale': 'log', 'type': 'float'},
    # только один из двух параметров может быть задан: learning_rate или lr/batch_size
    # 'decay': {'type': 'float', 'range': [1/2**5, 1], 'default': 0.0, 'step': 2, 'scale': 'log', 'zero_point': 1},

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





tune_hparams = {
    'method': {'type': 'str',
               'values': {
                   'grid': {'params': ['radius', 'grid_metric', 'start_point', 'tuned_params']},
                   # 'history': {'params': ['exact_category_match']}
               },
               'default': 'grid',
               'title': 'Метод оптимизации гиперпараметров'},
    # conditional parameters:
    'tuned_params': {'type': 'list',
                     'default': ['model_arch'],
                     'values': ['model_arch', 'transfer_learning', 'optimizer', 'batch_size', 'lr/batch_size'],
                     'cond': True, 'title': 'Оптимизируемые гиперпараметры',
                     'description': 'Гиперпараметры, по которым будет производиться оптимизация'},
    'radius': {'type': 'int', 'range': [1, 5], 'default': 1, 'step': 1, 'scale': 'lin', 'title': 'Радиус', 'cond': True},
    'grid_metric': {'type': 'str', 'values': ['l1', 'max'], 'default': 'l1',
                    'title': 'Метрика на сетке', 'cond': True},
    'start_point': {'type': 'str', 'values': ['random', 'auto'], 'default': 'auto',
                    'title': 'Начальная точка', 'cond': True},
    # 'exact_category_match': {'type': 'bool', 'default': False,
    #                         'title': 'Точное совпадение списка категорий', 'cond': True},
}


def get_hparams(params_table, **kwargs) -> dict:
    """
    Функция для дополнения значений гиперпараметров из таблицы по умолчанию новыми значениями.

    Args:
        params_table (dict): таблица гиперпараметров (например, nn_hparams)
        **kwargs: новые значения гиперпараметров
    Returns:
        таблица гиперпараметров с новыми значениями
    """
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


class _TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
        self.start_of_train = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs=None):
        self.total_time = (time.time() - self.start_of_train)


def create_data_subset(objects, cur_experiment_dir, crop_bbox=True, with_segmentation=False, with_bbox=False, temp_dir='tmp', split_points=(0.7, 0.85)):
    """ Создание подвыборки данных для обучения

    Args:
        objects (list): список объектов, которые должны быть включены в подвыборку
        cur_experiment_dir (str): путь к директории, в которой будет производиться обучение
        crop_bbox (bool): если True, то изображения будут обрезаны по bounding box (для классификации)
        with_segmentation (bool): наличие данных для задачи сегментации
        with_bbox (bool): наличие данных для задачи детектирования
        temp_dir (str): путь к временной папке
        split_points (tuple of float): точки разбиения на train, val, test
    Returns:
        словарь путей к csv-файлам с разметкой для train, val, test
    """
    if _emulation:
        crop_bbox = False
    if not os.path.exists(temp_dir) and crop_bbox:
        os.makedirs(temp_dir, exist_ok=True)
    return cur_db().load_specific_categories_annotations(list(objects), normalize_cats=True,
                                                         with_segmentation=with_segmentation,
                                                         with_bbox=with_bbox,
                                                         split_points=split_points,
                                                         cur_experiment_dir=cur_experiment_dir,
                                                         crop_bbox=crop_bbox,
                                                         cropped_dir=temp_dir + '/crops/')[1]


class _EmulateGen:
    def __init__(self, data):
        self.filenames = [f'{i}.jpg' for i in range(len(data))]


def create_generators(model, data, augmen_params, preprocessing_function, batch_size, num_classes):
    """
    Создание генераторов изображений по заданным параметрам аугментации
    Эту функцию следует вызывать перед первым обучением, а также
    при смене параметров аугментации после обучения модели

    Args:
        model (keras.Model): модель, для которой создаются генераторы
        data (dict): словарь путей к csv-файлам с разметкой для train, val, test
        augmen_params (dict): словарь параметров аугментации
        preprocessing_function (str): имя функции предобработки изображений для заданной архитектуры
        batch_size (int): размер батча
        num_classes (int): количество классов
    Returns:
        генераторы для train, val, test
    """
    df_train = pd.read_csv(data['train'], delimiter=';')
    df_validate = pd.read_csv(data['validate'], delimiter=';')
    df_test = pd.read_csv(data['test'], delimiter=';')
    # Определяем размерность входных данных из модели
    flow_args = dict(target_size=model.input_shape[1:3], class_mode='raw', batch_size=batch_size)

    if _emulation:
        return _EmulateGen(df_train), _EmulateGen(df_validate), _EmulateGen(df_test)

    if preprocessing_function is not None:
        preprocessing_function = eval(preprocessing_function)
    data_gen = ImageDataGenerator(**augmen_params, preprocessing_function=preprocessing_function)

    flow_args['class_mode'] = 'binary' if num_classes == 2 else 'categorical'
    flow_args['classes'] = list(map(str, range(num_classes)))
    df_train['target'] = df_train['target'].apply(str)
    df_validate['target'] = df_validate['target'].apply(str)
    df_test['target'] = df_test['target'].apply(str)
    
    train_generator = data_gen.flow_from_dataframe(df_train, x_col=list(df_train.columns)[0],
                                                   y_col=list(df_train.columns)[1], **flow_args)
    val_generator = data_gen.flow_from_dataframe(df_validate, x_col=list(df_validate.columns)[0],
                                                 y_col=list(df_validate.columns)[1], **flow_args)
    test_generator = data_gen.flow_from_dataframe(df_test, x_col=list(df_test.columns)[0],
                                                  y_col=list(df_test.columns)[1], **flow_args)

    return train_generator, val_generator, test_generator


def create_model(base, last_layers, dropout=0.0, input_shape=None, transfer_learning=True):
    """
    Создание модели по заданным параметрам архитектуры

    Args:
        base (str): имя базовой архитектуры
        last_layers (list): список слоев, которые будут добавлены к базовой архитектуре
        dropout (float): коэффициент dropout (0.0 - 1.0)
        input_shape (tuple): размерность входных данных
        transfer_learning (bool): использовать ли transfer learning (True) или обучать с нуля (False)
    Returns:
        созданная модель
    """
    def create_layer(type, **kwargs):
        return getattr(keras.layers, type)(**kwargs)

    if base.lower() in pretrained_models:
        # load pretrained model with weights without last layers
        base_model = pretrained_models[base.lower()](include_top=False,
                                             weights='imagenet' if transfer_learning else None,
                                             input_shape=input_shape or _model_input_shape[base.lower()])
    else:
        base_model = keras.models.load_model(f'{_data_dir}/architectures/{base}.h5')

    # insert dropout layer if needed
    input_shape = base_model.input_shape[1:]
    x = keras.layers.Input(shape=input_shape)
    y = base_model(x)
    if dropout > 0:
        y = keras.layers.Dropout(dropout)(y)
    for layer in last_layers:
        y = create_layer(**layer)(y)

    return keras.models.Model(inputs=x, outputs=y)


class FitLog:
    """
    Класс для логирования процесса обучения. Может использоваться для построения
    графиков в параллельном процессе без необходимости записи логов в файл.
    """
    def __init__(self, task, hparams):
        """
        Args:
            task (NNTask or None): решаемая задача
            hparams (dict): гиперпараметры обучения
        """
        self.task = task
        self.hparams = hparams
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.test_acc = []
        self.train_ends = []
        self.best_val_acc = 0
        self.start_time = time.time()
        self.best_epoch = None

    def add_epoch(self, epoch, loss, accuracy, val_loss=None, val_accuracy=None, **kwargs):
        """
        Добавление информации о текущей эпохе обучения
        (предполагается вызывать из keras.callbacks.Callback.on_epoch_end)

        Args:
            epoch (int): номер эпохи
            loss (float): значение функции потерь на обучающей выборке
            accuracy (float): значение точности на обучающей выборке
            val_loss (float): значение функции потерь на валидационной выборке
            val_accuracy (float): значение точности на валидационной выборке
        """
        self.loss.append(loss)
        self.val_loss.append(val_loss)
        self.acc.append(accuracy)
        self.val_acc.append(val_accuracy)
        if val_accuracy > self.best_val_acc:
            self.best_val_acc = val_accuracy
            self.best_epoch = (len(self.train_ends), epoch)

    def add_train_end(self, **kwargs):
        """
        Метод вызывается в конце обучения (предполагается вызывать из keras.callbacks.Callback.on_train_end)

        Args:
            **kwargs: параметры из keras.callbacks.Callback.on_train_end
        """
        self.train_ends.append((len(self.loss), time.time() - self.start_time))

    def add_test(self, accuracy):
        """
        Добавление информации о точности на тестовой выборке

        Args:
            accuracy (float): значение точности на тестовой выборке
                (метод обязательно должен быть вызван после обучения, иначе история обучения будет некорректной)
        """
        self.test_acc.append(accuracy)

    @property
    def best_acc(self):
        """ лучшая точность на тестовой выборке """
        return max(self.test_acc) if self.test_acc else 0


class ExperimentLog:
    """
    Класс для логирования результатов экспериментов. Может использоваться для построения
    графиков в параллельном процессе без необходимости записи логов в файл.

    Содержит в себе список историй обучения (FitLog), произведенных в рамках эксперимента.
    """
    def __init__(self, task=None):
        self.task = task
        self.fit_runs = []
        self.best_run = None
        self.current_run = None

    def new_experiment(self, hparams):
        """
        Создаёт новую историю обучения (FitLog) для эксперимента с заданными гиперпараметрами.

        Args:
            hparams (dict): гиперпараметры обучения
        """
        self.current_run = FitLog(self.task, hparams)
        if self.best_run is None:
            self.best_run = self.current_run
        self.fit_runs.append(self.current_run)
        return self.current_run

    def update_best(self):
        """:meta private:"""
        if self.current_run is not None and self.current_run.best_acc > self.best_run.best_acc:
            self.best_run = self.current_run

    @property
    def best_acc(self):
        """ Лучшая точность на тестовой выборке """
        self.update_best()
        return self.best_run.best_acc

    @property
    def best_val_acc(self):
        """ Лучшая точность на валидационной выборке (может быть выше, чем best_acc) """
        self.update_best()
        return self.best_run.best_val_acc


class ExperimentHistory:
    """
    Класс для сохранения истории экспериментов в csv файл.
    """
    def __init__(self, task, exp_name, exp_path, data):
        """
        Args:
            task (NNTask): задача обучения нейронной сети
            exp_name (str): название эксперимента
            exp_path (str): путь к папке с экспериментами
            data (dict): словарь с путями к обучающей, валидационной и тестовой выборкам
        """
        self.experiment_number = 0
        self.exp_name = exp_name
        self.exp_path = exp_path
        self.data = data
        self.task_type = task.type
        self.objects = task.objects

        self.history = pd.DataFrame(columns=['Index', 'task_type', 'objects', 'exp_name', 'model_arch', 'last_layers',
                                             'augmen_params', 'loss', 'metrics', 'epochs', 'stop_criterion', 'data',
                                             'optimizer', 'batch_size', 'learning_rate', 'metric_test_value',
                                             'train_subdir', 'time_stat', 'total_time', 'additional_params'])

        self.save()

    def add_row(self, params, metric, train_subdir, time_stat, total_time, save=True):
        """
        Добавление строки об очередном запуске обучения в историю экспериментов.

        Args:
            params (dict): словарь с гиперпараметрами обучения
            metric (float): значение метрики на тестовой выборке
            train_subdir (str): путь к папке с логами обучения и обученной моделью
            time_stat (list): список временных меток для эпох обучения
            total_time (float): общее время обучения
            save (bool): сохранить ли сразу обновлённую историю экспериментов в файл
        """

        if self.task_type == 'classification':
            arch = params['model_arch']
            last_layers =  params['last_layers']
            augmen_params = params['augmen_params']
            loss = params['loss']

        elif self.task_type == 'detection':
            arch = params['model_arch']
            last_layers =  [[]]
            par = {'bounding_box_format': params['bounding_box_format'],
                   'augmen_params_flip' : params['augmen_params_flip'],
                   'resize_min': params['resize_min'],
                   'resize_max': params['resize_max'],
                   'pad_to_aspect_ratio': params['pad_to_aspect_ratio'],
                   'drop_remainder':params['drop_remainder']}
            augmen_params = [par]
            loss = [params['classification_loss'],params['box_loss']]

        elif self.task_type == 'segmentation':
            arch = params['model_arch']
            last_layers =  [[]]
            par = {'bounding_box_format': params['bounding_box_format'],
                   'ROTATION_FACTOR_min' : params['ROTATION_FACTOR_min'],
                   'ROTATION_FACTOR_max' : params['ROTATION_FACTOR_max'],
                   'augmen_params_flip' : params['augmen_params_flip'],
                   'resize_min': params['resize_min'],
                   'resize_max': params['resize_max'],
                   'pad_to_aspect_ratio': params['pad_to_aspect_ratio'],
                   'drop_remainder':params['drop_remainder']}
            augmen_params = [par]
            loss = params['loss']

        
        self.experiment_number += 1
        row = ({'Index': self.experiment_number,  # номер эксперимента
                'task_type': self.task_type,  # тип задачи
                'objects': [self.objects],  # список объектов, на распознавание которых обучается модель
                'exp_name': self.exp_name,  # название эксперимента

                'model_arch': arch,  # базовая часть модели
                'last_layers': last_layers,  # последние слои модели
                'augmen_params': augmen_params,  # параметры аугментации
                'loss': loss,  # функция потерь
                'metrics': params['metrics'],  # метрика, по которой оценивается качество модели
                'epochs': params['epochs'],  # количество эпох обучения
                'stop_criterion': params.get('stop_criterion',''),  # критерий остановки обучения (TODO: не используется, исправить!!!)
                'data': self.data,  # набор данных, на котором проводится обучение

                'optimizer': params['optimizer'],  # оптимизатор
                'batch_size': params['batch_size'],  # размер батча
                'learning_rate': params['learning_rate'],  # скорость обучения

                'metric_test_value': metric,  # значение метрики на тестовой выборке
                'train_subdir': train_subdir,  # папка, в которой хранятся результаты текущего обучения
                'time_stat': time_stat,  # список длительностей всех эпох обучения
                'total_time': total_time,  # общее время обучения
                'additional_params': [{}]})

        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)


        if save:
            self.save()

    def save(self):
        """ Сохранение истории экспериментов в файл. """
        self.history.to_csv(self.exp_path + '/history.csv', index=False)

    def get_best_model(self):
        """ Получение информации о лучшей модели. """
        best_model = self.history.loc[self.history['metric_test_value'].idxmax()]
        return best_model

    def get_best_model_path(self):
        """ Получение пути к лучшей модели. """
        best_model = self.get_best_model()
        return best_model['train_subdir'] + '/best_model.h5'

    def get_best_model_params(self):
        """ Получение параметров лучшей модели. """
        best_model = self.get_best_model()
        return {'optimizer': best_model['optimizer'],
                'batch_size': best_model['batch_size'],
                'learning_rate': best_model['learning_rate']}


class StopFlag:
    """ Флаг, который позволяет остановить обучение модели из другого потока """
    def __init__(self):
        self.flag = False

    def __call__(self):
        """ Устанавливает флаг в True (команда "остановить обучение") """
        self.flag = True


class _CheckStopCallback(keras.callbacks.Callback):
    def __init__(self, stop_flag, timeout=None):
        super().__init__()
        self.t0 = time.time()
        self.timeout = timeout if timeout else 1e20
        self.stop_flag = stop_flag or StopFlag()

    def on_batch_end(self, batch, logs=None):
        if self.stop_flag.flag:
            self.model.stop_training = True
        elif time.time() - self.t0 > self.timeout:
            self.model.stop_training = True
            printlog(f'\nTraining stopped by timeout ({self.timeout} sec)')


class _NotifyCallback(keras.callbacks.Callback):
    def __init__(self, fit_log):
        super().__init__()
        self.fit_log = fit_log

    def on_batch_end(self, batch, logs=None):
        pcall('train_callback', 'batch', batch=batch, logs=logs, model=self.model)

    def on_epoch_end(self, epoch, logs=None):
        if self.fit_log:
            self.fit_log.add_epoch(epoch, **logs)
        pcall('train_callback', 'epoch', epoch=epoch, logs=logs, model=self.model)

    def on_train_end(self, logs=None):
        if self.fit_log:
            self.fit_log.add_train_end(**logs)
        pcall('train_callback', 'finish', logs=logs, model=self.model)

    def on_train_begin(self, logs=None):
        pcall('train_callback', 'start', logs=logs, model=self.model)


def _emulate_fit(model, x, steps_per_epoch, epochs, callbacks, validation_data):
    loss_begin = 0.2 + random.random()*0.3
    loss_end = 0.1 + random.random()*0.1
    loss = loss_begin
    acc_begin = 0.5 + random.random()*0.3
    acc_end = 0.8 + random.random()*0.15
    acc = acc_begin
    best_acc = 0
    best_loss = 1
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()
    for epoch in range(epochs):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        for batch in range(steps_per_epoch):
            if model.stop_training:
                break
            noise = (random.random()-0.5)*0.1
            loss = max(0, max(loss_begin + (loss_end - loss_begin) * (batch+epoch*steps_per_epoch) / (steps_per_epoch*epochs*0.5), loss_end) + noise)
            noise = (random.random()-0.5)*0.1
            acc = min(1, min(acc_begin + (acc_end - acc_begin) * (batch+epoch*steps_per_epoch) / (steps_per_epoch*epochs*0.5), acc_end) + noise)
            best_acc = max(best_acc, acc)
            best_loss = min(best_loss, loss)
            for callback in callbacks:
                callback.on_batch_end(batch, logs={'loss': loss, 'accuracy': acc})
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={'loss': loss, 'accuracy': acc})
        if model.stop_training:
            break
    for callback in callbacks:
        callback.on_train_end(logs={'loss': best_loss, 'accuracy': best_acc})
    return [best_loss, best_acc]


def _save_history(filepath, objects, run_type, model_path, metrics, params, fmt=None):
    history = {'run_type': run_type,
               **params,
               'model_file': model_path,
               'result_path': os.path.dirname(model_path),
               'metric_name': 'accuracy',
               'metric_value': metrics['accuracy'],
               'metrics': metrics,
               'objects': objects}
    if fmt is None:
        fmt = filepath.split('.')[-1]
    if fmt == 'json':
        with open(filepath, 'w') as f:
            json.dump(history, f)
    elif fmt in ('pkl', 'pickle'):
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)
    elif fmt == 'yaml':
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(history, f)
    else:
        raise ValueError(f'Unknown format: {fmt}')
    cur_db().add_model_record(task_type='train',
                              categories=list(objects),
                              model_address=model_path,
                              metrics=metrics,
                              history_address=filepath)
    pcall('append_history', history)
    return history


def compile_model(model, hparams, measured_metrics, freeze_base=None):
    """
    Компилирует модель с заданными параметрами

    Args:
        model: модель
        hparams (dict): гиперпараметры оптимизации (оптимизатор, скорость обучения, ...)
        measured_metrics (list): список метрик, которые будут измеряться в процессе обучения
        freeze_base (bool): замораживать ли базовую часть модели (первый этап transfer learning)
    """
    optimizer, lr = hparams['optimizer'], hparams['learning_rate']
    opt_args = ['decay'] + nn_hparams['optimizer']['values'][optimizer].get('params', [])
    kwargs = {arg: hparams[arg] for arg in opt_args if arg in hparams}
    optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr, **kwargs)
    if freeze_base is True:
        printlog("Freeze base model layers")
        model.layers[1].trainable = False
    elif freeze_base is False:
        printlog("Unfreeze base model layers (fine-tuning)")
        model.layers[1].trainable = True
    printlog("Compile model")
    model.compile(optimizer=optimizer, loss=hparams['loss'], metrics=measured_metrics)


def prepare_callbacks(stop_flag, timeout, cur_subdir, check_metric, use_tensorboard, weights_name, fit_log,
                      early_stopping=True, patience=5, min_delta=0.001):
    """
    Подготавливает список колбэков для обучения модели

    Args:
        stop_flag (StopFlag): флаг остановки обучения
        timeout (float): таймаут обучения в секундах
        cur_subdir (str): путь к папке текущего запуска
        check_metric (str): метрика, по которой будет производиться остановка обучения и сохранение лучшей модели
        use_tensorboard (bool): сбрасывать ли логи для TensorBoard
        weights_name (str): имя файла, в который будут сохраняться веса модели
        fit_log (FitLog): лог обучения модели
        early_stopping (bool): использовать ли раннюю остановку
        patience (int): количество эпох, после которых обучение будет остановлено, если метрика не улучшилась
        min_delta (float): минимальное изменение метрики, которое считается улучшением
    Returns:
        пара (список колбэков, объект, в котором будут сохранены замеры времени по каждой эпохе)
    """
    if not _emulation:

        c_log = keras.callbacks.CSVLogger(cur_subdir + '/Log.csv', separator=',', append=True)
        c_ch = keras.callbacks.ModelCheckpoint(cur_subdir + f'/{weights_name}.h5', monitor=check_metric, verbose=1,
                                               save_best_only=True, save_weights_only=False, mode='auto')
        callbacks = [c_log, c_ch]
        if early_stopping:
            c_es = keras.callbacks.EarlyStopping(monitor=check_metric, min_delta=min_delta, patience=patience,
                                                 mode='auto')
            callbacks.append(c_es)
        # clear tensorboard logs
        if os.path.exists(tensorboard_logdir()):
            shutil.rmtree(tensorboard_logdir(), ignore_errors=True)
        os.makedirs(tensorboard_logdir(), exist_ok=True)

        if use_tensorboard:
            c_tb = keras.callbacks.TensorBoard(
                log_dir=tensorboard_logdir(),  # , datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            )
            callbacks.append(c_tb)
    else:
        callbacks = []

    c_t = _TimeHistory()
    callbacks += [c_t, _NotifyCallback(fit_log)]
    if stop_flag is not None or timeout is not None:
        callbacks.append(_CheckStopCallback(stop_flag, timeout))
    return callbacks, c_t


def _fit(model: keras.Model, generators, hparams, stop_flag, timeout, cur_subdir, check_metric, use_tensorboard,
         weights_name, fit_log, initial_epoch=0):
    callbacks, c_t = prepare_callbacks(stop_flag, timeout, cur_subdir, check_metric, use_tensorboard,
                                       weights_name=weights_name, fit_log=fit_log,
                                       early_stopping=hparams.get('early_stopping', True),
                                       patience=hparams.get('patience', 5),
                                       min_delta=hparams.get('min_delta', 0.001))
    if _emulation:
        scores = _emulate_fit(model, generators[0], max(1, len(generators[0].filenames) // hparams['batch_size']),
                              hparams['epochs'], callbacks, generators[1])
        model.save(cur_subdir + f'/{weights_name}.h5')
    else:
        printlog("Fit model")
        printlog(f"Train samples: {len(generators[0].filenames)}, batch size: {hparams['batch_size']}")
        # fit model
        model.fit(x=generators[0],
                  steps_per_epoch=max(1, len(generators[0].filenames) // hparams['batch_size']),
                  epochs=hparams['epochs'],
                  validation_data=generators[1],
                  callbacks=callbacks,
                  validation_steps=max(1, len(generators[1].filenames) // hparams['batch_size']),
                  initial_epoch=initial_epoch)

        # load best weights
        printlog("Load best weights")
        model.load_weights(cur_subdir + f'/{weights_name}.h5')

        # evaluate model
        scores = model.evaluate(generators[2], steps=None, verbose=1)
    if fit_log:
        fit_log.add_test(scores[1])
    return scores, c_t

    
def fit_model(nn_task, model, objects, hparams, generators, cur_subdir, history=None, stop_flag=None, need_recompile=False,
              use_tensorboard=False, timeout=None, exp_log=None):
    """ Обучение модели

    Args:
        model (keras.models.Model): модель, которую нужно обучить
        objects (list): список категорий, которые нужно распознавать
        hparams (dict): словарь с гиперпараметрами обучения
        generators (tuple): кортеж из трех генераторов: train, val, test
        cur_subdir (str): папка, в которой хранятся результаты текущего обучения
        history (ExperimentHistory or None): история экспериментов
        stop_flag (StopFlag or None): флаг, с помощью которого можно остановить обучение из другого потока
        need_recompile (bool): обязательно ли перекомпилировать модель
        use_tensorboard (bool): нужно ли использовать tensorboard
        timeout (float): таймаут обучения в секундах
        exp_log (ExperimentLog): лог эксперимента
    Returns:
        (list[float], dict): Достигнутые значения метрик на тестовой выборке во время обучения, а также
            словарь со значениями гиперпараметров, метрик и путей к модели и истории
    """
    if timeout is None:
            timeout = 1e10
    t0 = time.time()

    date = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    if exp_log is None:
        fit_log = None
    else:
        fit_log = exp_log.new_experiment(hparams)

    measured_metrics = hparams['metrics']
    if not isinstance(measured_metrics, list):
        measured_metrics = [measured_metrics]

    check_metric = 'val_' + measured_metrics[0]

    if nn_task.type == 'classification':
    
        transfer_learning = hparams.get('transfer_learning', False)     
    
        if not transfer_learning:
            # if model is not compiled, compile it
            if not model.optimizer or transfer_learning or need_recompile:
                compile_model(model, hparams, measured_metrics, freeze_base=transfer_learning)
    
            # fit model
            scores, c_t = _fit(model, generators, hparams, stop_flag, timeout, cur_subdir, measured_metrics[0], use_tensorboard,
                               weights_name='best_weights', fit_log=fit_log)
        else:
            compile_model(model, hparams, measured_metrics, freeze_base=True)
            coef = hparams.get('fine_tune_lr_div', 10)
            first_fit_params = {**hparams, 'epochs': max(1, hparams['epochs'] // coef)}
            if hparams.get('early_stopping', True):
                first_fit_params['min_delta'] = hparams['min_delta']
                first_fit_params['patience'] = max(1, hparams['patience'] // coef)
            scores, c_t = _fit(model, generators, first_fit_params, stop_flag, timeout / 2,
                               cur_subdir, check_metric, use_tensorboard, weights_name='best_weights', fit_log=fit_log)
    
            new_hparams = hparams.copy()
            new_hparams['learning_rate'] = hparams['learning_rate'] / coef
            compile_model(model, new_hparams, measured_metrics, freeze_base=False)
            tune_scores, tune_c_t = _fit(model, generators, new_hparams, stop_flag, timeout - (time.time() - t0),
                                         cur_subdir, check_metric, use_tensorboard,
                                         weights_name='tune_best_weights', fit_log=fit_log, initial_epoch=len(c_t.times))
    
            if tune_scores[1] > scores[1]:
                scores = tune_scores
                shutil.copyfile(cur_subdir + '/tune_best_weights.h5', cur_subdir + '/best_weights.h5')
            try:
                os.remove(cur_subdir + '/tune_best_weights.h5')
            except FileNotFoundError:
                pass
            c_t.total_time += tune_c_t.total_time
            c_t.times += tune_c_t.times
        metric_value = scores[1]
        
    elif nn_task.type == 'detection':
        scores, c_t = fit_detection(nn_task, model, generators, hparams, stop_flag, timeout, cur_subdir, check_metric,  use_tensorboard,
                               weights_name='best_weights', fit_log=fit_log) 
        metric_value = scores[0]

    elif nn_task.type == 'segmentation':

        scores, c_t = fit_segmentation(nn_task, model, generators, hparams, stop_flag, timeout, cur_subdir, measured_metrics[0],  use_tensorboard,
                               weights_name='best_weights', fit_log=fit_log) 
        metric_value = scores[1]

    
    # save results to history
    if history is not None:
        history.add_row(hparams, metric_value, cur_subdir, c_t.times, c_t.total_time, save=True)

    printlog("Save history")
    if nn_task.type == 'classification':
        metrics = {'accuracy': scores[1]}
        for i, metric in enumerate(measured_metrics):
            metrics[metric] = scores[i + 1]
    #TODO  - concat with DB
    #record = _save_history(cur_subdir + '/history.json', objects, 'train', cur_subdir + '/best_weights.h5',
    #                       metrics, dict(hparams=hparams, date=date, times=c_t.times, total_time=c_t.total_time))
    return scores, None #, __#record

def fit_segmentation(nn_task, model, generators, hparams, stop_flag, timeout, cur_subdir, check_metric,  use_tensorboard,
                               weights_name, fit_log, initial_epoch=0):

    #TODO: concat with DB
    class_ids = nn_task.objects
    BATCH_SIZE = hparams.get('batch_size', 10)

    train_ds = generators['train']
    eval_ds = generators['eval']
    test_ds = generators['test']

    printlog("Create model")
    #model = sm.Unet('resnet18', encoder_weights='imagenet', classes=len(class_ids)) 
    if hparams['model_arch'] == 'Unet':
        model = sm.Unet(hparams['backbone'], encoder_weights='imagenet', classes=len(class_ids))
    if hparams['model_arch'] == 'Linknet':
        model = sm.Linknet(hparams['backbone'], encoder_weights='imagenet', classes=len(class_ids))
    if hparams['model_arch'] == 'FPN':
        model = sm.FPN(hparams['backbone'], encoder_weights='imagenet', classes=len(class_ids))
    if hparams['model_arch'] == 'UPSPNetnet':
        model = sm.PSPNet(hparams['backbone'], encoder_weights='imagenet', classes=len(class_ids))
    if hparams['model_arch'] == 'local_model':
        input_shape=hparams.get('input_shape', 320)
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        model = get_local_model(img_size=input_shape, num_classes=len(class_ids))
    model.save(cur_subdir + '/initial_model.h5')
    try:
        tf.keras.utils.plot_model(model, to_file=cur_subdir + '/model_plot.png', rankdir='TB', show_shapes=True)
    except ImportError as e:
        warnings.warn(f"Can't plot model: {e}")

    #Model compile
    printlog("Compile model")
    optimizer, lr = hparams['optimizer'], hparams['learning_rate']
    opt_args = ['decay'] + nnd_hparams['optimizer']['values'][optimizer].get('params', [])
    kwargs = {arg: hparams[arg] for arg in opt_args if arg in hparams}
    optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr, **kwargs) #TODO: ещё нужно посмотреть в segmentation_models.losses.

    model.compile(
        optimizer=optimizer,
        loss=hparams.get('loss', "sparse_categorical_crossentropy"),
        metrics=[check_metric],
    )

    #callbacks
    early_stopping = hparams.get('early_stopping', True)
    patience = hparams.get('patience', 5)
    min_delta = hparams.get('min_delta', 0.001)
    
    c_log = keras.callbacks.CSVLogger(cur_subdir + '/Log.csv', separator=',', append=True)
    c_ch = keras.callbacks.ModelCheckpoint(cur_subdir + '/' + weights_name  +'-{epoch:02d}.h5', monitor='val_' + check_metric, verbose=1,
                                               save_best_only=True, save_weights_only=True, mode='auto')
    callbacks = [c_log, c_ch]

    if early_stopping:
        c_es = keras.callbacks.EarlyStopping(monitor=check_metric, min_delta=min_delta, patience=patience, mode='auto')
      
        callbacks.append(c_es)
        
    # clear tensorboard logs
    if os.path.exists(tensorboard_logdir()):
        shutil.rmtree(tensorboard_logdir(), ignore_errors=True)
    os.makedirs(tensorboard_logdir(), exist_ok=True)

    if use_tensorboard:
        c_tb = keras.callbacks.TensorBoard(
            log_dir=tensorboard_logdir(),  # , datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
        callbacks.append(c_tb)

    c_t = _TimeHistory()
    callbacks += [c_t, _NotifyCallback(fit_log)]
    if stop_flag is not None or timeout is not None:
        callbacks.append(_CheckStopCallback(stop_flag, timeout))


    # Train the model
    printlog("Fit model")
    model.fit(
        train_ds,
        epochs=hparams.get('epochs', 20),
        validation_data=eval_ds,
        callbacks = callbacks,
        initial_epoch=initial_epoch
    )

    # load best weights
    printlog("Load best weights")
    search_dir = cur_subdir + '/'
    files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
    files.sort(key=lambda x: os.path.getmtime(x))
    files.reverse()
    weights_flag = False
    for elem in files:
        if "best_weights" in elem and ".h5" in elem:
            weights_flag = True
            break
    if not weights_flag:
        print('there is no weights file in cur_dir')
    model.load_weights(elem)


    printlog("Evaluate model")

    scores = model.evaluate(test_ds.take(20), verbose=1)

    print(scores)

    if fit_log:
        fit_log.add_test(scores[1])
    return scores, c_t

def get_local_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(
        x
    )

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def fit_detection(nn_task, model, generators, hparams, stop_flag, timeout, cur_subdir, check_metric,  use_tensorboard,
                               weights_name, fit_log, initial_epoch=0):
    

    class_ids = nn_task.objects
    class_mapping = dict(zip(range(len(class_ids)), class_ids))
    BATCH_SIZE = hparams.get('batch_size', 15)

    train_ds = generators['train']
    eval_ds = generators['eval']
    test_ds = generators['test']

    printlog("Create model")
    if hparams['model_arch'] == 'Yolov8':
        model = keras_cv.models.YOLOV8Detector(
            num_classes=len(class_mapping),
            bounding_box_format=hparams.get('bounding_box_format', "xywh"),
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(hparams.get('backbone', "yolo_v8_s_backbone_coco")),
            fpn_depth=1,
        )
    elif  hparams['model_arch'] == 'RetinaNet':
        model = keras_cv.models.RetinaNet.from_preset(
            hparams.get('backbone', "resnet50_imagenet"),
            num_classes=len(class_mapping),
            bounding_box_format=hparams.get('bounding_box_format', "xywh"),
        )
    
    for i in range(len(model.weights)):
        model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)
    model.save(cur_subdir + '/initial_model.h5')
    try:
        tf.keras.utils.plot_model(model, to_file=cur_subdir + '/model_plot.png', rankdir='TB', show_shapes=True)
    except ImportError as e:
        warnings.warn(f"Can't plot model: {e}")

    #Model compile
    printlog("Compile model")
    optimizer, lr = hparams['optimizer'], hparams['learning_rate']
    opt_args = ['decay'] + nnd_hparams['optimizer']['values'][optimizer].get('params', [])
    kwargs = {arg: hparams[arg] for arg in opt_args if arg in hparams}
    optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr, **kwargs)

    model.compile(
        classification_loss= hparams.get('classification_loss', "focal"),
        box_loss= hparams.get('box_loss', "smoothl1"),
        optimizer=optimizer,
        metrics=None,
    )

    #callbacks
    early_stopping = hparams.get('early_stopping', True)
    patience = hparams.get('patience', 5)
    min_delta = hparams.get('min_delta', 0.001)

    c_log = keras.callbacks.CSVLogger(cur_subdir + '/Log.csv', separator=',', append=True)
    c_ch = keras.callbacks.ModelCheckpoint(cur_subdir + '/' + weights_name  +'-{epoch:02d}.h5', monitor=check_metric, verbose=1,
                                               save_best_only=True, save_weights_only=True, mode='auto')
    callbacks = [c_log, c_ch]
    
    if early_stopping:
        c_es = keras.callbacks.EarlyStopping(monitor=check_metric, min_delta=min_delta, patience=patience, mode='auto')
      
        callbacks.append(c_es)
        
    # clear tensorboard logs
    if os.path.exists(tensorboard_logdir()):
        shutil.rmtree(tensorboard_logdir(), ignore_errors=True)
    os.makedirs(tensorboard_logdir(), exist_ok=True)

    if use_tensorboard:
        c_tb = keras.callbacks.TensorBoard(
            log_dir=tensorboard_logdir(),  # , datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
        callbacks.append(c_tb)

    c_t = _TimeHistory()
    callbacks += [c_t, _NotifyCallback(fit_log)]
    if stop_flag is not None or timeout is not None:
        callbacks.append(_CheckStopCallback(stop_flag, timeout))
    
    printlog("Fit model")
    model.fit(
        train_ds,#.take(20),
        validation_data=eval_ds,#.take(20),
        epochs= hparams.get('epochs', 20),
        callbacks = callbacks,
        initial_epoch=initial_epoch
    )

    # load best weights
    printlog("Load best weights")
    search_dir = cur_subdir + '/'
    files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
    files.sort(key=lambda x: os.path.getmtime(x))
    files.reverse()
    weights_flag = False
    for elem in files:
        if "best_weights" in elem and ".h5" in elem:
            weights_flag = True
            break
    if not weights_flag:
        print('there is no weights file in cur_dir')
    model.load_weights(elem)


    printlog("Evaluate model")

    scores = model.evaluate(test_ds, verbose=1)

    if fit_log:
        fit_log.add_test(scores[0])
    return scores, c_t




def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_mask(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    return image


def get_detection_dataset(df, batch_size):
    def load_dataset(image_path, classes, bbox):
        # Read Image
        image = load_image(image_path)
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bbox,
        }
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}
    bboxes=[]
    classes=[]
    image_paths=[]
    for fn,group in df.groupby('images'):
        image_paths.append(fn)
        bboxes.append(group['bbox'].to_list())
        classes.append(group['target'].to_list())
    bboxes = tf.ragged.constant(bboxes)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)
    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bboxes))
    ds = data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.ragged_batch(batch_size)

def get_segmentation_dataset(df, batch_size=16, imsz=(320,320), num_classes=2,hflip=0):
    def load_dataset(image_path, mask_path, classes):
        # Read Image
        image = load_image(image_path)
        image = tf.image.resize(image, imsz)
        mask= load_mask(mask_path)
        mask = tf.image.resize(mask, imsz)
        if random.random()<hflip:
            image=tf.image.flip_left_right(image)            
            mask=tf.image.flip_left_right(image)            
        mask = tf.concat([tf.math.equal(mask, label_id) for label_id in range(1,num_classes+1)], axis=-1) # [2,2,2]
        return {"images": tf.cast(image, tf.float32), "segmentation_masks": tf.convert_to_tensor(mask)}
    all_classes=[]
    image_paths=[]              
    mask_paths=[]              
    mask_dir='masks/'
    os.makedirs(mask_dir,exist_ok=True)
   
    for fn,group in df.groupby('images'):
        w, h = imagesize.get(fn)
        contours=group['segmentation'].to_list()
        classes=group['target'].to_list()
        mask=np.zeros((w,h,1),np.uint8)
        for i,contour in enumerate(contours):
            c=[np.array(cc).reshape(-1,2).astype(np.uint) for cc in contour]
            cv2.drawContours(mask,c,-1,classes[i]+1,-1)  #zero is background 
        mask_path=os.path.join(mask_dir,os.path.basename(fn).split('.')[0]+'.png')
        cv2.imwrite(mask_path,mask)
        image_paths.append(fn)
        mask_paths.append(mask_path)
        all_classes.append(classes)
    all_classes = tf.ragged.constant(all_classes)
    image_paths = tf.ragged.constant(image_paths)
    mask_paths = tf.ragged.constant(mask_paths)
    data = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths,all_classes))
    ds = data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.ragged_batch(batch_size)

def create_and_train_model(nn_task, hparams, objects, data, cur_subdir, history=None, stop_flag=None,
                           model=None, use_tensorboard=True, timeout=None, exp_log=None):
    """
    Создает модель, компилирует ее и обучает

    Args:
        hparams (dict): словарь с гиперпараметрами обучения
        objects (list): список категорий объектов в датасете
        data (dict): словарь путей к csv-файлам с разметкой для train, val, test
        cur_subdir (str):  папка, в которой хранятся результаты текущего обучения
        history (ExperimentHistory):  история экспериментов
        stop_flag (StopFlag or None): флаг, с помощью которого можно остановить обучение из другого потока
        model (None or keras.models.Model or str): модель, которую нужно обучить.
            Если None, то создается новая модель. Если str, то загружается модель из файла.
        use_tensorboard (bool): сбраывать ли данные для tensorboard
        timeout (float or None): таймаут в секундах. Если не None, то обучение прерывается по истечении этого времени.
        exp_log (ExperimentLog or None): лог экспериментов
    Returns:
        Список чисел -- достигнутые значения метрик на тестовой выборке во время обучения
    """
    print('create_and_train_model')
    if model is None:
        if nn_task.type == 'classification':
            printlog("Create model")
            model = create_model(hparams['model_arch'], hparams['last_layers'], hparams.get('dropout', 0.0),
                                 input_shape=hparams.get('input_shape', None),
                                 transfer_learning=hparams.get('transfer_learning', True))
            model.save(cur_subdir + '/initial_model.h5')
            try:
                tf.keras.utils.plot_model(model, to_file=cur_subdir + '/model_plot.png', rankdir='TB', show_shapes=True)
            except ImportError as e:
                warnings.warn(f"Can't plot model: {e}")
    elif isinstance(model, str):  # model is path to weights   #TODO check
        printlog("Load model")
        model = keras.models.load_model(model)
    elif not isinstance(model, keras.models.Model):
        
        raise TypeError('model must be either path to weights or keras.models.Model or None')

    printlog("Create generators")
    if nn_task.type == 'classification':
        generators = create_generators(model, data, hparams['augmen_params'],
                                       hparams.get('preprocessing_function', None),
                                       hparams['batch_size'], len(objects))
    elif nn_task.type == 'detection':
        class_ids = nn_task.objects
        class_mapping = dict(zip(range(len(class_ids)), class_ids))
        BATCH_SIZE = hparams.get('batch_size', 16)
        df_train = pd.read_csv(data['train'],delimiter=';',converters={'bbox': literal_eval})
        train_ds=get_detection_dataset(df_train,BATCH_SIZE)
        df_val = pd.read_csv(data['validate'],delimiter=';',converters={'bbox': literal_eval})
        val_ds=get_detection_dataset(df_val,BATCH_SIZE)
        df_test = pd.read_csv(data['test'],delimiter=';',converters={'bbox': literal_eval})
        test_ds=get_detection_dataset(df_test,BATCH_SIZE)
        input_shape=hparams.get('input_shape', 640)
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        augmenter = keras.Sequential(
            layers=[
                keras_cv.layers.RandomFlip(mode=hparams.get('augmen_params_flip', "horizontal"), bounding_box_format=hparams.get('bounding_box_format', "xywh")),
                keras_cv.layers.JitteredResize(
                    target_size=input_shape, scale_factor=(hparams.get('resize_min', 0.75), hparams.get('resize_max', 1.3)),
                    bounding_box_format=hparams.get('bounding_box_format', "xywh")
                ),
            ]
        )
        train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        inference_resizing = keras_cv.layers.Resizing(input_shape[0], input_shape[1], bounding_box_format=hparams.get('bounding_box_format', "xywh"),
                                                      pad_to_aspect_ratio=hparams.get('pad_to_aspect_ratio', True))
        val_ds = val_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
    
    
        train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        generators = {'train' : train_ds, 'eval' : val_ds, 'test' : test_ds}

    elif nn_task.type == 'segmentation':
        class_ids = nn_task.objects
        num_classes=len(class_ids)
        BATCH_SIZE = hparams.get('batch_size', 16)
        input_shape=hparams.get('input_shape', 320)
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        df_train = pd.read_csv(data['train'],delimiter=';',converters={'segmentation': literal_eval})
        train_ds=get_segmentation_dataset(df_train,BATCH_SIZE, input_shape, num_classes, 0.5)
        df_val = pd.read_csv(data['validate'],delimiter=';',converters={'segmentation': literal_eval})
        val_ds=get_segmentation_dataset(df_val,BATCH_SIZE, input_shape, num_classes)
        df_test = pd.read_csv(data['test'],delimiter=';',converters={'segmentation': literal_eval})
        test_ds=get_segmentation_dataset(df_test,BATCH_SIZE, input_shape, num_classes)
  
        train_ds = train_ds.map(unpackage_inputs).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(unpackage_inputs).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(unpackage_inputs).prefetch(tf.data.AUTOTUNE)

        generators = {'train' : train_ds, 'eval' : val_ds, 'test' : test_ds}

    return fit_model(nn_task, model, objects, hparams, generators, cur_subdir, history=history, stop_flag=stop_flag,
                     use_tensorboard=use_tensorboard, timeout=timeout, exp_log=exp_log)

def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks

def load_pascal_voc(split, dataset, bounding_box_format):
    ds = tfds.load(dataset, split=split, with_info=False, shuffle_files=True)
    ds = ds.map(
        lambda x: unpackage_raw_tfds_inputs(x, bounding_box_format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


def unpackage_raw_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

def train(nn_task, hparams, stop_flag=None, model=None, use_tensorboard=True, timeout=None, exp_log=None) -> Tuple[List[float], dict]:
    """
    Обучает модель для заданной задачи классификации изображений

    Args:
        nn_task (NNTask): задача обучения нейросети
        hparams (dict or str): словарь с гиперпараметрами обучения или "auto" для выбора по умолчанию
        stop_flag (StopFlag): флаг, с помощью которого можно остановить обучение из другого потока
        model (None or keras.Model or str): модель, которую нужно обучить.
            Если None, то создается новая модель. Если str, то загружается модель из файла.
        use_tensorboard (bool): сбрасывать ли данные для tensorboard во время обучения (по умолчанию True)
        timeout (int or None): таймаут в секундах. Если не None, то обучение прерывается по истечении этого времени.
        exp_log (ExperimentLog or None): лог экспериментов
    Returns:
        Список чисел -- достигнутые значения метрик на тестовой выборке во время обучения
    """
    if hparams == 'auto':
        from .nn_recommend import recommend_hparams
        hparams = recommend_hparams(nn_task)
        
    # first, check that all nn_task.objects are available in nnDB
    unavail = [str(nm) for cid, nm in zip(cur_db().get_cat_IDs_by_names(nn_task.objects), nn_task.objects) if cid < 0]
    if len(unavail) > 0:
        raise ValueError(f'`{"`, `".join(unavail)}` not available in the training dataset')
    test_ratio = hparams.get('test_frac', 0.15)
    val_ratio = hparams.get('val_frac', 0.15)
    #print(test_ratio,val_ratio)
    
    exp_name, exp_dir = _create_exp_dir('train', nn_task)
    printlog("Prepare data subset for training")
    data = create_data_subset(nn_task.objects, exp_dir,
                              crop_bbox=hparams.get('crop_bbox', not _emulation) and nn_task.type=='classification',
                              with_segmentation=(nn_task.type=='segmentation'),
                              with_bbox=(nn_task.type=='detection'),
                              split_points=(1 - val_ratio - test_ratio, 1 - test_ratio))
    history = ExperimentHistory(nn_task, exp_name, exp_dir, data)
    return create_and_train_model(nn_task, hparams, nn_task.objects, data, exp_dir, history=history, stop_flag=stop_flag,
                                  model=model, use_tensorboard=use_tensorboard, timeout=timeout, exp_log=exp_log)


def param_values(default=None, values=None, step=None, scale=None,
                 zero_point=None, type=None, return_str=False, **kwargs):
    """Возвращает список значений гиперпараметра по более сжатому описанию.

    Args:
        default: значение по умолчанию
        values: список возможных значений, которые будут перебираться по сетке (для категориальных гиперпараметров)
        step: шаг между значениями для числовых гиперпараметров
        scale: шкала значений для числовых гиперпараметров (lin, log, 1-log)
        zero_point: добавить ли точку 0 (только для шкалы log)
        type: тип значения (int, float, str, bool)
        return_str: возвращать ли значения в виде строк (например, для отображения в web-интерфейсе)
        **kwargs: дополнительные параметры, которые будут переданы в функцию values
    Returns:
        Список значений гиперпараметра
    """
    pos = None
    if 'range' in kwargs:
        mn, mx = kwargs['range']
        if scale == 'log':
            if not default:
                pos = 0
                default = mn
            back = round(math.log(mn/default, step))
            forward = round(math.log(mx/default, step))
            res = [default * step ** i for i in range(back, forward + 1)]
            if pos is None:
                pos = -back
                if zero_point:
                    pos += 1
        elif scale == '1-log':
            if default == 1:
                default = mx
            back = round(math.log((1-mx)/(1-default), step))
            forward = round(math.log((1-mn)/(1-default), step))
            res = [1-(1-default) * step ** i for i in range(forward, back-1, -1)]
            pos = forward
        elif scale == 'lin':
            back = round((mn - default) / step)
            forward = round((mx - default) / step)
            res = [default + step * i for i in range(back, forward + 1)]
            pos = -back
        else:
            raise ValueError(f'Unknown scale {scale}')
        if type == 'int':
            res = [int(round(x)) for x in res]
        if zero_point:
            res = ['0' if return_str else 0] + res
        if return_str:
            if type == 'float':
                if scale == 'log':
                    res = [f'{x:.3}' for x in res]
                elif scale == '1-log':
                    res = [f'{x:.6f}' for x in res]
                elif step >= 1:
                    res = [f'{x:.1f}' for x in res]
                else:
                    prec = int(round(-math.log(step, 10)+0.499))
                    res = [f'{round(x,prec):.{prec}f}' for x in res]
            else:
                res = [str(x) for x in res]
        return res, pos
    elif values is not None:
        if isinstance(values, dict):
            k, v = list(values.keys()), list(values.values())
            if default is not None:
                pos = k.index(default)
            return (k, v), pos
        if default is not None:
            pos = values.index(default)
        if return_str:
            return [str(x) for x in values], pos
        return list(values), pos
    else:
        raise ValueError('Either `range` or `values` should be specified')


class HyperParamGrid:
    def __init__(self, hparams, tuned_params):
        """
        Создает пространство гиперпараметров, в рамках которого можно перебирать комбинации гиперпараметров

        Args:
            hparams (dict): словарь с текущими значениями гиперпараметров
            tuned_params (list): набор гиперпараметров, которые будут подбираться
        """
        self.hparams = hparams
        self.tuned_params = tuned_params
        self.fixed_params = [p for p in hparams if p not in tuned_params]
        self.param_control = {p: grid_hparams_space[p]['values'] for p in tuned_params
                              if 'values' in grid_hparams_space[p] and isinstance(grid_hparams_space[p]['values'], dict)}
        active = set()
        for p in self.fixed_params:
            v = hparams[p]
            if p in self.param_control:
                active.update(self.param_control[p].get(v, ()))
        self.active = {p for p in self.tuned_params if not grid_hparams_space[p].get('cond', False) or p in active}

        self.axis = []
        self.deps = []
        for param in tuned_params:
            v, _ = param_values(**grid_hparams_space[param])
            if isinstance(v, tuple):
                self.axis.append(v[0])
                self.deps.append([x.get('params', []) for x in v[1]])
            else:
                self.axis.append(v)
                self.deps.append(None)

    def remove_inactive(self, point):  # replaces inactive dependent params with None
        active = {*self.active}
        for p, i, ax, dep in zip(self.tuned_params, point, self.axis, self.deps):
            if dep is not None and p in active:
                active.update(dep[i])

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
        return key, [key, res], {}

    def get_point(self, params):
        point = []
        for p, ax in zip(self.tuned_params, self.axis):
            if p == 'lr/batch_size' and 'learning_rate' in params and 'batch_size' in params:
                v = params['learning_rate'] / params['batch_size']
            elif p not in params:
                point.append(None)
                continue
            else:
                v = params[p]
            if isinstance(v, (int, float)):
                # find index or closest value in ax
                if not isinstance(ax[0], (int, float)):
                    raise ValueError(f'Parameter {p} is not numeric but value {v} is')
                point.append(np.argmin(np.abs(np.array(ax, dtype=np.float64) - float(v))))
            else:
                point.append(ax.index(v) if v in ax else None)
        return tuple(point)


def neighborhood_gen(c, shape, cat_axis, r, metric):
    """
    Функция возвращает все точки в окрестности центра, которые не выходят за границы сетки

    Args:
        c (tuple of int): центр окрестности
        shape (list of int): размеры сетки
        cat_axis (list of int): типы величин по осям (0 -- числовая, 1 -- категориальная)
        r (int): радиус окрестности
        metric (str): метрика расстояния на сетке ('max' -- максимум, 'l1' -- манхэттенское расстояние)
    Returns:
        генератор точек в окрестности точки c радиуса r
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

    Args:
        grid_size (list of int): Размеры сетки, на которой производится оптимизация.
        cat_axis (list of int): Типы величин по осям (0 -- числовая, 1 -- категориальная)
        func (callable): Функция, которую нужно оптимизировать.
        gridmap (callable): Функция, которая преобразует точку сетки в кортеж (key, args, kwargs), где
            key - ключ для кэширования, args и kwargs - аргументы функции func.
        start_point (Union[tuple, str]): Начальная точка. Если 'random', то начальная точка выбирается случайно.
        grid_metric (str): Метрика, по которой определяется расстояние между точками сетки ('l1' или 'max').
        radius (int): Радиус окрестности, в которой производится поиск лучшей точки.
    Returns:
        Генератор троек (coords, val, is_max), где
           - coords -- кортеж с координатами текущей точки,
           - val -- значение функции в этой точке,
           - is_max -- является ли точка локальным максимумом.
    """
    if start_point == 'random':
        start_point = tuple(np.random.randint(0, grid_size[i]) for i in range(len(grid_size)))
    else:
        start_point = tuple(np.random.randint(0, grid_size[i]) if start_point[i] is None else start_point[i]
                            for i in range(len(grid_size)))

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


def hparams_grid_tune(nn_task, data, exp_name, exp_dir, hparams, tuned_params, stop_flag=None, timeout=None,
                      use_tensorboard=True, exp_log=None,
                      start_point='auto', grid_metric='l1', radius=1, verbosity=1):
    """
    Оптимизирует параметры нейронной сети на сетке.

    Args:
        nn_task (NNTask): Задача, для которой оптимизируются параметры.
        data (dict): словарь путей к csv-файлам с разметкой для train, val, test
        exp_name (str): Имя эксперимента.
        exp_dir (str): Путь к директории, в которой сохраняются результаты оптимизации.
        hparams (dict): Исходные гиперпараметры, часть из них будет оптимизироваться.
        tuned_params (list): Параметры, которые будут оптимизироваться.
        stop_flag (StopFlag, optional): Флаг, который можно использовать для остановки оптимизации.
        timeout (float, optional): Максимальное время работы оптимизации.
        use_tensorboard (bool, optional): Сбрасывать ли данные для TensorBoard.
        exp_log (ExperimentLog, optional): Объект для логирования результатов оптимизации.

        start_point (str):
           Начальная точка. Допустимые значения:
               - `random` -- начальная точка выбирается полностью случайно
               - `auto` -- параметры, присутствующие в hparams, используются в качестве начального
                 значения этих параметров, остальные параметры выбираются случайно
        grid_metric (str): Метрика, по которой определяется расстояние между точками сетки ('l1' или 'max').
        radius (int): Радиус окрестности, в которой производится поиск лучшей точки.
        verbosity (int): Уровень логирования.
    Returns:
        (dict, float, dict):
           Кортеж (best_params, best_score, best_model_info), где
              - best_params -- лучшие найденные гиперпараметры,
              - best_score -- значение метрики на лучших гиперпараметрах,
              - best_model_info -- полная информация о лучшей обученной модели (включая фиксированные гиперпараметры и путь к сохранённой модели).
    """
    from .nn_recommend import recommend_hparams
    t0 = time.time()
    grid = HyperParamGrid(hparams, tuned_params)
    grid_size = list(map(len, grid.axis))
    cat_axis = ['values' in grid_hparams_space[p] for p in tuned_params]

    history = ExperimentHistory(nn_task, exp_name, exp_dir, data)
    best_point, best_score = None, None
    params_of_best = None
    timeout = timeout or 1e10

    point_num = 0

    def fit_and_get_score(key, params):
        nonlocal point_num
        point_num += 1
        printlog(f'==================== Point {point_num:2} ====================')
        printlog(f'Parameters:')
        for ki, pi in zip(key, tuned_params):
            if ki is not None:
                printlog(f'    {pi}: {params.get(pi, None)}')
            else:
                printlog(f'    {pi}: [inactive dependent parameter]')
        key_str = '_'.join([str(x) if x is not None else 'n' for x in key])
        cur_dir = os.path.join(exp_dir, key_str)
        if os.path.exists(cur_dir):
            warnings.warn(f'Experiment {key_str} already exists.')
        os.makedirs(cur_dir, exist_ok=True)
        try:
            new_params = recommend_hparams(nn_task, fixed_params=params, trace_source=verbosity>1)
            is_tuned_changed = any(params.get(p, None) is not None and new_params.get(p, None) != params[p] for p in tuned_params)
            if is_tuned_changed:
                printlog(f'Some tuned parameter changed by recommend_hparams, so this point is invalid; skipping.')
                return -np.inf
            scores, p = create_and_train_model(new_params, nn_task.objects, data, cur_dir, history=history,
                                               stop_flag=stop_flag, use_tensorboard=use_tensorboard,
                                               timeout=timeout - (time.time() - t0), exp_log=exp_log)
            val = nn_task.func(scores)
            nonlocal params_of_best, best_score, best_point
            if best_score is None or val > best_score:
                params_of_best = params
                best_point, best_score = p, val
            return val
        except Exception as e:
            printlog(f'Error in experiment {key_str}: {e}')
            return -np.inf

    if start_point == 'auto':
        start_point = grid.get_point(hparams)

    for point, value, is_max in grid_search_gen(grid_size, cat_axis, fit_and_get_score,
                                                grid, start_point, grid_metric, radius):
        if stop_flag is not None and stop_flag.flag:
            break
        printlog(f"Evaluated point: {point}, value: {value}")
        pcall('tune_step', point, value)
        if not nn_task.goals.get('maximize', True) and best_score >= nn_task.target:
            break
        if time.time() - t0 > timeout:
            break

    printlog(f"Best point: {best_point}, value: {best_score}")
    if best_score is not None and best_score >= nn_task.target:
        printlog("achieved target score")
    else:
        printlog("did not achieve target score")

    return best_point, best_score, params_of_best


def hparams_history_tune(nn_task, data, exp_name, exp_dir, hparams, tuned_params, stop_flag=None, timeout=None,
                         use_tensorboard=True, exp_log=None,
                         exact_category_match=False):
    """
    Оптимизирует параметры нейронной сети по истории экспериментов.
    Ищутся эксперименты, где текущая задача уже решалась, и находятся лучшие гиперпараметры.

    Args:
        nn_task (NNTask): Задача, для которой оптимизируются параметры.
        data (dict): словарь путей к csv-файлам с разметкой для train, val, test
        exp_name (str): Имя эксперимента.
        exp_dir (str): Путь к директории, в которой сохраняются результаты оптимизации.
        hparams (dict): Исходные гиперпараметры, часть из них будет оптимизироваться.
        tuned_params (list): Параметры, которые будут оптимизироваться.
        stop_flag (StopFlag or None): Флаг, который можно использовать для остановки оптимизации.
        timeout (float): Максимальное время, которое может занять оптимизация.
        use_tensorboard (bool): Сбрасывать ли логи для tensorboard.
        exp_log (ExperimentLog): Объект, в который будут записываться логи.
        exact_category_match (bool): Если True, то при поиске по истории считается, что категориальные
            параметры должны совпадать точно.
    Returns:
        (dict, float): Пара (best_params, best_score), где
         -   best_params -- лучшие найденные гиперпараметры,
         -   best_score -- значение метрики на лучших гиперпараметрах.
    """
    history = ExperimentHistory(nn_task, exp_name, exp_dir, data)
    best_point, best_score = None, None
    candidates = params_from_history(nn_task)
    candidates.sort(key=lambda x: x[nn_task.metric], reverse=True)
    for params in candidates:
        if stop_flag is not None and stop_flag.flag:
            break
        cur_params = {**hparams, **params}
        scores, _ = create_and_train_model(cur_params, nn_task.objects, data, exp_dir, history=history,
                                           stop_flag=stop_flag, timeout=timeout, use_tensorboard=use_tensorboard,
                                           exp_log=exp_log)
        score = nn_task.func(scores)
        printlog(f"Evaluated point: {params}, value: {score}")
        printlog("==================================================\n")
        pcall('tune_step', params, score)
        if best_score is None or score >= best_score:
            best_point, best_score = params, score
            if score >= nn_task.target:
                break

    printlog(f"Best point: {best_point}, value: {best_score}")
    if best_score is not None and best_score >= nn_task.target:
        printlog("achieved target score")
    else:
        printlog("did not achieve target score")

    return best_point, best_score


def tune(nn_task, tuned_params, method, hparams=None, stop_flag=None, timeout=None,
         use_tensorboard=True, exp_log=None, **kwargs):
    """
    Оптимизирует гиперпараметры обучения нейронной сети.

    Args:
        nn_task (NNTask): Задача, для которой оптимизируются параметры.
        tuned_params (list or str): Параметры, которые будут оптимизироваться (если 'all',
            то будут оптимизироваться все гиперпараметры, для которых предусмотрена оптимизация).
        method (str): Метод оптимизации (пока поддерживается только 'grid').
        hparams (dict): Исходные гиперпараметры, часть из них будет оптимизироваться.
        stop_flag (StopFlag, optional): Флаг, который можно использовать для остановки оптимизации.
        timeout (int, optional): Максимальное время, которое разрешено затратить на оптимизацию.
        use_tensorboard (bool): Сбрасывать ли логи для tensorboard.
        exp_log (ExperimentLog): Объект, в который будут записываться логи.
        **kwargs: Дополнительные параметры для метода оптимизации.

    Returns:
        (dict, float, dict): Кортеж (best_params, best_score, best_model_info), где
           - best_params -- лучшие найденные гиперпараметры,
           - best_score -- значение метрики на лучших гиперпараметрах.
           - best_model_info -- полная информация о лучшей обученной
                модели (включая фиксированные гиперпараметры и путь к сохранённой модели).
    """
    if timeout is None:
        timeout = 1e10
    if tuned_params == 'all':
        tuned_params = list(grid_hparams_space)
    exp_name, exp_path = _create_exp_dir(f'tune_{method}', nn_task)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    printlog(f"Experiment path: {exp_path}")
    if hparams is None:
        # взять дефолтные значения
        hparams = get_hparams(nn_hparams)

    data = create_data_subset(nn_task.objects, exp_path)
    if method == 'grid':
        tune_func = hparams_grid_tune
    elif method == 'history':
        tune_func = hparams_history_tune
    else:
        raise ValueError(f'Unknown tuning method: {method}')
    # check kwargs of tune_func (if some key is not in kwargs, warn)
    tune_kwargs = inspect.getfullargspec(tune_func).kwonlyargs
    for k in kwargs:
        if k not in tune_kwargs:
            warnings.warn(f'Unknown argument {k} for tune function {tune_func.__name__}')
    kwargs = {k: v for k, v in kwargs.items() if k in tune_kwargs}

    return tune_func(nn_task, data, exp_name, exp_path, hparams, tuned_params,
                     stop_flag=stop_flag, timeout=timeout, use_tensorboard=use_tensorboard, exp_log=exp_log,
                     **kwargs)


def _create_exp_dir(prefix, nn_task):
    """Создает директорию для эксперимента.

    Args:
        prefix (str): Префикс имени директории.
        nn_task (NNTask): Задача.
    Returns:
        Пара (exp_name, exp_path), где
           - exp_name -- имя директории,
           - exp_path -- путь к директории.
    """
    obj_set = sorted(cur_db().get_cat_IDs_by_names(list(nn_task.objects)))
    if len(obj_set) > 10:
        obj_set = obj_set[:10] + ['etc']
    obj_str = '_'.join(map(str, obj_set))
    msk = timezone('Europe/Moscow')
    msk_time = datetime.now(msk)
    tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')
    exp_name = f'{prefix}_{obj_str}_DT_{tt}'
    exp_path = f"{_data_dir}/trainedNN/{exp_name}"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    return exp_name, exp_path


def load_history(history_file) -> dict:
    """Загружает историю обучения из файла.

    Args:
        history_file (str): Путь к файлу с историей обучения.
    Returns:
        История обучения в виде словаря.
    """
    if history_file.endswith('.json'):
        with open(history_file, 'r') as f:
            history = json.load(f)
    elif history_file.endswith('.yaml'):
        import yaml
        with open(history_file, 'r') as f:
            history = yaml.load(f, Loader=yaml.SafeLoader)
    elif history_file.endswith('.pkl'):
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
    elif history_file.endswith('.csv'):
        history = pd.read_csv(history_file)
        # convert to dict
        history = history.to_dict(orient='list')
    else:
        raise ValueError(f'Unknown history file format: {history_file}')
    return history


def params_from_history(nn_task):
    """
    Args:
        nn_task (NNTask): Задача обучения нейронной сети
    Returns:
        list: Список историй обучения для данной задачи
    """
    req = {
        'min_metrics': {nn_task.metric: nn_task.target},
        'categories': list(nn_task.objects),
    }
    history = cur_db().get_models_by_filter(req)  # history is a pandas DataFrame
    print(f'History request: {req}')
    print(f'Found {len(history)} models')
    if len(history) == 0:
        return []
    results = []
    for i, model in history.iterrows():
        hist_file = model['history_address']
        model_file = model['model_address']

        if hist_file is None:
            continue
        if not os.path.exists(hist_file):
            warnings.warn(f'History file {hist_file} does not exist')
            continue
        try:
            params = load_history(hist_file)
            params['model_file'] = model_file
            results.append(params)
        except Exception as e:
            warnings.warn(f'Error while loading history file {hist_file}: {e}')
    return results
