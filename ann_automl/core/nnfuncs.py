import os
import time

import keras
import pandas as pd

from . import db_module

_data_dir = 'data'
_db_file = 'tests.sqlite'

nnDB = db_module.dbModule(dbstring=f'sqlite:///{_db_file}')  # TODO: уточнить путь к файлу базы данных


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
hyperparameters = {
    'batch_size': {'type': 'int', 'range': [1, 128], 'default': 32, 'step': 2, 'scale': 'log', 'name': "размер батча"},
    'epochs': {'type': 'int', 'range': [10, 1000], 'default': 150, 'step': 10, 'scale': 'lin', 'name': "количество эпох"},
    'optimizer': {'type': 'str', 'values': {
        'Adam': {'params': ['amsgrad', 'beta_1', 'beta_2', 'epsilon']},
        'SGD': {'scale': {'learning_rate': 10}, 'params': ['nesterov', 'momentum']},
        'RMSprop': {'params': ['rho', 'epsilon', 'momentum', 'centered']},
        'Adagrad': {'params': ['epsilon']},
        'Adadelta': {'params': ['rho', 'epsilon']},
        'Adamax': {'params': ['beta_1', 'beta_2', 'epsilon']},
        'Nadam': {'params': ['beta_1', 'beta_2', 'epsilon']},
    }, 'default': 'Adam', 'name': "оптимизатор"},
    'learning_rate': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'step': 2, 'scale': 'log',
                      'name': "скорость обучения"},
    'decay': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'name': 'декремент скорости обучения'},
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

    # conditional parameters (for optimizers)
    'nesterov': {'type': 'bool', 'default': False, 'name': 'Nesterov momentum', 'cond': True},  # для SGD
    'centered': {'type': 'bool', 'default': False, 'name': 'centered', 'cond': True},  # для RMSprop
    'amsgrad': {'type': 'bool', 'default': False, 'name': 'amsgrad для Adam', 'cond': True},  # для Adam

    'momentum': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'step': 0.01, 'scale': 'lin',
                 'name': 'momentum', 'cond': True},  # момент для SGD
    'rho': {'type': 'float', 'range': [0.5, 0.99], 'default': 0.9, 'name': 'rho', 'cond': True,
            'step': 2**0.25, 'scale': 'loglog'},  # коэффициент затухания для RMSprop
    'epsilon': {'type': 'float', 'range': [1e-8, 1e-1], 'default': 1e-7, 'step': 10, 'scale': 'log',
                'name': 'epsilon', 'cond': True},  # для RMSprop, Adagrad, Adadelta, Adamax, Nadam
    'beta_1': {'type': 'float', 'range': [0.5, 0.999], 'default': 0.9, 'name': 'beta_1 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': 'loglog'},  # для Adam, Nadam, Adamax
    'beta_2': {'type': 'float', 'range': [0.5, 0.9999], 'default': 0.999, 'name': 'beta_2 для Adam', 'cond': True,
               'step': 2**0.25, 'scale': 'loglog'},  # для Adam, Nadam, Adamax
}


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


def create_data_subset(objects, temp_dir='tmp'):
    """ Создание подвыборки данных для обучения

    Parameters
    ----------
    objects : list
        Список категорий, для которых необходимо создать подвыборку
    temp_dir : str
        Путь к папке, в которой будут созданы подвыборки
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    return nnDB.load_specific_categories_annotations(list(objects), normalizeCats=True,
                                                     splitPoints=[0.7, 0.85],
                                                     curExperimentFolder=temp_dir,
                                                     crop_bbox=True,
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


def create_layer(layer_type, **kwargs):
    return getattr(keras.layers, layer_type)(**kwargs)


def create_model(base, last_layers):
    input_shape = base.input_shape[1:]
    y = keras.models.load_model(f'{_data_dir}/architectures/{base}.h5')
    x = keras.layers.Input(shape=input_shape)
    y = y(x)
    for layer in last_layers:
        y = create_layer(**layer)(y)

    return keras.models.Model(inputs=x, outputs=y)


class ExperimentHistory:
    def __init__(self, task):
        self.experiment_number = 0
        self.exp_name = task.exp_name
        self.exp_path = task.exp_path
        self.task_type = task.task_type
        self.objects = task.objects
        self.data = task.data

        self.history = pd.DataFrame(columns=['Index', 'task_type', 'objects', 'exp_name', 'pipeline', 'last_layers',
                                             'augmen_params', 'loss', 'metrics', 'epochs', 'stop_criterion', 'data',
                                             'optimizer', 'batch_size', 'lr', 'metric_test_value',
                                             'train_subdir', 'time_stat', 'total_time', 'Additional_params'])

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
                'stop_criterion': params['stop_criterion'],
                # критерий остановки обучения (TODO: не используется, исправить!!!)
                'data': params['data'],  # набор данных, на котором проводится обучение

                'optimizer': params['optimizer'],  # оптимизатор
                'batch_size': params['batch_size'],  # размер батча
                'lr': params['lr'],  # скорость обучения

                'metric_test_value': metric,  # значение метрики на тестовой выборке
                'train_subdir': train_subdir,  # папка, в которой хранятся результаты текущего обучения
                'time_stat': time_stat,  # список длительностей всех эпох обучения
                'total_time': total_time,  # общее время обучения
                'Additional_params': [{}]})

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
                'lr': best_model['lr']}


class StopFlag:
    def __init__(self):
        self.flag = False


class CheckStopCallback(keras.callbacks.Callback):
    def __init__(self, stop_flag):
        super().__init__()
        self.stop_flag = stop_flag

    def on_batch_end(self, batch, logs=None):
        if self.stop_flag.flag:
            self.model.stop_training = True


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
    optimizer, lr = hparams['optimizer'], hparams['lr']
    opt_args = ['decay'] + hyperparameters['optimizer']['values'][optimizer].get('params', [])
    kwargs = {arg: hparams[arg] for arg in opt_args if arg in hparams}
    optimizer = getattr(keras.optimizers, optimizer)(learning_rate=lr, **kwargs)
    model.compile(optimizer=optimizer, loss=hparams['loss'], metrics=[hparams['metrics']])

    # set up callbacks
    check_metric = 'val_' + hparams['metrics']
    c_log = keras.callbacks.CSVLogger(cur_subdir + '/Log.csv', separator=',', append=True)
    c_ch = keras.callbacks.ModelCheckpoint(cur_subdir + '/weights-{epoch:02d}.h5', monitor=check_metric, verbose=1,
                                           save_best_only=True, save_weights_only=False, mode='auto')
    c_es = keras.callbacks.EarlyStopping(monitor=check_metric, min_delta=0.001, mode='auto', patience=5)
    c_t = TimeHistory()
    callbacks = [c_log, c_ch, c_es, c_t]
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
