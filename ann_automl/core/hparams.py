# !!! гиперпараметры и их значения сгенерированы автоматически !!!
# TODO: проверить их на корректность
hyperparameters = {
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
                       "- Adam - адаптивный метод градиентного спуска, основанный на оценках первого и второго моментов градиента\n"
                       "- SGD - стохастический градиентный спуск\n"
                       "- RMSprop - адаптивный метод градиентного спуска, основанный на оценках второго момента градиента\n"
    },
    'learning_rate': {'type': 'float',
                      'range': [1e-5, 1e-1],
                      'default': 1e-3,
                      'step': 2,
                      'scale': 'log',
                      'name': "скорость обучения",
                      },
    'decay': {'type': 'float',
              'range': [0, 1],
              'default': 0.0,
              'step': 0.01,
              'scale': 'lin',
              'name': 'декремент скорости обучения',
              'description': "Декремент скорости обучения. Если значение больше нуля, "
                             "то скорость обучения будет уменьшаться по формуле:\n"
                             "learning_rate = learning_rate * 1 / (1 + decay * epoch)"
              },
    'activation': {'type': 'str', 'values': ['softmax', 'elu', 'selu', 'softplus',
                                             'softsign', 'relu', 'tanh',
                                             'sigmoid', 'hard_sigmoid', 'linear'],
                   'default': 'relu', 'name': 'функция активации'},
    'loss': {
        'type': 'str',
        'values': ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                   'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge',
                   'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                   'binary_crossentropy', 'kullback_leibler_divergence', 'poisson',
                   'cosine_proximity'],
        'default': 'mean_squared_error',
        'name': 'функция потерь',
        'description': "Функция потерь. При обучении нейронной сети эта функция минимизируется.\n"
    },
    'metrics': {
        'type': 'str',
        'values': ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                   'sparse_categorical_accuracy', 'top_k_categorical_accuracy',
                   'sparse_top_k_categorical_accuracy'],
        'default': 'accuracy',
        'name': 'метрика'
    },
    'dropout': {
        'type': 'float',
        'range': [0, 1],
        'default': 0.0,
        'name': 'dropout',
        'description': "Вероятность отключения нейронов. Если значение больше нуля, то во время "
                       "обучения нейроны будут отключаться случайным образом c такой вероятностью."
    },
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
    'nesterov': {  # для SGD
        'type': 'bool',
        'default': False,
        'name': 'Nesterov momentum',
        'cond': True,
    },
    'centered': {  # для RMSprop
        'type': 'bool',
        'default': False,
        'name': 'centered',
        'cond': True,
    },
    'amsgrad': {  # для Adam
        'type': 'bool',
        'default': False,
        'name': 'amsgrad для Adam',
        'cond': True
    },

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
