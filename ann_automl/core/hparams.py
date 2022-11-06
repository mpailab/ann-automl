# !!! некоторые гиперпараметры и их значения сгенерированы автоматически !!!
# TODO: проверить их на корректность
hyperparameters = {
    'epochs': {
        'type': 'int', 
        'range': [10, 1000], 
        'default': 150, 
        'step': 10, 
        'scale': 'lin', 
        'title': "Количество эпох",
        'gui': {
            'widget': 'Slider',
            'group': 'Learning'
        }
    },
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
        'title': "Оптимизатор",
        'description': """
Оптимизатор, используемый при обучении нейронной сети:
- Adam - адаптивный метод градиентного спуска, основанный на оценках первого и второго моментов градиента
- SGD - стохастический градиентный спуск
- RMSprop - адаптивный метод градиентного спуска, основанный на оценках второго момента градиента
- Adagrad - адаптивный метод градиентного спуска, основанный на сумме квадратов градиента
- Adadelta - адаптивный метод градиентного спуска, основанный на сумме квадратов градиента и градиента на предыдущем шаге
- Adamax - адаптивный метод градиентного спуска, основанный на оценках первого и максимального второго моментов градиента
- Nadam - адаптивный метод градиентного спуска, основанный на оценках первого и взвешенного второго моментов градиента
""",
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'learning_rate': {
        'type': 'float',
        'range': [1e-5, 1e-1],
        'default': 1e-3,
        'step': 2,
        'scale': 'log',
        'title': "Скорость обучения",
        'gui': {
            'widget': 'Slider',
            'group': 'Learning'
        }
    },
    'decay': {
        'type': 'float',
        'range': [0, 1],
        'default': 0.0,
        'step': 0.01,
        'scale': 'lin',
        'title': 'Декремент скорости обучения',
        'description': """
Декремент скорости обучения. Если значение больше нуля, то скорость обучения будет уменьшаться по формуле:
learning_rate = learning_rate * 1 / (1 + decay * epoch)
""",
        'gui': {
            'widget': 'Slider',
            'group': 'Learning'
        }
    },
    'activation': {
        'type': 'str', 
        'values': ['softmax', 'elu', 'selu', 'softplus',
                   'softsign', 'relu', 'tanh',
                   'sigmoid', 'hard_sigmoid', 'linear'],
        'default': 'relu', 
        'title': 'Функция активации',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'loss': {
        'type': 'str',
        'values': ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                   'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge',
                   'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                   'binary_crossentropy', 'kullback_leibler_divergence', 'poisson',
                   'cosine_proximity'],
        'default': 'mean_squared_error',
        'title': 'Функция потерь',
        'description': """
Функция потерь. При обучении нейронной сети эта функция минимизируется.
""",
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'metrics': {
        'type': 'str',
        'values': ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                   'sparse_categorical_accuracy', 'top_k_categorical_accuracy',
                   'sparse_top_k_categorical_accuracy'],
        'default': 'accuracy',
        'title': 'Метрика',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'dropout': {
        'type': 'float',
        'range': [0, 1],
        'step': 0.01,
        'scale': 'lin',
        'default': 0.0,
        'title': 'Dropout',
        'description': """
Вероятность отключения нейронов. Если значение больше нуля, то во время обучения нейроны будут отключаться случайным образом c такой вероятностью.
""",
        'gui': {
            'widget': 'Slider',
            'group': 'Learning'
        }
    },
    'kernel_initializer': {
        'type': 'str', 
        'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                   'he_uniform'],
        'default': 'glorot_uniform', 
        'title': 'Инициализатор весов',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'bias_initializer': {
        'type': 'str', 
        'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                   'he_uniform'], 
        'default': 'zeros', 
        'title': 'Инициализатор смещений',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'kernel_regularizer': {
        'type': 'str', 
        'values': ['auto', 'l1', 'l2', 'l1_l2'],
        'default': 'auto', 
        'title': 'Регуляризатор весов',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'bias_regularizer': {
        'type': 'str', 
        'values': ['auto', 'l1', 'l2', 'l1_l2'],
        'default': 'auto', 
        'title': 'Регуляризатор смещений',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'activity_regularizer': {
        'type': 'str', 
        'values': ['auto', 'l1', 'l2', 'l1_l2'],
        'default': 'auto', 
        'title': 'Регуляризатор активации',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'kernel_constraint': {
        'type': 'str', 
        'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
        'default': 'auto', 
        'title': 'Ограничение весов',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'bias_constraint': {
        'type': 'str', 
        'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
        'default': 'auto', 
        'title': 'Ограничение смещений',
        'gui': {
            'widget': 'Select',
            'group': 'Learning'
        }
    },

    # conditional parameters (for optimizers)
    'nesterov': { 
        'type': 'bool',
        'default': False,
        'title': 'Nesterov momentum',
        'cond': [('optimizer', {'SGD'})],
        'gui': {
            'widget': 'Checkbox',
            'group': 'Optimizer'
        }
    },
    'centered': { 
        'type': 'bool',
        'default': False,
        'title': 'Centered',
        'cond': [('optimizer', {'RMSprop'})],
        'gui': {
            'widget': 'Checkbox',
            'group': 'Optimizer'
        }
    },
    'amsgrad': {
        'type': 'bool',
        'default': False,
        'title': 'Amsgrad',
        'cond': [('optimizer', {'Adam'})],
        'gui': {
            'widget': 'Checkbox',
            'group': 'Optimizer'
        }
    },
    'momentum': {
        'type': 'float', 
        'range': [0, 1], 
        'default': 0.0, 
        'step': 0.01, 
        'scale': 'lin',
        'title': 'momentum', 
        'cond': [('optimizer', {'SGD'})],
        'gui': {
            'widget': 'Slider',
            'group': 'Optimizer'
        }
    },
    'rho': {
        'type': 'float', 
        'range': [0.5, 0.99], 
        'default': 0.9,
        'step': 2**0.25, 
        'scale': 'loglog', 
        'title': 'rho', 
        'cond': [('optimizer', {'RMSprop'})],
        'gui': {
            'widget': 'Slider',
            'group': 'Optimizer'
        }
    },
    'epsilon': {
        'type': 'float', 
        'range': [1e-8, 1e-1], 
        'default': 1e-7, 
        'step': 10, 
        'scale': 'log',
        'title': 'epsilon', 
        'cond': [('optimizer', {'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'})],
        'gui': {
            'widget': 'Slider',
            'group': 'Optimizer'
        }
    },
    'beta_1': {
        'type': 'float', 
        'range': [0.5, 0.999], 
        'default': 0.9,
        'step': 2**0.25, 
        'scale': 'loglog', 
        'title': 'beta_1', 
        'cond': [('optimizer', {'Adam', 'Nadam', 'Adamax'})],
        'gui': {
            'widget': 'Slider',
            'group': 'Optimizer'
        }
    },
    'beta_2': {
        'type': 'float', 
        'range': [0.5, 0.9999], 
        'default': 0.999,
        'step': 2**0.25, 
        'scale': 'loglog', 
        'title': 'beta_2', 
        'cond': [('optimizer', {'Adam', 'Nadam', 'Adamax'})],
        'gui': {
            'widget': 'Slider',
            'group': 'Optimizer'
        }
    }, 
}
