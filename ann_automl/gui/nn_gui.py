import panel as pn
import param
import pandas as pd
import datetime as dt
import bokeh

from ann_automl.core.db_module import dbModule
from ann_automl.core.solver import Task
from ann_automl.gui.transition import Transition

css = '''
.bk.panel-widget-box {
    background: #fafafa;
    border-radius: 0px;
    border: 1px solid #dcdcdc;
    overflow: auto !important;
}
'''

pn.extension(raw_css=[css])
pn.config.sizing_mode='stretch_width'

hyperparameters = {
    'task': {
        'type': 'Task',
        'default': None,
        'name': 'Задача',
        'gui': {
            'window': 'Start',
            'widget': 'Info'
        }
    },
    'dataset': {
        'type': 'dbModule',
        'default': None,
        'name': 'Датасет',
        'gui': {
            'window': 'Start',
            'widget': 'Info'
        }
    },
    'task_category': {
        'type': 'str',
        'values': ['train', 'test', 'database', 'history'],
        'default': 'train',
        'name': 'Категория задачи',
        'gui': {
            'window': 'TaskParams',
            'widget': 'Select',
            'order': 1
        }
    },
    'task_type': {
        'type': 'str',
        'values': ['classification', 'segmentation', 'detection'],
        'default': 'classification',
        'name': 'Тип задачи',
        'gui': {
            'window': 'TaskParams',
            'widget': 'Select',
            'order': 2
        }
    },
    'objects_categories': {
        'type': 'list',
        'values': ['cat', 'dog', 'car', 'flower'],
        'default': [],
        'name': 'Категории объектов интереса',
        'gui': {
            'window': 'TaskParams',
            'widget': 'MultiChoice',
            'order': 3
        }
    },
    'target': {
        'type': 'str',
        'values': ['loss', 'metrics'],
        'default': 'loss',
        'name': 'Целевой параметр',
        'gui': {
            'window': 'TaskParams',
            'widget': 'Select',
            'order': 4
        }
    },
    'target_value': {
        'type': 'float',
        'range': [0, 1], 
        'default': 0.7, 
        'step': 0.05, 
        'scale': 'lin',
        'name': 'Целевой параметр',
        'gui': {
            'window': 'TaskParams',
            'widget': 'Slider',
            'order': 5
        }
    },
    'epochs': {
        'type': 'int', 
        'range': [10, 1000], 
        'default': 150, 
        'step': 10, 
        'scale': 'lin', 
        'name': "Количество эпох",
        'gui': {
            'window': 'ModelParams',
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
        'name': "Оптимизатор",
        'description': """
Оптимизатор, используемый при обучении нейронной сети:
- Adam - адаптивный метод градиентного спуска, основанный на оценках первого и второго моментов градиента
- SGD - стохастический градиентный спуск
- RMSprop - адаптивный метод градиентного спуска, основанный на оценках второго момента градиента
""",
        'gui': {
            'window': 'ModelParams',
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
        'name': "Скорость обучения",
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Декремент скорости обучения',
        'description': """
Декремент скорости обучения. Если значение больше нуля, то скорость обучения будет уменьшаться по формуле:
learning_rate = learning_rate * 1 / (1 + decay * epoch)
""",
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Функция активации',
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Функция потерь',
        'description': """
Функция потерь. При обучении нейронной сети эта функция минимизируется.
""",
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Метрика',
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Dropout',
        'description': """
Вероятность отключения нейронов. Если значение больше нуля, то во время обучения нейроны будут отключаться случайным образом c такой вероятностью.
""",
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Инициализатор весов',
        'gui': {
            'window': 'ModelParams',
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
        'name': 'Инициализатор смещений',
        'gui': {
            'window': 'ModelParams',
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'kernel_regularizer': {
        'type': 'str', 
        'values': ['auto', 'l1', 'l2', 'l1_l2'],
        'default': 'auto', 
        'name': 'Регуляризатор весов',
        'gui': {
            'window': 'ModelParams',
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'bias_regularizer': {
        'type': 'str', 
        'values': ['auto', 'l1', 'l2', 'l1_l2'],
        'default': 'auto', 
        'name': 'Регуляризатор смещений',
        'gui': {
            'window': 'ModelParams',
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'activity_regularizer': {
        'type': 'str', 
        'values': ['auto', 'l1', 'l2', 'l1_l2'],
        'default': 'auto', 
        'name': 'Регуляризатор активации',
        'gui': {
            'window': 'ModelParams',
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'kernel_constraint': {
        'type': 'str', 
        'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
        'default': 'auto', 
        'name': 'Ограничение весов',
        'gui': {
            'window': 'ModelParams',
            'widget': 'Select',
            'group': 'Learning'
        }
    },
    'bias_constraint': {
        'type': 'str', 
        'values': ['auto', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
        'default': 'auto', 
        'name': 'Ограничение смещений',
        'gui': {
            'window': 'ModelParams',
            'widget': 'Select',
            'group': 'Learning'
        }
    },

    # conditional parameters (for optimizers)
    'nesterov': { 
        'type': 'bool',
        'default': False,
        'name': 'Nesterov momentum',
        'cond': [('optimizer', set(['SGD']))],
        'gui': {
            'window': 'ModelParams',
            'widget': 'Checkbox',
            'group': 'Optimizer'
        }
    },
    'centered': { 
        'type': 'bool',
        'default': False,
        'name': 'Centered',
        'cond': [('optimizer', set(['RMSprop']))],
        'gui': {
            'window': 'ModelParams',
            'widget': 'Checkbox',
            'group': 'Optimizer'
        }
    },
    'amsgrad': {
        'type': 'bool',
        'default': False,
        'name': 'Amsgrad',
        'cond': [('optimizer', set(['Adam']))],
        'gui': {
            'window': 'ModelParams',
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
        'name': 'momentum', 
        'cond': [('optimizer', set(['SGD']))],
        'gui': {
            'window': 'ModelParams',
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
        'name': 'rho', 
        'cond': [('optimizer', set(['RMSprop']))],
        'gui': {
            'window': 'ModelParams',
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
        'name': 'epsilon', 
        'cond': [('optimizer', set(['RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']))],
        'gui': {
            'window': 'ModelParams',
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
        'name': 'beta_1', 
        'cond': [('optimizer', set(['Adam', 'Nadam', 'Adamax']))],
        'gui': {
            'window': 'ModelParams',
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
        'name': 'beta_2', 
        'cond': [('optimizer', set(['Adam', 'Nadam', 'Adamax']))],
        'gui': {
            'window': 'ModelParams',
            'widget': 'Slider',
            'group': 'Optimizer'
        }
    }, 
}



class Window(param.Parameterized):

    params = param.Dict({ p : hyperparameters[p]['default'] for p in hyperparameters })
    ready = param.Boolean(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def close(self):
        self.ready=True

    def _create_parameter_widget(self, name, desc):

        def changeValue(attr, old, new):
            self.params[name] = new

        def changeActiveValue(attr, old, new):
            self.params[name] = name in new
        
        kwargs =  {'name': name, 'margin': (5, 10, 5, 10) }

        if desc['gui']['widget'] == 'Select':
            widget = bokeh.models.Select(**kwargs, title=desc['name'], value=self.params[name], 
                options=[x for x in desc['values']])
            widget.on_change('value', changeValue)

        elif desc['gui']['widget'] == 'MultiChoice':
            widget = bokeh.models.MultiChoice(**kwargs, title=desc['name'], value=self.params[name], 
                options=[x for x in desc['values']])
            widget.on_change('value', changeValue)

        elif desc['gui']['widget'] == 'Slider':
            widget = bokeh.models.Slider(**kwargs, title=desc['name'], value=self.params[name], 
                start=desc['range'][0], end=desc['range'][1], step=desc['step'])
            widget.on_change('value', changeValue)

        elif desc['gui']['widget'] == 'Checkbox':
            widget = bokeh.models.CheckboxGroup(**kwargs, labels=[desc['name']], 
                active=[desc['name']] if self.params[name] else [])
            widget.on_change('active', changeActiveValue)

        return widget

    def create_parameters_widgets(self, window, group=None):
        widgets = []
        for par, desc in hyperparameters.items():
            if ('gui' in desc and 'window' in desc['gui'] and desc['gui']['window'] == window and
                (group is None or 'group' in desc['gui'] and desc['gui']['group'] == group) ):
                widget = self._create_parameter_widget(par, desc)
                order = desc['gui']['order'] if 'order' in desc['gui'] else 0
                widgets.append((widget, order))
        widgets.sort(key=lambda x: x[1])
        return [ x[0] for x in widgets ]

    def is_parameter_widget_visible(self, widget):
        return ( widget.name not in hyperparameters or 
                'cond' not in hyperparameters[widget.name] or 
                all (self.params[par] in values for par, values in hyperparameters[widget.name]['cond']) )



class Start(Window):

    next_window = param.Selector(default='ModelParams', objects=['TaskParams', 'DatasetParams', 'ModelParams', 'TrainedModels'])
    
    def __init__(self, **params):
        super().__init__(**params)
        
        if self.params['task'] is not None:
            print( str(self.params['task']))

        self.task_widget = pn.Column(
            '## ' + hyperparameters['task']['name'],
            pn.WidgetBox(
                bokeh.models.PreText(text='Не определена' if self.params['task'] is None else str(self.params['task']),
                    height=300),
                '### ', 
                height=200, height_policy="fixed", css_classes=['panel-widget-box'], margin=(-10,5,10,5)))
            

        self.dataset_widget = pn.Column(
            '## ' + hyperparameters['dataset']['name'],
            pn.WidgetBox( 
                bokeh.models.Paragraph(text='Не задан' if self.params['dataset'] is None else 'Должно быть описание',
                    height=300),
                '### ', 
                height=200, height_policy="fixed", css_classes=['panel-widget-box'], margin=(-10,5,10,5)))

        self.task_button=pn.widgets.Button(name='Определить задачу', align='start', width=100, button_type='primary')
        self.task_button.on_click(self.on_click_task)

        self.dataset_button=pn.widgets.Button(name='Задать датасет', align='start', width=100, button_type='primary')
        self.dataset_button.on_click(self.on_click_dataset)

        self.checkbox = pn.widgets.Checkbox(name='Подобрать готовые модели')

        self.next_button=pn.widgets.Button(name='Далее', align='end', width=100, button_type='primary')
        self.next_button.on_click(self.on_click_next)

    def on_click_task(self, event):
        self.next_window = 'TaskParams'
        self.close()

    def on_click_dataset(self, event):
        self.next_window = 'DatasetParams'
        self.close()

    def on_click_next(self, event):
        if self.checkbox.value:
            self.next_window = 'TrainedModels'
        self.close()

    def panel(self):
        return pn.Column(
            '# Главное меню',
            pn.Row(
                pn.Column(self.task_widget, self.task_button),
                pn.Spacer(width=10),
                pn.Column(self.dataset_widget, self.dataset_button)),
            pn.Spacer(height=10),
            self.checkbox,
            self.next_button,
            margin=(0,0,0,10))



class TaskParams(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.params_widgets = self.create_parameters_widgets('TaskParams')

        self.apply_button=pn.widgets.Button(name='Применить', align='end', width=100, button_type='primary')
        self.apply_button.on_click(self.on_click_apply)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_apply(self, event):
        
        # CORE:  
        self.params['task'] = Task(
            self.params['task_category'], 
            self.params['task_type'],
            self.params['objects_categories'], 
            goal={self.params['target']: self.params['target_value']})

        self.close()

    def on_click_back(self, event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Параметры задачи',
            *self.params_widgets,
            pn.Spacer(height=10),
            pn.Row(self.back_button, self.apply_button))



class DatasetParams(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.params_widgets = self.create_parameters_widgets('DatasetParams')

        self.apply_button=pn.widgets.Button(name='Применить', align='end', width=100, button_type='primary')
        self.apply_button.on_click(self.on_click_apply)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_apply(self, event):
        
        # CORE:  
        self.params['dataset'] = dbModule()

        self.close()

    def on_click_back(self, event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Параметры датасета',
            *self.params_widgets,
            pn.Spacer(height=10),
            pn.Row(self.back_button, self.apply_button))



class ModelParams(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.params_widgets = [
            ("Параметры обучения", self.create_parameters_widgets('ModelParams', group='Learning')),
            ("Параметры оптимизатора", self.create_parameters_widgets('ModelParams', group='Optimizer'))
        ]

        self.learning_params_widgets = self.create_parameters_widgets('ModelParams', group='Learning')
        self.optimizer_params_widgets = self.create_parameters_widgets('ModelParams', group='Optimizer')

        def to_column(widgets):
            return bokeh.models.Column( bokeh.models.Spacer(height=10), *widgets,
                sizing_mode="stretch_width", height=500, height_policy="fixed", css_classes=['scrollable'])

        self.tabs = bokeh.models.Tabs(
            tabs=[ bokeh.models.Panel(title=title, child=to_column(widgets)) for title, widgets in self.params_widgets ])

        def panelActive(attr, old, new):
            if self.tabs.active == 1:
                for widget in self.tabs.tabs[self.tabs.active].child.children:
                    widget.visible = self.is_parameter_widget_visible(widget)
        
        self.tabs.on_change('active', panelActive)
        
        self.next_button=pn.widgets.Button(name='Далее', align='end', width=100, button_type='primary')
        self.next_button.on_click(self.on_click_back)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

        self.qq = pn.widgets.StaticText(name='Static Text', value=self.params['task_category'])

    def on_click_back(self, event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню параметров',
            self.tabs,
            pn.Spacer(height=10),
            pn.Row(self.back_button, self.next_button))



class TrainedModels(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.models_list = pn.widgets.Select(name='Список обученных моделй', options=['Модель 1', 'Модель 2', 'Модель 3', 'Модель 4', 'Модель 5'], size=31)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_back(self,event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню обученных моделей', 
            pn.Row(
                self.models_list,
                pn.WidgetBox('## Описание модели', min_width=500, height=500)
            ),
            self.back_button
        )



pipeline = Transition(
    stages=[
        ('Start', Start),
        ('TaskParams', TaskParams),
        ('DatasetParams', DatasetParams),
        ('ModelParams', ModelParams),
        ('TrainedModels', TrainedModels)
    ],
    graph={
        'Start': ('TaskParams', 'DatasetParams', 'ModelParams', 'TrainedModels'),
        'TaskParams': 'Start',
        'DatasetParams': 'Start',
        'ModelParams': 'Start',
        'TrainedModels': 'Start'
    },
    root='Start',
    ready_parameter='ready', 
    next_parameter='next_window',
    auto_advance=True
)


template = pn.template.MaterialTemplate(
    title="Ann Automl App",
    sidebar=[pn.pane.Markdown("## Settings")],
    main=[
        pipeline.stage, 
        # pn.layout.Divider(margin=(50, 0, 50, 0)), pn.Row(pipeline.network, pipeline.buttons)
    ]
)

# template.servable()

# class ModelParams(param.Parameterized):
#     x                       = param.Parameter(default=3.14, doc="X position")
#     y                       = param.Parameter(default="Not editable", constant=True)
#     string_value            = param.String(default="str", doc="A string")
#     num_int                 = param.Integer(50000, bounds=(-200, 100000))
#     unbounded_int           = param.Integer(23)
#     float_with_hard_bounds  = param.Number(8.2, bounds=(7.5, 10))
#     float_with_soft_bounds  = param.Number(0.5, bounds=(0, None), softbounds=(0,2))
#     unbounded_float         = param.Number(30.01, precedence=0)
#     hidden_parameter        = param.Number(2.718, precedence=-1)
#     integer_range           = param.Range(default=(3, 7), bounds=(0, 10))
#     float_range             = param.Range(default=(0, 1.57), bounds=(0, 3.145))
#     dictionary              = param.Dict(default={"a": 2, "b": 9})
#     boolean                 = param.Boolean(True, doc="A sample Boolean parameter")
#     color                   = param.Color(default='#FFFFFF')
#     date                    = param.Date(dt.datetime(2017, 1, 1),
#                                          bounds=(dt.datetime(2017, 1, 1), dt.datetime(2017, 2, 1)))
#     dataframe               = param.DataFrame(pd._testing.makeDataFrame().iloc[:3])
#     select_string           = param.ObjectSelector(default="yellow", objects=["red", "yellow", "green"])
#     select_fn               = param.ObjectSelector(default=list,objects=[list, set, dict])
#     int_list                = param.ListSelector(default=[3, 5], objects=[1, 3, 5, 7, 9], precedence=0.5)
#     single_file             = param.FileSelector(path='../../*/*.py*', precedence=0.5)
#     multiple_files          = param.MultiFileSelector(path='../../*/*.py?', precedence=0.5)
#     timestamps = []
#     record_timestamp        = param.Action(lambda x: x.timestamps.append(dt.datetime.utcnow()),
#                                            doc="""Record timestamp.""", precedence=0.7)
