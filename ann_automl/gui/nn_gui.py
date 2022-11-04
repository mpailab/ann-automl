import panel as pn
import param
import pandas as pd
import datetime as dt
import bokeh
from typing import Callable

from ann_automl.core.db_module import DBModule
from ann_automl.core.solver import Task
from ann_automl.core.hparams import hyperparameters
from ann_automl.gui.transition import Transition

css = '''
.bk.panel-widget-box {
    background: #fafafa;
    border-radius: 0px;
    border: 1px solid #dcdcdc;
    overflow: auto !important;
}
'''

DB = DBModule("sqlite:///../tests.sqlite")

pn.extension(raw_css=[css])
pn.config.sizing_mode='stretch_width'

gui_params = {
    
    'task': {
        'default': None,
        'title': 'Задача'
    },
    'db': {
        'default': { 
            ds['dataset_info']['description'][0]: {
                'url': ds['dataset_info']['url'][0],
                'version': ds['dataset_info']['version'][0],
                'year': ds['dataset_info']['year'][0],
                'contributor': ds['dataset_info']['contributor'][0],
                'date_created': ds['dataset_info']['date_created'][0],
                'categories': { 
                    x[0]: { 
                        'select': False,
                        'number': x[1]
                    } 
                    for x in ds['categories'].values()
                }
            } 
            for df in [DB.get_all_datasets()] for ds_id in df['ID'] for ds in [DB.get_full_dataset_info(ds_id)]
        },
        'title': 'База данных'
    },
    'dataset': {
        'default': "Dogs vs Cats",
        'title': 'Датасет'
    },
    'task_category': {
        'type': 'str',
        'values': ['train', 'test', 'database', 'history'],
        'default': 'train',
        'title': 'Категория задачи',
        'gui': {
            'group': 'Task',
            'widget': 'Select'
        }
    },
    'task_type': {
        'type': 'str',
        'values': ['classification', 'segmentation', 'detection'],
        'default': 'classification',
        'title': 'Тип задачи',
        'gui': {
            'group': 'Task',
            'widget': 'Select'
        }
    },
    'task_objects': {
        'type': 'list',
        'values': [],
        'default': [],
        'title': 'Категории изображений',
        'gui': {
            'group': 'Task',
            'widget': 'MultiChoice'
        }
    },
    'task_goal': {
        'type': 'str',
        'values': ['loss', 'metrics'],
        'default': 'loss',
        'title': 'Целевой функционал',
        'gui': {
            'group': 'Task',
            'widget': 'Select'
        }
    },
    'task_goal_value': {
        'type': 'float',
        'range': [0, 1], 
        'default': 0.7, 
        'step': 0.05, 
        'scale': 'lin',
        'title': 'Значение целевого функционала',
        'gui': {
            'group': 'Task',
            'widget': 'Slider'
        }
    }
}

gui_params.update({ k:v for k,v in hyperparameters.items() if 'gui' in v })


class Window(param.Parameterized):

    params = param.Dict({ p : gui_params[p]['default'] for p in gui_params })
    ready = param.Boolean(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def close(self):
        self.ready=True

    def param_widget(self, name: str, change_callback: Callable[[], None]):

        def changeValue(attr, old, new):
            self.params[name] = new
            change_callback()

        def changeActiveValue(attr, old, new):
            self.params[name] = name in new
            change_callback()
        
        desc = gui_params[name]
        kwargs =  {
            'name': name,
            'margin': (5, 10, 5, 10)
        }

        if desc['gui']['widget'] == 'Select':
            widget = bokeh.models.Select(**kwargs, title=desc['title'], value=self.params[name], 
                options=[x for x in desc['values']])
            widget.on_change('value', changeValue)

        elif desc['gui']['widget'] == 'MultiChoice':
            widget = bokeh.models.MultiChoice(**kwargs, title=desc['title'], value=self.params[name], 
                options=[x for x in desc['values']])
            widget.on_change('value', changeValue)

        elif desc['gui']['widget'] == 'Slider':
            widget = bokeh.models.Slider(**kwargs, title=desc['title'], value=self.params[name], 
                start=desc['range'][0], end=desc['range'][1], step=desc['step'])
            widget.on_change('value', changeValue)

        elif desc['gui']['widget'] == 'Checkbox':
            widget = bokeh.models.CheckboxGroup(**kwargs, labels=[desc['title']], 
                active=[desc['title']] if self.params[name] else [])
            widget.on_change('active', changeActiveValue)

        return widget

    def group_params_widgets(self, group: str, change_callback: Callable[[], None]):
        # widgets = []
        for par, desc in gui_params.items():
            if 'gui' in desc and 'group' in desc['gui'] and desc['gui']['group'] == group:
                widget = self.param_widget(par, change_callback)
                yield widget
        #         widgets.append(widget)
        # return widgets

    def is_param_widget_visible(self, widget):
        return ( widget.name not in gui_params or 
                'cond' not in gui_params[widget.name] or 
                all (self.params[par] in values for par, values in gui_params[widget.name]['cond']) )



class Start(Window):

    next_window = param.Selector(default='Params', objects=['Params', 'TrainedModels', 'DatasetLoader'])
    
    def __init__(self, **params):
        super().__init__(**params)

        self.is_need_dataset_apply_button_visible = True

        def changeDatasetCallback(attr, old, new):
            ds = new[0]
            self.params['dataset'] = ds

            self.dataset_description.text=f"<b>Описание:</b> {ds}"
            self.dataset_url.text=f"<b>Источник:</b> <a href=\"{self.params['db'][ds]['url']}\">{self.params['db'][ds]['url']}</a>"
            self.dataset_contributor.text=f"<b>Создатель:</b> {self.params['db'][ds]['contributor']}"
            self.dataset_year.text=f"<b>Год создания:</b> {self.params['db'][ds]['year']}"
            self.dataset_version.text=f"<b>Версия:</b> {self.params['db'][ds]['version']}"
            self.dataset_categories_selector.options=list(self.params['db'][ds]['categories'].keys())
            self.is_need_dataset_apply_button_visible = False
            self.dataset_categories_selector.value=[category for category in self.params['db'][ds]['categories'] if self.params['db'][ds]['categories'][category]['select']]
            self.is_need_dataset_apply_button_visible = True

        self.dataset_selector = bokeh.models.MultiSelect(
            value=["Dogs vs Cats"], 
            options=list(self.params['db'].keys()),  
            height_policy="max", margin=(5,15,5,5)
        )
        self.dataset_selector.on_change('value', changeDatasetCallback)
        
        def changeDatasetCategoriesCallback(attr, old, new):
            if self.is_need_dataset_apply_button_visible:
                self.dataset_apply_button.visible = True

        self.dataset_description = bokeh.models.Div(
            text=f"<b>Описание:</b> {self.params['dataset']}"
        )
        self.dataset_url = bokeh.models.Div(
            text=f"<b>Источник:</b> <a href=\"{self.params['db'][self.params['dataset']]['url']}\">{self.params['db'][self.params['dataset']]['url']}</a>"
        )
        self.dataset_contributor = bokeh.models.Div(
            text=f"<b>Создатель:</b> {self.params['db'][self.params['dataset']]['contributor']}"
        )
        self.dataset_year = bokeh.models.Div(
            text=f"<b>Год создания:</b> {self.params['db'][self.params['dataset']]['year']}"
        )
        self.dataset_version = bokeh.models.Div(
            text=f"<b>Версия:</b> {self.params['db'][self.params['dataset']]['version']}"
            )

        self.dataset_categories_selector = bokeh.models.MultiChoice(
            value=[],
            options=list(self.params['db'][self.params['dataset']]['categories'].keys())
        )
        self.dataset_categories_selector.on_change('value', changeDatasetCategoriesCallback)

        self.dataset_info = pn.Column(
            self.dataset_description,
            self.dataset_url,
            self.dataset_contributor,
            self.dataset_year,
            self.dataset_version,
            bokeh.models.Div(text="<b>Категории изображений:</b>"),
            self.dataset_categories_selector
        )

        self.dataset_load_button=pn.widgets.Button(name='Добавить датасет', align='start', width=120, button_type='primary')
        self.dataset_load_button.on_click(self.on_click_dataset_load)

        self.dataset_apply_button=pn.widgets.Button(name='Применить', align='end', width=100, button_type='primary', visible=False)
        self.dataset_apply_button.on_click(self.on_click_dataset_apply)

        self.db_interface = pn.Column(
            '# База данных изображений',
            pn.Row(self.dataset_selector, self.dataset_info, margin=(-5,5,15,5)),
            pn.Row(self.dataset_load_button, self.dataset_apply_button)
        )
        
        def changeTaskParamCallback():
            self.task_apply_button.visible = True

        for widget in self.group_params_widgets('Task', changeTaskParamCallback):
            setattr(self, f"{widget.name}_selector", widget)

        self.task_apply_button=pn.widgets.Button(name='Применить', align='start', width=100, button_type='primary')
        self.task_apply_button.on_click(self.on_click_task_apply)

        self.task_interface = pn.Column(
            '# Задача анализа изображений',
            self.task_category_selector,
            self.task_type_selector,
            self.task_objects_selector,
            self.task_goal_selector,
            self.task_goal_value_selector,
            self.task_apply_button
        )

        self.checkbox = pn.widgets.Checkbox(name='Подобрать готовые модели')

        self.next_button=pn.widgets.Button(name='Далее', align='end', width=100, button_type='primary')
        self.next_button.on_click(self.on_click_next)

    def on_click_dataset_load(self, event):
        self.next_window = 'DatasetLoader'
        self.close()

    def on_click_dataset_apply(self, event):
        ds = self.params['dataset']
        categories = set(self.dataset_categories_selector.value)
        for category in self.params['db'][ds]['categories']:
            self.params['db'][ds]['categories'][category]['select'] = category in categories

        task_objects = set({})
        for ds in self.params['db']:
            for category in self.params['db'][ds]['categories']:
                if self.params['db'][ds]['categories'][category]['select']:
                    task_objects.add(category)
        self.task_objects_selector.options=list(task_objects)

        self.dataset_apply_button.visible = False

    def on_click_task_apply(self, event):
        # CORE:  
        self.params['task'] = Task(
            self.params['task_category'], 
            self.params['task_type'],
            self.params['task_objects'], 
            goal={self.params['task_goal']: self.params['task_goal_value']})
        self.task_apply_button.visible = False

    def on_click_next(self, event):
        if self.checkbox.value:
            self.next_window = 'TrainedModels'
        self.close()

    def panel(self):
        return pn.Column(
            self.db_interface,
            pn.Spacer(height=10),
            self.task_interface,
            pn.Spacer(height=10),
            self.checkbox,
            self.next_button,
            margin=(0,0,0,10))



class DatasetLoader(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_back(self, event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню загрузки датасета',
            pn.Spacer(height=10),
            self.back_button)



class Params(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.params_widgets = [
            ("Параметры обучения", self.group_params_widgets('Learning', lambda: None)),
            ("Параметры оптимизатора", self.group_params_widgets('Optimizer', lambda: None))
        ]

        def to_column(widgets):
            return bokeh.models.Column( bokeh.models.Spacer(height=10), *widgets,
                sizing_mode="stretch_width", height=500, height_policy="fixed", css_classes=['scrollable'])

        self.tabs = bokeh.models.Tabs(
            tabs=[ bokeh.models.Panel(title=title, child=to_column(widgets)) for title, widgets in self.params_widgets ])

        def panelActive(attr, old, new):
            if self.tabs.active == 1:
                for widget in self.tabs.tabs[self.tabs.active].child.children:
                    widget.visible = self.is_param_widget_visible(widget)
        
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
        ('DatasetLoader', DatasetLoader),
        ('Params', Params),
        ('TrainedModels', TrainedModels)
    ],
    graph={
        'Start': ('DatasetLoader', 'Params', 'TrainedModels'),
        'DatasetLoader': 'Start',
        'Params': 'Start',
        'TrainedModels': 'Start'
    },
    root='Start',
    ready_parameter='ready', 
    next_parameter='next_window',
    auto_advance=True
)


interface = pn.template.MaterialTemplate(
    title="Ann Automl App",
    sidebar=[pn.pane.Markdown("## Settings")],
    main=[
        pipeline.stage, 
        # pn.layout.Divider(margin=(50, 0, 50, 0)), pn.Row(pipeline.network, pipeline.buttons)
    ]
)