import sys

import panel as pn
import param
import pandas as pd
import datetime as dt
import bokeh
from typing import Callable

from ann_automl.core.db_module import DBModule
from ann_automl.core.solver import Task
from ..utils.process import process
from .params import hyperparameters
from ann_automl.gui.transition import Transition
from ..core.nn_solver import NNTask
from ..core.nnfuncs import nnDB as DB, StopFlag, train

css = '''
.bk.panel-widget-box {
    background: #fafafa;
    border-radius: 0px;
    border: 1px solid #dcdcdc;
    overflow: auto !important;
}
'''

# DB = DBModule("sqlite:///../tests.sqlite")

pn.extension(raw_css=[css])
pn.config.sizing_mode = 'stretch_width'

# GUI titles of datasets attributes
datasets_attr_title = {
    'description': 'Описание',
    'url': 'Источник',
    'contributor': 'Создатель',
    'year': 'Год создания',
    'version': 'Версия',
    'categories': 'Категории изображений'
}

gui_params = {
    'task': {
        'default': None,
        'title': 'Задача'
    },
    'db': {
        'default': {
            ds['description'] : ds
            for db in [DB.get_all_datasets_info(full_info=True)] for ds in db.values()
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
    'task_target': {
        'type': 'str',
        'values': ['loss', 'metrics'],
        'default': 'loss',
        'title': 'Целевой функционал',
        'gui': {
            'group': 'Task',
            'widget': 'Select'
        }
    },
    'task_target_value': {
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
    },
    'task_maximize_target': {
        'type': 'bool',
        'default': True,
        'title': 'Максимизировать целевой функционал после достижения значения task_target_value',
        'gui': {
            'group': 'Task',
            'widget': 'Checkbox'
        }
    },
}

gui_params.update({k: v for k, v in hyperparameters.items() if 'gui' in v})


class Window(param.Parameterized):

    params = param.Dict({p: gui_params[p]['default'] for p in gui_params})
    ready = param.Boolean(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def close(self):
        self.ready=True

    def param_widget(self, name: str, change_callback: Callable[[], None]):
        print(name, file=sys.stderr)
        def change_value(attr, old, new):
            self.params[name] = new
            change_callback()

        def change_active_value(attr, old, new):
            self.params[name] = name in new
            change_callback()
        
        desc = gui_params[name]
        kwargs = {
            'name': name,
            'margin': (5, 10, 5, 10)
        }

        if desc['gui']['widget'] == 'Select':
            widget = bokeh.models.Select(**kwargs, title=desc['title'], value=self.params[name], 
                options=[x for x in desc['values']])
            widget.on_change('value', change_value)

        elif desc['gui']['widget'] == 'MultiChoice':
            widget = bokeh.models.MultiChoice(**kwargs, title=desc['title'], value=self.params[name], 
                options=[x for x in desc['values']])
            widget.on_change('value', change_value)

        elif desc['gui']['widget'] == 'Slider':
            widget = bokeh.models.Slider(**kwargs, title=desc['title'], value=self.params[name], 
                start=desc['range'][0], end=desc['range'][1], step=desc['step'])
            widget.on_change('value', change_value)

        elif desc['gui']['widget'] == 'Checkbox':
            widget = bokeh.models.CheckboxGroup(**kwargs, labels=[desc['title']],
                active=[0] if self.params[name] else [])
            widget.on_change('active', change_active_value)

        else:
            raise ValueError(f'Unsupported widget type {desc["gui"]["widget"]}')

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

            self.dataset_description.text=f"<b>Название:</b> {ds}"
            self.dataset_url.text=f"<b>Источник:</b> <a href=\"{self.params['db'][ds]['url']}\">{self.params['db'][ds]['url']}</a>"
            self.dataset_contributor.text=f"<b>Создатель:</b> {self.params['db'][ds]['contributor']}"
            self.dataset_year.text=f"<b>Год создания:</b> {self.params['db'][ds]['year']}"
            self.dataset_version.text=f"<b>Версия:</b> {self.params['db'][ds]['version']}"

            supercategories = list(self.params['db'][ds]['categories'].keys())
            categories = list(self.params['db'][ds]['categories'][supercategories[0]].keys())
            category_number = int(self.params['db'][ds]['categories'][supercategories[0]][categories[0]])
            category_number_suf = "штук" if 5 <= category_number % 10 and category_number % 10 <= 9 or 10 <= category_number and category_number <= 14 else "штуки" if 2 <= category_number % 10 and category_number % 10 <= 4 else "штука"

            self.dataset_supercategories_selector.options = supercategories
            self.dataset_supercategories_selector.value = supercategories[0]
            self.dataset_categories_selector.options = categories
            self.dataset_categories_selector.value = categories[0]
            self.dataset_categorie_number.text = f"{str(category_number)} {category_number_suf}"

            # self.is_need_dataset_apply_button_visible = False
            # self.dataset_categories_selector.value=[category for category in self.params['db'][ds]['categories'] if self.params['db'][ds]['categories'][category]['select']]
            # self.is_need_dataset_apply_button_visible = True

        self.dataset_selector = bokeh.models.MultiSelect(
            value=["Dogs vs Cats"], 
            options=list(self.params['db'].keys()),  
            max_width=450, width_policy='min', height_policy="max", margin=(5,15,5,5)
        )
        self.dataset_selector.on_change('value', changeDatasetCallback)

        def changeDatasetSupercategoriesCallback(attr, old, new):
            ds = self.params['dataset']
            categories = list(self.params['db'][ds]['categories'][new].keys())
            category_number = int(self.params['db'][ds]['categories'][new][categories[0]])
            category_number_suf = "штук" if 5 <= category_number % 10 and category_number % 10 <= 9 or 10 <= category_number and category_number <= 14 else "штуки" if 2 <= category_number % 10 and category_number % 10 <= 4 else "штука"
            self.dataset_categories_selector.options = categories
            self.dataset_categorie_number.text = f"{str(category_number)} {category_number_suf}"
            # if self.is_need_dataset_apply_button_visible:
            #     self.dataset_apply_button.visible = True

        def changeDatasetCategoriesCallback(attr, old, new):
            ds = self.params['dataset']
            category_number = int(self.params['db'][ds]['categories'][self.dataset_supercategories_selector.value][new])
            category_number_suf = "штук" if 5 <= category_number % 10 and category_number % 10 <= 9 or 10 <= category_number and category_number <= 14 else "штуки" if 2 <= category_number % 10 and category_number % 10 <= 4 else "штука"
            self.dataset_categorie_number.text = f"{str(category_number)} {category_number_suf}"
            # if self.is_need_dataset_apply_button_visible:
            #     self.dataset_apply_button.visible = True

        ds = self.params['dataset']

        self.dataset_description = bokeh.models.Div(
            text=f"<b>Описание:</b> {ds}"
        )
        self.dataset_url = bokeh.models.Div(
            text=f"<b>Источник:</b> <a href=\"{self.params['db'][ds]['url']}\">{self.params['db'][ds]['url']}</a>"
        )
        self.dataset_contributor = bokeh.models.Div(
            text=f"<b>Создатель:</b> {self.params['db'][ds]['contributor']}"
        )
        self.dataset_year = bokeh.models.Div(
            text=f"<b>Год создания:</b> {self.params['db'][ds]['year']}"
        )
        self.dataset_version = bokeh.models.Div(
            text=f"<b>Версия:</b> {self.params['db'][ds]['version']}"
        )

        supercategories = list(self.params['db'][ds]['categories'].keys())
        self.dataset_supercategories_selector = bokeh.models.Select(
            options=supercategories, value=supercategories[0], width=450, width_policy='fixed'
        )
        self.dataset_supercategories_selector.on_change('value', changeDatasetSupercategoriesCallback)

        categories = list(self.params['db'][ds]['categories'][supercategories[0]].keys())
        self.dataset_categories_selector = bokeh.models.Select(
            options=categories, value=categories[0], width=450, width_policy='fixed'
        )
        self.dataset_categories_selector.on_change('value', changeDatasetCategoriesCallback)

        category_number = int(self.params['db'][ds]['categories'][supercategories[0]][categories[0]])
        category_number_suf = "штук" if category_number % 10 == 0 or 5 <= category_number % 10 and category_number % 10 <= 9 or 10 <= category_number and category_number <= 14 else "штуки" if 2 <= category_number % 10 and category_number % 10 <= 4 else "штука"
        self.dataset_categorie_number = bokeh.models.Div(
            text=f"{str(category_number)} {category_number_suf}",
            align='center'
        )

        self.dataset_info = pn.Column(
            self.dataset_description,
            self.dataset_url,
            self.dataset_contributor,
            self.dataset_year,
            self.dataset_version,
            pn.Row(
                bokeh.models.Div(text="<b>Категории изображений:</b>", min_width=160),
                self.dataset_supercategories_selector,
                self.dataset_categories_selector,
                self.dataset_categorie_number
            )
        )

        self.dataset_load_button=pn.widgets.Button(name='Добавить датасет', align='start', width=120, button_type='primary')
        self.dataset_load_button.on_click(self.on_click_dataset_load)

        self.dataset_apply_button=pn.widgets.Button(name='Применить', align='end', width=100, button_type='primary')
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
            self.task_target_selector,
            self.task_target_value_selector,
            self.task_apply_button
        )

        self.checkbox = pn.widgets.Checkbox(name='Подобрать готовые модели')

        self.next_button=pn.widgets.Button(name='Далее', align='end', width=100, button_type='primary')
        self.next_button.on_click(self.on_click_next)

    def on_click_dataset_load(self, event):
        self.next_window = 'DatasetLoader'
        self.close()

    def on_click_dataset_apply(self, event):
        # ds = self.params['dataset']
        # categories = set(self.dataset_categories_selector.value)
        # for category in self.params['db'][ds]['categories']:
        #     self.params['db'][ds]['categories'][category]['select'] = category in categories

        # task_objects = set({})
        # for ds in self.params['db']:
        #     for category in self.params['db'][ds]['categories']:
        #         if self.params['db'][ds]['categories'][category]['select']:
        #             task_objects.add(category)
        self.task_objects_selector.options=[c for ds in self.dataset_selector.value
                                            for sc in self.params['db'][ds]['categories']
                                            for c in self.params['db'][ds]['categories'][sc]]

        with open('click_logs.txt', 'a') as f:
            f.write(f"{self.task_objects_selector.options}")


    def on_click_task_apply(self, event):
        # CORE:  
        self.params['task'] = NNTask(
            task_ct=self.params['task_category'],
            task_type=self.params['task_type'],
            objects=self.params['task_objects'],
            metric=self.params['task_target'],
            target=self.params['task_target_value'],
            goals={'maximize': self.params['task_maximize_target']}
            # goal={self.params['task_goal']: self.params['task_goal_value']}
            )
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
            margin=(0, 0, 0, 10))


class DatasetLoader(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.dataset_description_setter = bokeh.models.TextInput(
            title="Название:",
            placeholder="Введите название датасета",
            margin=(0, 10, 15, 10)
        )

        self.dataset_url_setter = bokeh.models.TextInput(
            title="Источник:",
            placeholder="Введите url датасета",
            margin=(0, 10, 15, 10)
        )

        self.dataset_contributor_setter = bokeh.models.TextInput(
            title="Создатель:",
            placeholder="Введите разработчиков датасета",
            margin=(0, 10, 15, 10)
        )

        self.dataset_year_setter = bokeh.models.DatePicker(
            title='Дата создания:',
            margin=(0, 10, 15, 10)
        )

        self.dataset_version_setter = bokeh.models.TextInput(
            title="Версия:",
            placeholder="Введите версию датасета",
            margin=(0, 10, 15, 10)
        )

        self.anno_file_setter = pn.Column(
            bokeh.models.Div(
                text="Файл с аннотациями:",
                margin=(0, 10, 0, 10)
            ),
            bokeh.models.FileInput(
                accept = '.json',
                margin=(0, 10, 15, 10)
            )
        )

        self.dataset_dir_setter = bokeh.models.TextInput(
            title="Каталог с изображениями:",
            placeholder="Путь до каталога с изображениями",
            margin=(0, 10, 15, 10)
        )

        self.apply_button=pn.widgets.Button(name='Применить', align='end', width=100, button_type='primary')
        self.apply_button.on_click(self.on_click_apply)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_apply(self, event):
        # DB.fill_coco(
        #     self.anno_file_setter.filename,
        #     self.dataset_dir_setter.value,
        #     ds_info={
        #         "description": self.dataset_description_setter.value,
        #         "url": self.dataset_url_setter.value,
        #         "version": self.dataset_version_setter.value,
        #         "year": self.dataset_year_setter.value,
        #         "contributor": self.dataset_contributor_setter.value,
        #         "date_created": self.dataset_year_setter.value
        #     }
        # )
        self.params['db'] = {
            ds['description'] : ds
            for db in [DB.get_all_datasets_info(full_info=True)] for ds in db.values()
        }
        self.close()

    def on_click_back(self, event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню загрузки датасета',
            self.dataset_description_setter,
            self.dataset_url_setter,
            self.dataset_contributor_setter,
            self.dataset_year_setter,
            self.dataset_version_setter,
            self.anno_file_setter,
            self.dataset_dir_setter,
            pn.Spacer(height=10),
            pn.Row(self.back_button, self.apply_button))



class Params(Window):

    next_window = param.Selector(default='Start', objects=['Start', 'Training'])

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
        self.next_button.on_click(self.on_click_next)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

        self.qq = pn.widgets.StaticText(name='Static Text', value=self.params['task_category'])

    def on_click_next(self, event):
        # вызвать train
        self.params['stop_flag'] = stop_flag = StopFlag()
        hparams = {gui_params[k]['param_key']: v for k, v in self.params.items()
                   if gui_params.get(k, {}).get('param_from', '') == 'train'}
        self.params['process_id'] = p = process(train)(stop_flag=stop_flag, hparams=hparams, start=False)
        self.next_window = 'Training'
        self.close()

    def on_click_back(self, event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню параметров',
            self.tabs,
            pn.Spacer(height=10),
            pn.Row(self.back_button, self.next_button))



class Training(Window):

    next_window = param.Selector(default='Start', objects=['Start'])

    def __init__(self, **params):
        super().__init__(**params)

        self.stop_button=pn.widgets.Button(name='Стоп', align='end', width=100, button_type='primary')
        self.stop_button.on_click(self.on_click_stop)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary', disabled=True)
        self.back_button.on_click(self.on_click_back)

    def on_click_stop(self, event):
        self.params['stop_flag']()
        self.back_button.disabled = False
        pass

    def on_click_back(self,event):
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню обучения модели',
            pn.WidgetBox('### Output', min_width=500, height=500),
            pn.Row(self.back_button, self.stop_button)
        )




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
        ('Training', Training),
        ('TrainedModels', TrainedModels)
    ],
    graph={
        'Start': ('DatasetLoader', 'Params', 'TrainedModels'),
        'DatasetLoader': 'Start',
        'Params': ('Start', 'Training'),
        'Training': 'Start',
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
