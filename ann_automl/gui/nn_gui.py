import os
import sys
import time
import traceback

import panel as pn
import param
import pandas as pd
import datetime as dt
import bokeh
from typing import Any, Callable
from bokeh.models import CustomJS, Div, Row, Column, Button, Select, Slider,\
                         MultiChoice, MultiSelect, CheckboxGroup, CheckboxButtonGroup, \
                         DatePicker, TextInput, TextAreaInput, Spacer, Tabs, Panel
import bokeh.plotting.figure as Figure

from ..utils.process import process
from .params import hyperparameters
from ann_automl.gui.transition import Transition
import ann_automl.gui.tensorboard as tb
from ..core.nn_solver import loss_target, metric_target, NNTask, recommend_hparams
from ..core.nnfuncs import cur_db, StopFlag, train, param_values
from ..core import nn_rules_simplified

Callback = Callable[[Any, Any, Any], None]

# Launch TensorBoard
tb.start("--logdir ./logs --host 0.0.0.0 --port 6006")

css = '''
.bk.panel-widget-box {
    background: #fafafa;
    border-radius: 0px;
    border: none;
    box-shadow: 0 1px 5px grey;
    padding: 15px 15px;
    overflow: auto !important;
}
'''

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
            for db in [cur_db().get_all_datasets_info(full_info=True)] for ds in db.values()
        },
        'title': 'База данных'
    },
    'dataset': {
        'default': None,
        'title': 'Текущий датасет'
    },
    'selected_datasets': {
        'default': [],
        'title': 'Выделенный список датасетов',
        'gui': {
            'widget': 'MultiChoice',
            'info': True
        }
    },
    'task_category': {
        'values': ['train', 'test', 'database', 'history'],
        'default': 'train',
        'title': 'Категория задачи',
        'gui': {
            'group': 'Task',
            'widget': 'Select'
        }
    },
    'task_type': {
        'values': ['classification', 'segmentation', 'detection'],
        'default': 'classification',
        'title': 'Тип задачи',
        'gui': {
            'group': 'Task',
            'widget': 'Select',
            'info': True
        }
    },
    'task_objects': {
        'values': [],
        'default': [],
        'title': 'Категории изображений',
        'gui': {
            'group': 'Task',
            'widget': 'MultiChoice',
            'info': True
        }
    },
    'task_target_func': {
        'values': ['Метрика обучения'],
        'default': 'Метрика обучения',
        'title': 'Целевой функционал',
        'gui': {
            'group': 'Task',
            'widget': 'Select',
            'info': True
        }
    },
    'task_target_value': {
        'type': 'float',
        'range': [0, 1], 
        'default': 0.9, 
        'step': 0.01, 
        'scale': 'lin',
        'title': 'Значение целевого функционала',
        'gui': {
            'group': 'Task',
            'widget': 'Slider',
            'info': True
        }
    },
    'task_maximize_target': {
        'type': 'bool',
        'default': True,
        'title': 'Оптимизировать целевой функционал',
        'gui': {
            'group': 'Task',
            'widget': 'Checkbox',
            'info': True
        }
    },
    'tune': {
        'type': 'bool',
        'default': False,
        'title': 'Оптимизировать гиперпараметры',
        'gui': {
            'group': 'General',
            'widget': 'Checkbox',
            'info': True
        }
    },
}

gui_params.update({k: v for k, v in hyperparameters.items() if 'gui' in v})


class Window(param.Parameterized):

    params = param.Dict({p: gui_params[p]['default'] for p in gui_params})
    ready = param.Boolean(False)
    next_window = param.Selector(
        objects=['Database', 'DatasetLoader', 'Task', 'Params', 'Training', 'History'])
    prev_window = param.Selector(
        objects=['Database', 'DatasetLoader', 'Task', 'Params', 'Training', 'History'])
    logs = param.String()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # print(f"{self.__class__.__name__}.__init__ called")

    def close(self):
        self.prev_window = self.__class__.__name__
        print(f"close {self.prev_window}")
        self.ready=True

    def on_click_back(self, event):
        print(f"back to {self.prev_window}")
        self.next_window = self.prev_window
        self.close()

    def _params_widgets(self, param_widget_maker, group: str, *args, **kwargs):
        print("_params_widgets >", self, param_widget_maker, group, *args, **kwargs)
        for par, desc in gui_params.items():
            if 'gui' in desc and (group == '' or desc['gui'].get('group', '') == group):
                yield param_widget_maker(self, par, *args, **kwargs)

    def param_widget_info(self, name: str):
        print("param_widget_info >", self, name)
        
        desc = gui_params[name]
        info = ""
        if desc['gui']['widget'] == 'Select':
            info = self.params[name]

        elif desc['gui']['widget'] == 'MultiChoice':
            info = ', '.join(self.params[name])

        elif desc['gui']['widget'] == 'Slider':
            info = str(self.params[name])

        elif desc['gui']['widget'] == 'Checkbox':
            info = 'Да' if self.params[name] else 'Нет'

        else:
            raise ValueError(f'Unsupported widget type {desc["gui"]["widget"]}')

        return Row(
            Div(text=f"{desc['title']}:",
                min_height=20, sizing_mode='stretch_height', width=250, align='center', margin=(0, 10, 0, 0)),
            Div(text=f"<b>{info}</b>",
                min_height=20, sizing_mode='stretch_height', width=150, align='center', margin=(0, 0, 0, 0)),
            margin=(15, 30, 0, 15)
        )

    def params_widget_infos(self, group: str = ''):
        print("params_widget_infos >", self, group)
        return self._params_widgets(self.param_widget_info.__func__, group)

    def param_widget_setter(self, name: str, change_callback: Callback):
        print("param_widget_setter >", self, name, change_callback)

        def change_value(attr, old, new):
            self.params[name] = new
            change_callback(attr, old, new)

        def change_active_value(attr, old, new):
            self.params[name] = name in new
            change_callback(attr, old, new)
        
        desc = gui_params[name]
        kwargs = {
            'name': name,
            'margin': (5, 10, 5, 10)
        }

        if desc['gui']['widget'] == 'Select':
            widget = Select(**kwargs, title=desc['title'],
                value=self.params[name], options=[x for x in desc['values']])
            widget.on_change('value', change_value)

        elif desc['gui']['widget'] == 'MultiChoice':
            widget = MultiChoice(**kwargs, title=desc['title'],
                value=self.params[name], options=[x for x in desc['values']])
            widget.on_change('value', change_value)

        elif desc['gui']['widget'] == 'Slider':

            str_values, cur_index = param_values(return_str=True,
                **{**desc, 'default': self.params[name]})
            values, _ = param_values(**desc)

            try:
                formatter = bokeh.models.FuncTickFormatter(
                    code=f"const labels = {str_values};\nreturn labels[tick];")
            except:
                traceback.print_exc()
                raise

            def change_slider_value(attr, old, new):
                old = values[max(0,min(old,len(values)-1))]
                new = values[max(0,min(new,len(values)-1))]
                self.params[name] = new
                change_callback(attr, old, new)

            widget = Slider(**kwargs, title=desc['title'], value=cur_index,
                start=0, end=len(values)-1, step=1, format=formatter)
            widget.on_change('value', change_slider_value)

        elif desc['gui']['widget'] == 'Checkbox':
            widget = CheckboxGroup(**kwargs, labels=[desc['title']],
                active=[0] if self.params[name] else [])
            widget.on_change('active', change_active_value)

        else:
            raise ValueError(f'Unsupported widget type {desc["gui"]["widget"]}')

        return widget

    def params_widget_setters(self,
            group: str = '',
            change_callback: Callback = lambda attr, old, new: None):
        print("params_widget_setters >", self, group, change_callback)
        return self._params_widgets(self.param_widget_setter.__func__,
                                    group, change_callback)

    def is_param_widget_visible(self, widget):
        return (widget.name not in gui_params or
                'cond' not in gui_params[widget.name] or 
                all (self.params[p] in v for p, v in gui_params[widget.name]['cond']))


class Database(Window):

    def __init__(self, **params):
        super().__init__(**params)

        def changeDatasetCallback(attr, old, new):
            new_ds = new[0]
            self.init_dataset_info_interface(new_ds)
            self.dataset_info.visible = True
            self.dataset_apply_button.disabled = False

        self.dataset_selector = MultiSelect(
            options=list(self.params['db'].keys()),  
            min_width=120, width_policy="max", height_policy="max", margin=(5,15,5,5)
        )
        self.dataset_selector.on_change('value', changeDatasetCallback)

        def changeDatasetSupercategoriesCallback(attr, old, new):
            ds = self.params['dataset']
            supercategory = new
            categories = self.get_dataset_categories(ds, supercategory)
            self.dataset_categories_selector.options = categories
            self.dataset_categories_selector.value = categories[0]
            self.dataset_categories_info.text = \
                self.get_dataset_category_info(ds, supercategory, categories[0])

        def changeDatasetCategoriesCallback(attr, old, new):
            ds = self.params['dataset']
            supercategory = self.dataset_supercategories_selector.value
            category = new
            self.dataset_categories_info.text = \
                self.get_dataset_category_info(ds, supercategory, category)

        self.dataset_description = Div()
        self.dataset_url = Div()
        self.dataset_contributor = Div()
        self.dataset_date = Div()
        self.dataset_version = Div()
        self.dataset_supercategories_selector = \
            Select(width=250, width_policy='fixed')
        self.dataset_supercategories_selector.on_change('value', 
            changeDatasetSupercategoriesCallback)
        self.dataset_categories_selector = \
            Select(width=250, width_policy='fixed')
        self.dataset_categories_selector.on_change('value', 
            changeDatasetCategoriesCallback)
        self.dataset_categories_info = Div(align='center', min_width=150)

        ds = None
        if len(self.dataset_selector.options) > 0:
            ds = self.dataset_selector.options[0]
            self.init_dataset_info_interface(ds)

        self.dataset_info = Column(
            self.dataset_description,
            self.dataset_url,
            self.dataset_contributor,
            self.dataset_date,
            self.dataset_version,
            Row(
                Div(text="<b>Категории изображений:</b>", min_width=160),
                self.dataset_supercategories_selector,
                self.dataset_categories_selector,
                self.dataset_categories_info
            ),
            visible=ds is not None
        )

        self.selected_datasets = Div(
            text = "<b>Используемые датасеты:</b> " +
                f"{', '.join(self.params['selected_datasets'])}",
            visible=ds is not None, margin=(5,5,10,10))

        self.dataset_load_button=Button(
            label='Добавить датасет', button_type='primary', width=120)
        self.dataset_load_button.on_click(self.on_click_dataset_load)

        self.dataset_apply_button=Button(
            label='Использовать выбранные датасеты', button_type='primary', 
            width=220, disabled=True)
        self.dataset_apply_button.on_click(self.on_click_dataset_apply)

        self.next_button=Button(
            label='Далее', button_type='primary', 
            width=70, disabled=len(self.params['selected_datasets']) == 0)
        self.next_button.on_click(self.on_click_next)

    def get_dataset_supercategories(self, ds):
        return list(self.params['db'][ds]['categories'].keys())

    def get_dataset_categories(self, ds, supercategory):
        return list(self.params['db'][ds]['categories'][supercategory].keys())

    def get_dataset_category_info(self, ds, supercategory, category):
        n = int(self.params['db'][ds]['categories'][supercategory][category])
        suf = "изображений" if n % 10 in {0,5,6,7,8,9} or n in {11,12,13,14} else \
              "изображения" if n % 10 in {2,3,4} else \
              "изображение"
        return f"{str(n)} {suf}"

    def init_dataset_info_interface(self, ds):
        description = ds
        url = self.params['db'][ds]['url']
        contributor = self.params['db'][ds]['contributor']
        data = self.params['db'][ds]['date_created']
        version = self.params['db'][ds]['version']
        supercategories = self.get_dataset_supercategories(ds)
        categories = self.get_dataset_categories(ds, supercategories[0])

        self.params['dataset'] = ds
        self.dataset_description.text=f"<b>Название:</b> {description}"
        self.dataset_url.text=f"<b>Источник:</b> <a href=\"{url}\">{url}</a>"
        self.dataset_contributor.text=f"<b>Создатель:</b> {contributor}"
        self.dataset_date.text=f"<b>Дата создания:</b> {data}"
        self.dataset_version.text=f"<b>Версия:</b> {version}"
        self.dataset_supercategories_selector.options = supercategories
        self.dataset_supercategories_selector.value = supercategories[0]
        self.dataset_categories_selector.options = categories
        self.dataset_categories_selector.value = categories[0]
        self.dataset_categories_info.text = \
            self.get_dataset_category_info(ds, supercategories[0], categories[0])

    def on_click_dataset_load(self, event):
        self.next_window = 'DatasetLoader'
        self.close()

    def on_click_dataset_apply(self, event):
        self.params['selected_datasets'] = self.dataset_selector.value
        self.selected_datasets.text = \
            f"<b>Используемые датасеты:</b> {', '.join(self.params['selected_datasets'])}"

        gui_params['task_objects']['values'] = list({ 
            category for ds in self.dataset_selector.value
                     for supercategory in self.params['db'][ds]['categories']
                     for category in self.params['db'][ds]['categories'][supercategory]
        })
        print("ds_dict = ", self.params['db'][self.dataset_selector.value[0]])
        cur_db().ds_filter = list(self.dataset_selector.value)
        self.dataset_apply_button.disabled = True
        self.next_button.disabled = False

    def on_click_next(self, event):
        self.next_window = 'Task'
        self.close()

    def panel(self):
        return pn.Column(
            '# База данных изображений',
            Div(text="<b>Доступные датасеты:</b>", margin=(-10,0,0,10)),
            Row(self.dataset_selector, self.dataset_info, margin=(0,5,5,5)),
            self.selected_datasets,
            Row(
                self.dataset_load_button, 
                self.dataset_apply_button, 
                self.next_button,
                spacing=5
            )
        )


class DatasetLoader(Window):

    def __init__(self, **params):
        super().__init__(**params)

        self.dataset_checker = Div(align="center")

        self.dataset_description_setter = TextInput(
            title="Название:", placeholder="Введите название датасета")
        self.dataset_url_setter = TextInput(
            title="Источник:", placeholder="Введите url датасета")
        self.dataset_contributor_setter = TextInput(
            title="Создатель:", placeholder="Введите разработчиков датасета")
        self.dataset_date_setter = DatePicker(
            title='Дата создания:')
        self.dataset_version_setter = TextInput(
            title="Версия:", placeholder="Введите версию датасета")
        self.anno_file_setter = TextInput(
            title="Файл с аннотациями:", placeholder="Путь до файла в формате json")
        self.dataset_dir_setter = TextInput(
            title="Каталог с изображениями:", placeholder="Путь до каталога")

        self.apply_button=Button(
            label='Загрузить', button_type='primary', width=100)
        self.apply_button.on_click(self.on_click_apply)
        
        self.back_button=Button(
            label='Назад', button_type='primary', width=80)
        self.back_button.on_click(self.on_click_back)

        self.error_message = Div(text="", visible=False)

    def on_click_apply(self, event):
        err = ""
        if not self.dataset_description_setter.value:
            err += "Название датасета не может быть пустым.\n"
        if not self.anno_file_setter.value:
            err += "Не выбран файл с аннотациями.\n"
        elif not os.path.exists(self.anno_file_setter.value):
            err += "Файл с аннотациями не найден\n"
        elif not self.anno_file_setter.value.endswith('.json'):
            err += "Файл с аннотациями должен быть в формате json\n"
        if not self.dataset_dir_setter.value:
            err += "Не указан каталог с изображениями.\n"
        elif not os.path.exists(self.dataset_dir_setter.value):
            err += "Каталог с изображениями не найден\n"

        if err:
            self.error_message.text = '<br>'.join(err.split('\n'))
            self.error_message.visible = True
            self.dataset_checker.text = \
                '<font color=red>Не удалось загрузить датасет</font>'
            return
            
        self.error_message.text = ""
        self.error_message.visible = False
        self.dataset_checker.text = ""

        try:
            cur_db().fill_in_coco_format(
                self.anno_file_setter.value,
                self.dataset_dir_setter.value,
                ds_info={
                        "description": self.dataset_description_setter.value,
                        "url": self.dataset_url_setter.value,
                        "version": self.dataset_version_setter.value,
                        "year": self.dataset_year_setter.value,
                        "contributor": self.dataset_contributor_setter.value,
                        "date_created": self.dataset_year_setter.value
                    }
            )
            self.params['db'] = {
                ds['description'] : ds
                for db in [cur_db().get_all_datasets_info(full_info=True)]
                for ds in db.values()
            }
            self.close()

        except Exception as e:
            # format exception
            stack = traceback.format_exc()
            self.error_message.value = '<br>'.join(stack.split('\n') + [str(e)])
            self.error_message.visible = True

    def panel(self):
        return pn.Column(
            pn.pane.Markdown('# Меню загрузки датасета', margin=(5, 5, -10, 5)),
            self.dataset_description_setter,
            self.dataset_url_setter,
            self.dataset_contributor_setter,
            self.dataset_date_setter,
            self.dataset_version_setter,
            self.anno_file_setter,
            self.dataset_dir_setter,
            Row(
                self.back_button, self.apply_button, self.dataset_checker,
                margin=(15, 5, 15, 5)
            ),
            self.error_message
        )


class Task(Window):

    def __init__(self, **params):
        super().__init__(**params)
        
        def changeTaskParamCallback(attr, old, new):
            self.apply_button.disabled = False

        print("==========================================================")

        for widget in self.params_widget_setters('Task', changeTaskParamCallback):
            print("%", widget)
            setattr(self, f"{widget.name}_selector", widget)

        print("==========================================================")

        def changeTaskObjects(attr, old, new):
            changeTaskParamCallback(attr, old, new)
            self.task_objects_checker.text = ""
        self.task_objects_selector.on_change('value', changeTaskObjects)

        self.task_interface = pn.Column(
            '# Задача анализа изображений',
            self.task_category_selector,
            self.task_type_selector,
            self.task_objects_selector,
            self.task_target_func_selector,
            self.task_target_value_selector,
            self.task_maximize_target_selector
        )
        
        self.task_objects_checker = Div(align="center", margin=(5, 5, 5, 25))

        self.checkbox = CheckboxGroup(labels=['Подобрать готовые модели'])

        self.apply_button=Button(
            label='Создать задачу', button_type='primary', 
            width=100, disabled=self.params['task'] is not None)
        self.apply_button.on_click(self.on_click_apply)

        self.next_button=Button(
            label='Далее', button_type='primary', 
            width=100, disabled=self.params['task'] is None)
        self.next_button.on_click(self.on_click_next)

        self.back_button=Button(
            label='Назад', button_type='primary', width=100)
        self.back_button.on_click(self.on_click_back)

    def on_click_apply(self, event):

        if len(self.task_objects_selector.value) < 2:
            self.task_objects_checker.text = \
                '<font color=red>Выберите не менее двух категорий изображений</font>'

        else:
            # CORE:
            self.params['task'] = NNTask(
                category=self.params['task_category'],
                type=self.params['task_type'],
                objects=self.params['task_objects'],
                func={
                    'Метрика обучения': metric_target
                }[self.params['task_target_func']],
                target=self.params['task_target_value'],
                goals={
                    'maximize': self.params['task_maximize_target']
                }
            )
            hparams = recommend_hparams(self.params['task'], trace_solution=True)
            self.params['recommended_hparams'] = hparams
            for k, v in hparams.items():
                key = 'train.' + k
                self.params[key] = v

            self.apply_button.disabled = True
            self.next_button.disabled = False

    def on_click_next(self, event):
        if len(self.checkbox.active) == 0:
            self.next_window = 'Params'
        else:
            self.next_window = 'History'
        self.close()

    def panel(self):
        return pn.Column(
            '# Задача анализа изображений',
            self.task_category_selector,
            self.task_type_selector,
            self.task_objects_selector,
            self.task_target_func_selector,
            self.task_target_value_selector,
            self.task_maximize_target_selector,
            Row(self.apply_button, self.task_objects_checker),
            pn.Spacer(height=10),
            self.checkbox,
            Row(self.back_button, self.next_button)
        )


class Params(Window):

    def __init__(self, **params):
        super().__init__(**params)

        self.params_widgets = [
            ("Общие параметры", self.params_widget_setters('General')),
            ("Параметры автонастройки", self.params_widget_setters('Tune')),
            ("Параметры обучения", self.params_widget_setters('Learning')),
            ("Параметры оптимизатора", self.params_widget_setters('Optimizer'))
        ]

        def to_column(widgets):
            return Column( Spacer(height=10), *widgets,
                sizing_mode="stretch_width", height=500, height_policy="fixed", css_classes=['scrollable'])

        self.tabs = Tabs(
            tabs=[Panel(title=title, child=to_column(widgets)) for title, widgets in self.params_widgets])

        def panelActive(attr, old, new):
            if self.tabs.active == 3:
                for widget in self.tabs.tabs[self.tabs.active].child.children:
                    widget.visible = self.is_param_widget_visible(widget)

        self.tabs.on_change('active', panelActive)
        
        self.next_button=Button(label='Запустить обучение', width=150, button_type='primary')
        self.next_button.on_click(self.on_click_next)

        self.back_button=Button(label='Назад', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_next(self, event):
        # вызвать train
        self.next_window = 'Training'
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню параметров',
            self.tabs,
            pn.Spacer(height=10),
            Row(self.back_button, self.next_button))


class Training(Window):

    def __init__(self, **params):
        super().__init__(**params)

        self.is_start = self.prev_window == 'Params'

        print("Create params_box ... ", end='', flush=True)
        self.params_box = Column(
            *self.params_widget_infos(),
            height=730, height_policy='fixed', visible=True,
            css_classes=['panel-widget-box'], margin=(10,10,10,10))
        print("ok")

        print("Create output_box ... ", end='', flush=True)
        self.output = TextAreaInput(
            value = self.logs, min_width=500, sizing_mode='stretch_both', disabled=True)

        self.output_box = Column(
            self.output,
            height=730, height_policy='fixed', sizing_mode='stretch_both',
            css_classes=['panel-widget-box'], margin=(10,10,10,10))
        print("ok")

        print("Create tools_box ... ", end='', flush=True)
        self.loss_acc_plot = Figure(
            title='Loss and Accuracy', x_axis_label='Epoch', y_axis_label='Loss/Accuracy',
            plot_width=500, plot_height=250, sizing_mode='stretch_both')
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.last_epoch = 0

        self.tools_box = Column(
            self.loss_acc_plot,
            height=730, height_policy='fixed', sizing_mode='stretch_both',
            css_classes=['panel-widget-box'], margin=(10,10,10,10))
        print("ok")

        print("Create box_button_group ... ", end='', flush=True)
        self.box_button_group = CheckboxButtonGroup(
            labels=['Параметры', 'Журнал сообщений', 'Инструменты'],
            button_type='primary', active=[0,1,2], margin=(5,30,5,5))
        self.box_button_group.on_click(self.on_click_box_button)
        print("ok")

        print("Create tensorboard_button ... ", end='', flush=True)
        self.tensorboard_button=Button(
            label='Tensorboard', align='start', width=100, button_type='primary')
        self.tensorboard_button.js_on_click(
            CustomJS(code='window.open("http://localhost:6006/#scalars");'))
        print("ok")

        self.back_button=Button(
            label='Назад', width=100, button_type='primary',
            disabled=self.is_start)
        self.back_button.on_click(self.on_click_back)

        self.next_button=Button(
            label='История обучения', width=150, button_type='primary',
            disabled=self.is_start)
        self.next_button.on_click(self.on_click_next)

        self.stop_button=Button(
            label='Стоп', width=100, button_type='primary',
            disabled=not self.is_start)
        self.stop_button.on_click(self.on_click_stop)

        self.continue_button=Button(
            label='Продолжить обучение', width=100, button_type='primary',
            visible=not self.is_start)
        self.continue_button.on_click(self.on_click_continue)

        if self.is_start:
            self.start()

    def start(self):
        self.is_stop = False
        self.is_break = False

        # create timer to update bokeh widgets
        self.bokeh_timer = bokeh.io.curdoc().add_periodic_callback(
            self.update_bokeh_server, 1000)

        self.stop = StopFlag()
        hparams = self.params.get('recommended_hparams', {})
        hparams.update({gui_params[k]['param_key']: v for k, v in self.params.items()
                        if gui_params.get(k, {}).get('param_from', '') == 'train'})
        self.process = process(train)(
            nn_task=self.params['task'],
            stop_flag=self.stop, hparams=hparams, start=False)
        self.process.set_handler(
            'print', lambda *args, **kwargs: self.msg(*args, **kwargs))
        self.process.set_handler(
            'train_callback', lambda *args, **kwargs: self.on_train_callback(*args, **kwargs))
        self.process.on_finish = lambda _: self.on_process_finish()
        print('Запуск процесса обучения')
        self.process.start()

    def update_bokeh_server(self, *args, **kwargs):
        self.output.value = self.logs

        # TODO: сделать здесь по-нормальному обновление графиков (через поток данных)
        if len(self.epochs) > self.last_epoch+1:
            ll = len(self.epochs)-1
            # update self.loss_acc_plot
            self.loss_acc_plot.line(
                list(self.epochs[self.last_epoch:]),
                list(self.losses[self.last_epoch:]),
                legend_label='Loss', line_color='red')
            self.loss_acc_plot.line(
                list(self.epochs[self.last_epoch:]),
                list(self.accuracies[self.last_epoch:]),
                legend_label='Accuracy', line_color='green')
            if self.val_losses:
                self.loss_acc_plot.line(
                    list(self.epochs[self.last_epoch:]),
                    list(self.val_losses[self.last_epoch:]),
                    legend_label='Val Loss', line_color='blue')
            if self.val_accuracies:
                self.loss_acc_plot.line(
                    list(self.epochs[self.last_epoch:]),
                    list(self.val_accuracies[self.last_epoch:]),
                    legend_label='Val Accuracy', line_color='black')
            self.last_epoch = ll

        if self.is_stop:
            self.back_button.disabled = False
            self.next_button.disabled = False
            self.stop_button.disabled = True
            self.continue_button.visible = self.is_break
            bokeh.io.curdoc().remove_periodic_callback(self.bokeh_timer)
            self.bokeh_timer = None
            self.is_stop = False
            self.is_break = False

    def add_plot_point(self, epoch, loss, accuracy, val_loss, val_accuracy):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        if val_loss is not None:
            self.val_losses.append(min(5, val_loss))
        if val_accuracy is not None and 0 <= val_accuracy <= 1:
            self.val_accuracies.append(val_accuracy)

    def msg(self, *args, file=None, end='\n', **kwargs):
        if file is None:
            self.logs += ''.join(map(str, args)) + end
        else:
            print(*args, **kwargs, file=file, end=end)

    def on_process_finish(self):
        self.is_stop = True

    def on_train_callback(self, tp, batch=None, epoch=None, logs=None, model=None):
        if tp == 'epoch':
            self.msg(f'Эпоха {epoch}: {logs}')
            self.add_plot_point(epoch, logs['loss'], logs['accuracy'], logs.get('val_loss', None), logs.get('val_accuracy', None))
            time.sleep(0.1)
        elif tp == 'finish':
            self.msg('Обучение завершено')

    def on_click_next(self,event):
        self.next_window = 'History'
        self.close()

    def on_click_box_button(self, active):
        self.params_box.visible = 0 in active
        self.output_box.visible = 1 in active
        self.tools_box.visible = 2 in active

    def on_click_stop(self, event):
        self.stop()
        self.is_break = True
        self.stop_button.disabled = True

    def on_click_continue(self, event):
        self.back_button.disabled = True
        self.next_button.disabled = True
        self.stop_button.disabled = False
        self.continue_button.visible = False
        self.output.value += "\nContinue training ...\n"
        self.start()

    def panel(self):
        return pn.Column(
            '# Меню обучения модели',
            Row(
                self.back_button,
                self.next_button,
                self.box_button_group,
                self.tensorboard_button,
                self.stop_button,
                self.continue_button,
                margin=(-5,5,10,0)
            ),
            Row(
                self.params_box,
                self.output_box,
                self.tools_box,
                height=750, height_policy='fixed', margin=(5,5,5,-5)
            )
        )


class History(Window):

    def __init__(self, **params):
        super().__init__(**params)

        self.models_list = pn.widgets.Select(
            name='Список обученных моделй',
            options=['Модель 1', 'Модель 2', 'Модель 3', 'Модель 4', 'Модель 5'],
            size=31)

        self.next_button=Button(
            label='Далее', align='end', width=100, button_type='primary')
        self.next_button.on_click(self.on_click_next)

        self.back_button=Button(
            label='Назад', align='start', width=100, button_type='primary')
        self.back_button.on_click(self.on_click_back)

    def on_click_next(self, event):
        self.next_window = 'Params'
        self.close()

    def panel(self):
        return pn.Column(
            '# Меню обученных моделей', 
            pn.Row(
                self.models_list,
                pn.WidgetBox('## Описание модели', min_width=500, height=500)
            ),
            Row(self.back_button, self.next_button)
        )


pipeline = Transition(
    stages=[
        ('Database', Database),
        ('DatasetLoader', DatasetLoader),
        ('Task', Task),
        ('Params', Params),
        ('Training', Training),
        ('History', History)
    ],
    graph={
        'Database': ('DatasetLoader', 'Task'),
        'DatasetLoader': 'Database',
        'Task': ('Database', 'Params', 'History'),
        'Params': ('Task', 'Training', 'History'),
        'Training': ('Params', 'History'),
        'History': ('Task', 'Training', 'Params')
    },
    root='Database',
    ready_parameter='ready', 
    next_parameter='next_window',
    auto_advance=True
)


interface = pn.template.MaterialTemplate(
    title="Ann Automl",
    sidebar=[pn.pane.Markdown("## Settings")],
    main=[
        pipeline.stage, 
        # pn.layout.Divider(margin=(50, 0, 50, 0)), pn.Row(pipeline.network, pipeline.buttons)
    ],
    modal=[pn.pane.Markdown("## Modal")],
    header_background="#f57c00",
    favicon="https://static.centro.org.uk/img/wmca/favicons/mstile-150x150.png"
)
