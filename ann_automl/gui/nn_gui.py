import os
import sys
import time
import traceback
import json

import panel as pn
import param
import pandas as pd
import datetime as dt
import numpy as np
import bokeh
from typing import Any, Callable, Dict, Optional
from bokeh.models import CustomJS, Div, Row, Column, Select, Slider, RadioGroup,\
                         MultiChoice, MultiSelect, CheckboxGroup, CheckboxButtonGroup, \
                         DatePicker, TextInput, TextAreaInput, Spacer, \
                         ColumnDataSource, DataTable, TableColumn, Dropdown, \
                         NumberFormatter, Widget
from bokeh.events import ButtonClick
import bokeh.plotting.figure as Figure
from datetime import date
from random import randint, random, sample

from ..utils.process import process
from .params import hyperparameters, widget_type
import ann_automl.gui.tensorboard as tensorboard
import ann_automl.gui.qsl_label as qsl_label
import ann_automl.core.smart_labeling as labeling
from ..core.nn_task import loss_target, metric_target, NNTask
from ..core.nnfuncs import cur_db, StopFlag, train, tune, param_values, tensorboard_logdir, params_from_history
from ..core.nn_recommend import recommend_hparams

Callback = Callable[[Any, Any, Any], None]
Params = Optional[Dict[str, Any]]

HOST = "0.0.0.0"
#HOST = "localhost"
PORT_QSL = 8080

# Launch TensorBoard
tensorboard.start("--logdir {logdir} --host {host} --port {port}".format(
                  logdir=tensorboard_logdir(),
                  host=HOST,
                  port="6006"))

shadow_border_css = '''
.bk.ann-automl-shadow-border {
    border-radius: 0px;
    border: none;
    box-shadow: 0 1px 5px grey;
}
'''

scroll_css = '''
.bk.ann-automl-scroll {
    overflow-x: visible !important;
    overflow-y: auto !important;
}
'''

active_header_button_css = '''
.bk.ann-active-head-btn button.bk.bk-btn.bk-btn-default {
    color: white;
    font-size: 16pt;
    background-color: #f57c00;
    border-color: #f57c00;
    min-height: 64px;
    max-width: max-content;
    border-radius: 0px;
}
'''

inactive_header_button_css = '''
.bk.ann-inactive-head-btn button.bk.bk-btn.bk-btn-default {
    color: #ffffffa3;
    font-size: 16pt;
    background-color: #f57c00;
    border-color: #f57c00;
    min-height: 64px;
    max-width: max-content;
    border-radius: 0px;
}
'''

pn.extension(raw_css=[
    shadow_border_css, scroll_css,
    active_header_button_css, inactive_header_button_css
])
pn.config.sizing_mode = 'stretch_width'


task_params = {
    'type': { 'title': 'Тип задачи', 'type': 'str', 'default': 'classification',
              'values': ['classification', 'segmentation', 'detection'] },
    'objects': { 'title': 'Категории изображений', 'type': 'list',
                 'values': [], 'default': [] },
    'func': { 'title': 'Целевой функционал', 'type': 'str',
              'values': ['Метрика'], 'default': 'Метрика' },
    'value': { 'title': 'Значение целевого функционала', 'type':
               'float', 'range': [0, 1], 'step': 0.01, 'scale': 'lin', 'default': 0.9 },
    'maximize': { 'title': 'Продолжать оптимизацию после достижения целевого значения', 'type': 'bool',
                  'default': False }
}

labeling_params = {
    'images_name': { 'title': 'Название базы изображений', 'type': 'str', 'default': 'microtest' },
    'images_path': { 'title': 'Путь к базе изображений', 'type': 'str', 'default': '/auto/projects/brain/ann-automl-gui/datasets/test1/Images example.zip' },
    'images_zip': { 'title': 'База изображений запакована в zip-архив?', 'type': 'bool', 'default': True },
    'nn_core': { 'title': 'Ядро разметчика (используемая нейросеть)',
                 'type': 'str', 'default': 'yolov5s',
                 'values': ['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x'] },
    'save_path': { 'title': 'Каталог для сохранения размеченного датасета', 'type': 'str', 'default': '/auto/projects/brain/ann-automl-gui/datasets/test1//tmpdir1' },
}

dataset_params = {
    'description': { 'title': 'Название', 'type': 'str', 'default': '' },
    'url': { 'title': 'Url', 'type': 'str', 'default': '' },
    'contributor': { 'title': 'Создатель', 'type': 'str', 'default': '' },
    'date_created': { 'title': 'Дата создания', 'type': 'date', 'default': None },
    'version': { 'title': 'Версия', 'type': 'str', 'default': '' },
    'anno_file': { 'title': 'Файл с аннотациями', 'type': 'str', 'default': '' },
    'dir': { 'title': 'Каталог с изображениями', 'type': 'str', 'default': '' },
}

general_params = {
    'tune': { 'title': 'Оптимизировать гиперпараметры', 'type': 'bool','default': False }
}

gui_params = {
    **{
        k: {**v, 'gui': {'group': 'general', 'widget': widget_type(v)}}
        for k,v in general_params.items()
    },
    **{
        f'task_{k}': {**v, 'gui': {'group': 'task', 'widget': widget_type(v)}}
        for k,v in task_params.items()
    },
    **{
        f'dataset_{k}': {**v, 'gui': {'group': 'dataset', 'widget': widget_type(v)}}
        for k,v in dataset_params.items()
    },
    **{
        f'labeling_{k}': {**v, 'gui': {'group': 'labeling', 'widget': widget_type(v)}}
        for k,v in labeling_params.items()
    },
    **hyperparameters
}


def Box(*args, **kwargs):
    return Column(*args, **kwargs, spacing=10, height=700, height_policy='fixed',
                  css_classes=['ann-automl-shadow-border', 'ann-automl-scroll'],
                  margin=(10, 10, 10, 10))


def Button(label, on_click_func, *args, js=False, **kwargs):
    assert 'label' not in kwargs
    if 'button_type' not in kwargs:
        kwargs['button_type'] = 'primary'
    if 'width' not in kwargs:
        kwargs['width'] = 8*len(label) + 50
    button = bokeh.models.Button(*args, **kwargs, label=label)
    if js:
        button.js_on_click(on_click_func)
    else:
        button.on_click(on_click_func)
    return button


def Delimiter(*args, **kwargs, ):
    return Spacer(*args, **kwargs, height=3, background="#b8b8b8",
                  margin=(10, 30, 10, 15))


def Table(source, columns, *args, **kwargs):
    return DataTable(*args, **kwargs, source=source, columns=columns,
                     index_position=None, autosize_mode='fit_columns',
                     sizing_mode='stretch_both',
                     css_classes=['ann-automl-shadow-border', 'ann-automl-scroll'],
                     margin=(10, 10, 10, 10))


def Toggle(label, on_click_func, *args, **kwargs):
    button = bokeh.models.Toggle(*args, **kwargs, label=label,
                                 button_type='primary', width=8*len(label) + 50)
    button.on_click(on_click_func)
    return button


js_open_tensorboard = CustomJS(code='window.open("http://localhost:6006/#scalars");')
js_open_qsl_label = CustomJS(code=f'window.open("http://localhost:{PORT_QSL}");')


class ParamWidget(object):

    _widgets = {}

    def __init__(self, name: str, desc: Dict[str, Any]):

        assert 'gui' in desc and 'default' in desc and 'title' in desc and \
               'group' in desc['gui'] and 'widget' in desc['gui'] and \
                name not in self._widgets

        title = f"{desc['title']}"
        if desc['gui']['widget'] != 'Checkbox':
            title += ':'
        value = desc['default']
        kwargs = {
            'name': name,
            'tags': [desc['gui']['group']],
            'css_classes': ['ann-automl-align'],
            'min_height': 50,
            'sizing_mode': 'stretch_width',
            'margin': (5, 35, 5, 20)
        }

        prefix = name.split('_')[0]

        def default_getter():
            return self._obj.value

        def default_setter(value):
            self._obj.value = value

        def get_values():
            return self._obj.options

        def set_values(values):
            self._obj.options = values

        self._attr = 'value'
        self._title = title
        self._value = value
        self._default_value = value
        self._clear_value = None
        self._getter = default_getter
        self._setter = default_setter
        self._values_getter = None
        self._values_setter = None
        self._conditions = {}
        self._dependencies = []

        if desc['gui']['widget'] == 'Select':
            self._obj = Select(title=title, value=value,
                               options=[x for x in desc['values']], **kwargs)
            self._values_getter = get_values
            self._values_setter = set_values

        elif desc['gui']['widget'] == 'MultiChoice':
            self._obj = MultiChoice(title=title, value=value,
                                    options=[x for x in desc['values']], **kwargs)
            self._values_getter = get_values
            self._values_setter = set_values

        elif desc['gui']['widget'] == 'Slider':

            str_values, cur_index = param_values(
                return_str=True, **{**desc, 'default': value})
            self._values, _ = param_values(**desc)

            try:
                formatter = bokeh.models.FuncTickFormatter(
                    code=f"const labels = {str_values};\nreturn labels[tick];")
            except Exception:
                traceback.print_exc()
                raise

            kwargs['min_height'] = 40
            self._obj = Slider(title=title, value=cur_index,
                               start=0, end=len(self._values)-1,
                               step=1, format=formatter, **kwargs)

            def slicer_getter():
                return self._values[max(0,min(self._obj.value,len(self._values)-1))]

            def slicer_setter(value):
                self._obj.value = np.argmin(np.abs(np.array(self._values) - value))

            self._getter = slicer_getter
            self._setter = slicer_setter

        elif desc['gui']['widget'] == 'Checkbox':
            kwargs['min_height'] = 20
            self._obj = CheckboxGroup(labels=[title],
                                      active=[0] if value else [], **kwargs)

            def checkbox_getter() -> bool:
                return len(self._obj.active) > 0

            def checkbox_setter(value: bool):
                self._obj.active = [0] if value else []

            self._attr = 'active'
            self._getter = checkbox_getter
            self._setter = checkbox_setter

        elif desc['gui']['widget'] == 'Text':
            self._obj = TextInput(title=title, value=value, **kwargs)
            self._clear_value = ""

        elif desc['gui']['widget'] == 'Date':
            if value is None:
                self._value = date.today()
                self._default_value = self._value
                self._obj = DatePicker(title=title, value=date.today(), **kwargs)
            else:
                self._obj = DatePicker(title=title, value=value, **kwargs)

            def date_setter(value):
                try:
                    self._obj.value = value
                except ValueError:
                    self._obj.value = date.today()

            self._setter = date_setter
            self._clear_value = date.today()

        else:
            raise ValueError(f'Unsupported widget type {desc["gui"]["widget"]}')

        self._widgets[name] = self

        if 'cond' in desc:
            for p, vs in desc['cond']:
                widget = self._widgets[p]
                self._conditions[widget] = vs
                widget.add_dependence(self)

        def default_callback(attr, old, new):
            for widget in self._dependencies:
                if widget.active():
                    widget.activate()
                else:
                    widget.hide()

        self._callbacks = [default_callback]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def add_dependence(self, widget):
        self._dependencies.append(widget)

    @property
    def name(self):
        return self._obj.name

    @property
    def group(self):
        return self._obj.tags[0]

    @property
    def title(self):
        return self._title

    @property
    def value(self):
        return self._getter()

    @value.setter
    def value(self, value):
        self._setter(value)

    @property
    def values(self):
        assert self._values_getter is not None
        return self._values_getter()

    @values.setter
    def values(self, values):
        assert self._values_setter is not None
        self._values_setter(values)

    def active(self):
        return all (w.value in vs for w,vs in self._conditions)

    def activate(self):
        self._obj.visible = True

    def hide(self):
        self._obj.visible = False

    def enable(self):
        self._obj.disabled = False

    def disable(self):
        self._obj.disabled = True

    def on_change(self, *callbacks: Callback):
        self._callbacks.extend(callbacks)
        self._obj.on_change(self._attr, *self._callbacks)

    def clear(self):
        assert self._clear_value is not None
        self.value = self._clear_value

    def reset(self):
        self.value = self._value

    def default(self):
        self.value = self._default_value

    def set_default(self, value):
        self._default_value = value
        self._value = value
        self.value = value

    def update(self):
        self._value = self.value

    @property
    def interface(self):
        return self._obj


class NNGui(object):

    def __init__(self, hparams: Dict[str, Any] = hyperparameters):

        self.task = None
        self._task = Widget(name="")
        self.dataset = None
        self.datasets = None
        self._datasets = Widget(name="")
        self.database = {
            ds['description'] : ds
            for db in [cur_db().get_all_datasets_info(full_info=True)] for ds in db.values()
        }
        self.hparams_vals = {}
        self.hparams_desc = hparams

        self.make_params_widgets(gui_params)
        self.init_labeling_interface()
        self.init_database_interface()
        self.init_task_interface()
        self.init_train_interface()
        self.init_history_interface()
        self.init_interface()
        #self.activate_database_interface()
        self.activate_labeling_interface()

    def make_params_widgets(self, params: Dict[str, Any]):
        self.general_params = []
        self.labeling_params = []
        self.task_params = []
        self.dataset_params = []
        self.train_params = []
        self.optimizer_params = []
        self.tune_params = []
        for par, desc in params.items():
            if 'gui' in desc and 'default' in desc and 'title' in desc and \
                    'group' in desc['gui'] and 'widget' in desc['gui']:
                widget = ParamWidget(par, desc)
                setattr(self, widget.name, widget)
                getattr(self, f'{widget.group}_params').append(widget)

    def hparams(self):
        res = self.hparams_vals
        res.update({
                self.hparams_desc[w.name]['param_key']: w.value
                for w in [*self.train_params, *self.optimizer_params]
            })
        return res
    
    def on_click_labeling_start(self):
        err = ""
        if self.labeling_images_name.value == "":
            err = "Название сета изображений не может быть пустым"
        elif not os.path.exists(self.labeling_images_path.value):
            err = "Каталог с изображениями не найден"
        elif self.labeling_save_path.value == "":
            err = "Каталог для сохранения данных не указан"

        if err:
            self.labeling_error.text = f'<font color=red>{err}</font>'
            self.labeling_error.visible = True
            return
        
        labeling_args = {"images_zip" : self.labeling_images_zip.value,
                        "images_name": self.labeling_images_name.value,
                        "images_path": self.labeling_images_path.value,
                        "nn_core": self.labeling_nn_core.value}
        
        self.labeling_working_dir = os.path.join(self.labeling_save_path.value, labeling_args['images_name'])
        
        for widget in self.labeling_params:
            widget.disable()
        self.labeling_start_button.disabled = True

        self.labeling_logs.visible = True
        self.labeling_logs.text = "<b>Выполняется загрузка изображения и автоматическая разметка</b>"
        print(f"smart_labeling.pre_processing... ", end='')
        labels_dict = labeling.pre_processing(labeling_args, self.labeling_working_dir)
        print("ok")
        labels_file = f"{self.labeling_working_dir}/labels.json"
        with open(labels_file, "w") as outfile:
            json.dump(labels_dict, outfile)
        
        self.labeling_logs.text = f"""<b>Автоматическая разметка завершена.
        Ручная доразметка может быть произведена
        <a href="https://{HOST}:{PORT_QSL}">здесь</a>.</b>"""
        self.qsl_label_proc = qsl_label.launch(labels_file, host = HOST, port = PORT_QSL)
        self.labeling_open_qsl_tab.active = 1 #Open tab
        self.labeling_finish_button.visible = True
    
    def on_click_labeling_finish(self):
        self.labeling_logs.text = ""
        self.labeling_logs.visible = False
        self.qsl_label_proc.kill()
        self.labeling_open_qsl_tab.active = 0

        print(f"smart_labeling.post_processing... ", end='')
        annotations_dict = labeling.post_processing(self.labeling_working_dir)
        print("ok")
        with open(f"{self.labeling_working_dir}/annotations/annotations.json", "w") as outfile:
            json.dump(annotations_dict, outfile)

        self.labeling_finish_button.visible = False
        self.labeling_start_button.disabled = False
        for widget in self.labeling_params:
            widget.enable()

    def init_labeling_interface(self):
        self.labeling_logs = Div(align="start", visible=False)
        self.labeling_error = Div(align="center", visible=False, margin=(5, 5, 5, 25))

        self.labeling_start_button = Button('Запустить разметчик',
                                         self.on_click_labeling_start)
        #self.labeling_start_button.js_on_event(ButtonClick, js_open_qsl_label)
        self.labeling_finish_button = Button('Завершить разметку',
                                         self.on_click_labeling_finish, visible=False)
        self.labeling_buttons = [self.labeling_start_button, self.labeling_finish_button, self.labeling_error]

        # Фиктивный виджет. Открытие вкладки для qsl label при установки поля active в значение 1.
        self.labeling_open_qsl_tab = RadioGroup(labels=["Passive state", "Open the tab"],
                                                active=0, visible = False)
        js_conditional_open_qsl_label = CustomJS(
            code=f"""
            const value = cb_obj.active
            if (value == "1")
                window.open("http://localhost:{PORT_QSL}");
            """
            )                                                
        self.labeling_open_qsl_tab.js_on_change("active", js_conditional_open_qsl_label)

        self.labeling_interfaces = [
            Column(self.labeling_images_name.interface,
                   self.labeling_images_path.interface,
                   self.labeling_images_zip.interface,
                   self.labeling_nn_core.interface,
                   self.labeling_save_path.interface,
                   self.labeling_open_qsl_tab,
                   sizing_mode='stretch_both') ]

    def activate_labeling_interface(self):
        self.menu_button.label = "Разметка"
        self.buttons_interface.children = self.labeling_buttons
        self.window_interface.children = self.labeling_interfaces
        self.logs_interface.children = [self.labeling_logs]

    def get_dataset_supercategories(self, ds):
        return list(self.database[ds]['categories'].keys())

    def get_dataset_categories(self, ds, supercategory):
        return [c for c in self.database[ds]['categories'][supercategory]]

    def get_dataset_category(self, ds, supercategory, category):
        n = int(self.database[ds]['categories'][supercategory][category])
        suf = "изображений" if n % 10 in {0,5,6,7,8,9} or n % 100 in {11,12,13,14} else \
              "изображения" if n % 10 in {2,3,4} else \
              "изображение"
        return f"{str(n)} {suf}"

    def on_click_dataset_add(self, event):
        for widget in self.dataset_params:
            widget.clear()
            widget.enable()
        self.dataset_anno_file.activate()
        self.dataset_dir.activate()
        self.dataset_add_button.disabled = True
        self.dataset_select_button.disabled = True
        self.dataset_load_button.visible = True
        self.dataset_cancel_button.visible = True

    def on_click_dataset_load(self, event):
        err = ""
        if self.dataset_description.value == "":
            err = "Название датасета не может быть пустым"
        elif self.dataset_anno_file.value == "":
            err = "Не выбран файл с аннотациями"
        elif not os.path.exists(self.dataset_anno_file.value):
            err = "Файл с аннотациями не найден"
        elif not self.dataset_anno_file.value.endswith('.json'):
            err = "Файл с аннотациями должен быть в формате json"
        elif self.dataset_dir.value == "":
            err = "Не указан каталог с изображениями"
        elif not os.path.exists(self.dataset_dir.value):
            err = "Каталог с изображениями не найден"

        if err:
            self.dataset_error.text = f'<font color=red>{err}</font>'
            self.dataset_error.visible = True
            return

        try:
            cur_db().fill_in_coco_format(
                self.dataset_anno_file.value,
                self.dataset_dir.value,
                ds_info={
                        "description": self.dataset_description.value,
                        "url": self.dataset_url.value,
                        "version": self.dataset_version.value,
                        "year": self.dataset_year.value,
                        "contributor": self.dataset_contributor.value,
                        "date_created": self.dataset_year.value
                    }
            )
            self.database = {
                ds['description'] : ds
                for db in [cur_db().get_all_datasets_info(full_info=True)]
                for ds in db.values()
            }

        except Exception as e:
            self.dataset_error.text = \
                '<font color=red>Не удалось загрузить датасет</font>'
            self.dataset_error.visible = True
            stack = traceback.format_exc()
            self.database_logs.value = '<br>'.join(stack.split('\n') + [str(e)])
            self.database_logs.visible = True
            return

        dataset = self.dataset_description.value
        self.dataset_selector.value = [dataset]
        self.setup_dataset(dataset, update_params=False)
        for widget in self.dataset_params:
            widget.disable()
        self.dataset_anno_file.hide()
        self.dataset_dir.hide()
        self.dataset_add_button.disabled = False
        self.dataset_select_button.disabled = False
        self.dataset_load_button.visible = False
        self.dataset_cancel_button.visible = False

    def on_click_dataset_cancel(self, event):
        self.dataset_error.text = ""
        self.dataset_error.visible = False
        self.database_logs.text = ""
        self.database_logs.visible = False
        if self.dataset is not None:
            self.dataset_selector.value = [self.dataset]
            self.setup_dataset(self.dataset)
        for widget in self.dataset_params:
            widget.disable()
        self.dataset_anno_file.hide()
        self.dataset_dir.hide()
        self.dataset_add_button.disabled = False
        self.dataset_select_button.disabled = False
        self.dataset_load_button.visible = False
        self.dataset_cancel_button.visible = False

    def on_click_dataset_select(self, event):
        self.datasets = self.dataset_selector.value
        self._datasets.name = "SELECTED"
        self.datasets_info.text = \
            f"<p><b>Датасеты:</b> {', '.join(self.datasets)}</p>"
        self.datasets_info.visible = True
        self.task_objects.values = list({
            category for ds in self.datasets
                     for supercategory in self.database[ds]['categories']
                     for category in self.database[ds]['categories'][supercategory]
        })
        self.dataset_select_button.disabled = True
        self.menu_task_button.css_classes=['ann-active-head-btn']

    def setup_dataset(self, dataset, update_params=True):
        self.dataset = dataset

        if update_params:
            for widget in self.dataset_params:
                name = '_'.join(widget.name.split('_')[1:])
                if name in self.database[dataset]:
                    widget.value = self.database[dataset][name]

        supercategories = self.get_dataset_supercategories(dataset)
        self.dataset_supercategory.options = supercategories
        self.dataset_supercategory.value = supercategories[0]

        categories = self.get_dataset_categories(dataset, supercategories[0])
        self.dataset_category.options = categories
        self.dataset_category.value = categories[0]

        self.dataset_categories_num.text = \
            self.get_dataset_category(dataset, supercategories[0], categories[0])

    def init_database_interface(self):
        self.dataset_error = Div(align="center", visible=False, margin=(5, 5, 5, 25))
        self.database_logs = Div(align="start", visible=False)

        def changeDataset(attr, old, new):
            self.setup_dataset(new[0])
            self.dataset_select_button.disabled = False

        datasets = list(self.database.keys())
        self.dataset_selector = MultiSelect(
            value=datasets[:1] if len(datasets) > 0 else [], options=datasets,
            min_width=120, width_policy="max", height_policy="max", margin=(5,15,5,5)
        )
        self.dataset_selector.on_change('value', changeDataset)

        def changeSupercategory(attr, old, new):
            categories = self.get_dataset_categories(self.dataset, new)
            self.dataset_category.options = categories
            self.dataset_category.value = categories[0]
            self.dataset_categories_num.text = \
                self.get_dataset_category(self.dataset, new, categories[0])

        def changeCategory(attr, old, new):
            supercategory = self.dataset_supercategory.value
            self.dataset_categories_num.text = \
                self.get_dataset_category(self.dataset, supercategory, new)

        self.dataset_supercategory = Select()
        self.dataset_supercategory.on_change('value', changeSupercategory)
        self.dataset_category = Select()
        self.dataset_category.on_change('value', changeCategory)
        self.dataset_categories_num = Div(align='center', min_width=150)
        self.dataset_categories = \
            Row(Div(text="<b>Категории изображений:</b>", min_width=160),
                self.dataset_supercategory, self.dataset_category,
                self.dataset_categories_num)

        self.dataset_add_button = Button('Добавить новый датасет',
                                         self.on_click_dataset_add)
        self.dataset_load_button = Button('Загрузить', self.on_click_dataset_load,
                                          visible=False)
        self.dataset_cancel_button = Button('Отменить', self.on_click_dataset_cancel,
                                            visible=False)
        self.dataset_select_button = Button('Использовать выбранные датасеты',
                                            self.on_click_dataset_select,
                                            disabled=True)

        self.database_buttons = [self.dataset_select_button, self.dataset_add_button,
                                 self.dataset_cancel_button, self.dataset_load_button,
                                 self.dataset_error]

        self.database_interfaces = [
            Column(Div(text="<b>Доступные датасеты:</b>"),
                   self.dataset_selector, self.dataset_categories, margin=(0, 0, 0, 5)),
            Column(self.dataset_description.interface,
                   self.dataset_url.interface,
                   self.dataset_contributor.interface,
                   self.dataset_date_created.interface,
                   self.dataset_version.interface,
                   self.dataset_anno_file.interface,
                   self.dataset_dir.interface,
                   sizing_mode='stretch_both') ]

        if len(datasets) > 0:
            self.setup_dataset(datasets[0])
        for widget in self.dataset_params:
            widget.disable()
        self.dataset_anno_file.hide()
        self.dataset_dir.hide()

        self.database_interface_init = True

    def activate_database_interface(self):
        self.menu_button.label = "База данных"
        self.buttons_interface.children = self.database_buttons
        self.window_interface.children = self.database_interfaces
        self.logs_interface.children = [self.database_logs]

    def on_click_task_apply(self, event):

        if len(self.task_objects.value) < 2:
            self.task_error.text = \
                '<font color=red>Выберите не менее двух категорий изображений</font>'
            self.task_error.visible = True

        else:
            cur_db().ds_filter = list(self.datasets)

            print("Creating an NNTask object ... " , end='', flush=True)
            self.task = NNTask(
                type=self.task_type.value,
                objects=self.task_objects.value,
                func={ 'Метрика': metric_target }[self.task_func.value],
                target=self.task_value.value,
                goals={ 'maximize': self.task_maximize.value }
            )
            self._task.name = "NNTask"
            print("ok")

            print("Getting recommended hyperparamters ... ", end='', flush=True)
            self.hparams_vals = recommend_hparams(self.task, trace_solution=True)
            for widget in [*self.train_params, *self.optimizer_params]:
                par = '_'.join(widget.name.split('_')[1:])
                if par in self.hparams_vals:
                    widget.set_default(self.hparams_vals[par])
            print("ok")

            print("Loading learning history ... ", end='', flush=True)
            learning_histories = params_from_history(self.task)
            # TODO: может всё-таки историю не всегда, а только по запросу загружать?
            self.update_history_interface(learning_histories)
            print("ok")

            self.task_type_info.text = f"<p><b>Задача:</b> {self.task_type.value}</p>"
            self.task_type_info.visible = True
            self.task_objects_info.text = f"<p><b>Категории:</b> {', '.join(self.task_objects.value)}</p>"
            self.task_objects_info.visible = True
            self.task_func_info.text = f"<p><b>Функционал:</b> {self.task_func.value}</p>"
            self.task_func_info.visible = True
            self.task_value_info.text = f"<p><b>Значение:</b> {self.task_value.value:.2f}</p>"
            self.task_value_info.visible = True
            self.task_maximize_info.text = f"<p><b>Максимально оптимизировать:</b> {'Да' if self.task_maximize.value else 'Нет'}</p>"
            self.task_maximize_info.visible = True
            self.task_apply_button.disabled = True
            self.menu_train_button.css_classes=['ann-active-head-btn']
            self.menu_history_button.css_classes=['ann-active-head-btn']

    def init_task_interface(self):
        self.task_error = Div(align="center", visible=False, margin=(5, 5, 5, 25))
        self.task_logs = Div(align="start", visible=False)

        def changeTaskParam(attr, old, new):
            self.task_apply_button.disabled = False

        for widget in self.task_params:
            widget.on_change(changeTaskParam)

        def changeTaskObjects(attr, old, new):
            changeTaskParam(attr, old, new)
            self.task_error.text = ""
            self.task_error.visible = False
        self.task_objects.on_change(changeTaskObjects)

        self.task_apply_button = Button('Создать задачу', self.on_click_task_apply)

        self.task_buttons = [self.task_apply_button, self.task_error]
        self.task_interfaces = [Column(self.task_type.interface,
                                       self.task_objects.interface,
                                       self.task_func.interface,
                                       self.task_value.interface,
                                       self.task_maximize.interface,
                                       sizing_mode='stretch_both')]

        self.task_interface_init = True

    def activate_task_interface(self):
        if self.datasets is None:
            return
        self.menu_button.label = "Задача"
        self.buttons_interface.children = self.task_buttons
        self.window_interface.children = self.task_interfaces
        self.logs_interface.children = [self.task_logs]

    def on_click_train_box_button(self, active):
        self.train_params_box.visible = 0 in active
        self.trining_output_box.visible = 1 in active
        self.trining_tools_box.visible = 2 in active

    def update_bokeh_server(self, *args, **kwargs):
        self.train_output.value = self.train_logs
        epochs = self.loss_acc_plot_attr['epochs']
        losses = self.loss_acc_plot_attr['losses']
        accuracies = self.loss_acc_plot_attr['accuracies']
        val_losses = self.loss_acc_plot_attr['val_losses']
        val_accuracies = self.loss_acc_plot_attr['val_accuracies']
        last_epoch = self.loss_acc_plot_attr['last_epoch']

        # TODO: сделать здесь по-нормальному обновление графиков (через поток данных)
        if len(epochs) > last_epoch+1:
            ll = len(epochs)-1
            # update self.loss_acc_plot
            self.loss_acc_plot.line(list(epochs[last_epoch:]),
                                    list(losses[last_epoch:]),
                                    legend_label='Loss', line_color='red')
            self.loss_acc_plot.line(list(epochs[last_epoch:]),
                                    list(accuracies[last_epoch:]),
                                    legend_label='Accuracy', line_color='green')
            if val_losses:
                self.loss_acc_plot.line(list(epochs[last_epoch:]),
                                        list(val_losses[last_epoch:]),
                                        legend_label='Val Loss', line_color='blue')
            if val_accuracies:
                self.loss_acc_plot.line(list(epochs[last_epoch:]),
                                        list(val_accuracies[last_epoch:]),
                                        legend_label='Val Accuracy', line_color='black')
            self.loss_acc_plot_attr['last_epoch'] = ll

        if self.is_train_stop:
            self.start_button.disabled = False
            self.stop_button.disabled = True
            self.continue_button.visible = self.is_train_break
            bokeh.io.curdoc().remove_periodic_callback(self.bokeh_timer)
            self.bokeh_timer = None
            self.is_train_stop = False
            self.is_train_break = False
            for widget_interface in self.train_params_box.children:
                widget_interface.disabled = False

    def add_plot_point(self, epoch, loss, accuracy, val_loss, val_accuracy):
        self.loss_acc_plot_attr['epochs'].append(epoch)
        self.loss_acc_plot_attr['losses'].append(loss)
        self.loss_acc_plot_attr['accuracies'].append(accuracy)
        if val_loss is not None:
            self.loss_acc_plot_attr['val_losses'].append(min(5, val_loss))
        if val_accuracy is not None and 0 <= val_accuracy <= 1:
            self.loss_acc_plot_attr['val_accuracies'].append(val_accuracy)

    def msg(self, *args, file=None, end='\n', **kwargs):
        if file is None:
            self.train_logs += ''.join(map(str, args)) + end
        else:
            print(*args, **kwargs, file=file, end=end)

    def on_process_finish(self):
        self.is_train_stop = True

    def on_train_callback(self, tp, batch=None, epoch=None, logs=None, model=None):
        if model is not None:
            self.model = model
        if tp == 'epoch':
            self.msg(f'Эпоха {epoch}: {logs}')
            self.add_plot_point(epoch, logs['loss'], logs['accuracy'],
                                logs.get('val_loss', None),
                                logs.get('val_accuracy', None))
            time.sleep(0.1)
        elif tp == 'finish':
            self.msg('Обучение завершено')

    def append_history(self, history):
        self.history_table.source.stream({
            'dates': [str(history['date'])],
            'funcs': [str(history['metric_name'])],
            'values': [str(history['metric_value'])],
            'times': [str(history['total_time'])]
        })
        self.history_hparams.append(history['hparams'])

    def on_click_start(self, event):
        self.is_train_stop = False
        self.is_train_break = False
        self.start_button.disabled = True
        self.stop_button.disabled = False
        for widget_interface in self.train_params_box.children:
            widget_interface.disabled = True

        # create timer to update bokeh widgets
        self.bokeh_timer = bokeh.io.curdoc().add_periodic_callback(
            self.update_bokeh_server, 1000)

        # get params
        initial_params = self.hparams()
        hparams = recommend_hparams(self.task, fixed_params=self.hparams(), trace_solution=True)
        # print differece between initial and recommended params
        changed_params = {k: f'{initial_params.get(k, None)} --> {v}' for k, v in hparams.items() if v != initial_params.get(k, None)}
        if changed_params:
            self.msg('Изменённые параметры:')
            for k, v in changed_params.items():
                self.msg(f'\t{k}: {v}')

        self.stop = StopFlag()
        if self.tune.value:
            from ..core.nnfuncs import tune_hparams
            param_names = tune_hparams['method']['values'][self.tune_method.value]['params']
            additional_args = {k: getattr(self, f'tune_{k}').value for k in param_names}
            self.process = process(tune)(nn_task=self.task, stop_flag=self.stop,
                                         # tuned_params=['optimizer', 'batch_size', 'learning_rate'],
                                         # tuned_params=self.tune_tuned_params.value,
                                         method=self.tune_method.value,
                                         hparams=hparams,
                                         **additional_args,
                                         start=False)
        else:
            self.process = process(train)(nn_task=self.task, stop_flag=self.stop,
                                          hparams=hparams, start=False,
                                          model=self.model)
        self.process.set_handler(
            'print', lambda *args, **kwargs: self.msg(*args, **kwargs))
        self.process.set_handler(
            'train_callback', lambda *args, **kwargs: self.on_train_callback(*args, **kwargs))
        self.process.set_handler(
            'append_history', lambda *args, **kwargs: self.append_history(*args, **kwargs))
        self.process.on_finish = lambda _: self.on_process_finish()

        print('Запуск процесса обучения')
        self.process.start()

    def on_click_stop(self, event):
        self.is_train_break = True
        self.stop()

    def on_click_continue(self, event):
        self.continue_button.visible = False
        self.train_logs += 'Continue is not supported, use start button.\n'
        self.train_output.value = self.train_logs
        # TODO: реализовать интерфейс

    def init_train_interface(self):

        self.train_logs = ""

        print("Create train_params_box ... ", end='', flush=True)
        self.train_params_box = Box(Spacer(height=10),
                                    *[w.interface for w in self.general_params],
                                    Delimiter(),
                                    *[w.interface for w in self.train_params],
                                    Delimiter(),
                                    *[w.interface for w in self.optimizer_params],
                                    Delimiter(),
                                    *[w.interface for w in self.tune_params],
                                    Spacer(height=10))
        print("ok")

        print("Create train_output_box ... ", end='', flush=True)
        self.train_output = TextAreaInput(value = self.train_logs,
                                          min_width=500,
                                          sizing_mode='stretch_both',
                                          disabled=True)
        self.trining_output_box = Box(self.train_output, sizing_mode='stretch_width')
        print("ok")

        print("Create train_tools_box ... ", end='', flush=True)
        self.loss_acc_plot = Figure(title='Loss and Accuracy',
                                    x_axis_label='Epoch',
                                    y_axis_label='Loss/Accuracy',
                                    plot_width=500, plot_height=250,
                                    sizing_mode='stretch_both')
        self.loss_acc_plot_attr = dict(epochs=[], losses=[], accuracies=[],
                                       val_losses=[], val_accuracies=[],
                                       last_epoch=0)
        self.model = None
        self.trining_tools_box = Box(self.loss_acc_plot, sizing_mode='stretch_width')
        print("ok")

        print("Create box_button_group ... ", end='', flush=True)
        self.train_box_button_group = CheckboxButtonGroup(
            labels=['Параметры', 'Журнал', 'Инструменты'],
            button_type='primary', active=[0,1,2], margin=(5,30,5,5))
        self.train_box_button_group.on_click(self.on_click_train_box_button)
        print("ok")

        print("Create tensorboard_button ... ", end='', flush=True)
        self.tensorboard_button = Button('Tensorboard', js_open_tensorboard, js=True)
        print("ok")

        self.start_button = Button('Старт', self.on_click_start)
        self.stop_button = Button('Стоп', self.on_click_stop, disabled=True)
        self.continue_button = Button('Продолжить', self.on_click_continue, visible=False)

        self.train_buttons = [
                self.train_box_button_group, self.tensorboard_button,
                self.start_button, self.stop_button, self.continue_button
            ]
        self.train_interfaces = [
                self.train_params_box, self.trining_output_box,
                self.trining_tools_box
            ]

        self.train_interface_init = True

    def activate_train_interface(self):
        self.model = None
        if self.task is None:
            return
        self.menu_button.label = "Обучение"
        self.buttons_interface.children = self.train_buttons
        self.window_interface.children = self.train_interfaces
        self.logs_interface.children = []
        for widget in [*self.train_params, *self.optimizer_params]:
            widget.reset()
            widget.enable()
            widget.activate()

    def on_click_history_download(self, event):
        # TODO: реализовать интерфейс
        pass

    def on_click_history_apply(self, event):
        for widget in self.history_params:
            widget.update()
        self.history_apply_button.disabled = True

    def init_history_interface(self):
        self.history_model = dict(url="", fileName="")
        self.history_logs = Div(align="start", visible=False)
        try:
            self.history_hparams = []
            source = ColumnDataSource({
                    x:[] for x in ['dates', 'funcs', 'values', 'times']
                })
            self.history_table = Table(source, [
                    TableColumn(field="dates", title="Дата"),
                    TableColumn(field="funcs", title="Функционал"),
                    TableColumn(field="values", title="Значение"),
                    TableColumn(field="times", title="Время обучения",
                                formatter=NumberFormatter(format='00:00:00')),
                ])

            def on_select_history_item(attr, old, new):
                try:
                    for widget in [*self.train_params, *self.optimizer_params]:
                        widget.hide()
                    self.history_params = []
                    i = source.selected.indices[0]
                    for k,v in self.history_hparams[i].items():
                        if hasattr(self, f'train_{k}'):
                            # TODO: Среди рекомендованных параметров встречаются те,
                            # которые не имеют параметра 'gui', поэтому для них
                            # не были созданы виджеты
                            widget = getattr(self, f'train_{k}')
                            self.history_params.append(widget)
                            widget.value = v
                            widget.activate()
                    self.history_model["url"] = self.history_models[i]
                    # TODO: Надо согласовать название сохраняемого файла с моделью
                    self.history_model["fileName"] = os.path.basename(self.history_models[i])
                    self.history_download_button.disabled = False
                    self.history_apply_button.disabled = False
                except IndexError:
                    pass
            source.selected.on_change('indices', on_select_history_item)
        except Exception as e:
            traceback.print_exc()
            print(f'Exception occured during History.__init__: {e}')

        self.history_params_box = Box(Spacer(height=10),
                                      *[w.interface for w in self.train_params],
                                      Delimiter(),
                                      *[w.interface for w in self.optimizer_params],
                                      Spacer(height=10))

        self.history_download_button = Button('Скачать', self.on_click_history_download)
        self.history_download_button.js_on_click(CustomJS(
            args=dict(model=self.history_model),
            code='''let url = model.url;
                    let fileName = model.fileName;

                    const elStatus = document.getElementById('model_status');
                    function status(text) {
                        elStatus.innerHTML = text;
                    }

                    const elProgress = document.getElementById('model_progress');
                    function progress({loaded, total}) {
                        elProgress.innerHTML = Math.round(loaded/total*100)+'%';
                    }

                    function clear() {
                        elStatus.innerHTML = "";
                        elProgress.innerHTML = "";
                    }

                    async function main() {
                        status('downloading ...');
                        const response = await fetch(url);
                        const contentLength = response.headers.get('content-length');
                        const total = parseInt(contentLength, 10);
                        let loaded = 0;

                        const res = new Response(new ReadableStream({
                            async start(controller) {
                            const reader = response.body.getReader();
                            for (;;) {
                                const {done, value} = await reader.read();
                                if (done) break;
                                loaded += value.byteLength;
                                progress({loaded, total})
                                controller.enqueue(value);
                            }
                            controller.close();
                            },
                        }));
                        const blob = await res.blob();
                        status('download completed')
                        const aElement = document.createElement('a');
                        aElement.setAttribute('download', fileName);
                        const href = URL.createObjectURL(blob);
                        aElement.href = href;
                        aElement.setAttribute('target', '_blank');
                        aElement.click();
                        URL.revokeObjectURL(href);
                        clear()
                    }

                    if ( url != "" )
                    {
                        main();
                    }'''))

        self.history_apply_button = Button('Использовать', self.on_click_history_apply)

        self.history_buttons = [self.history_apply_button, self.history_download_button,
                                Div(text='<span id="model_status"></span> <span id="model_progress"></span>',
                                    align='center', width=300)]

        self.history_interfaces = [self.history_table, self.history_params_box]

    def update_history_interface(self, histories):
        try:
            data = { x:[] for x in ['dates', 'funcs', 'values', 'times'] }
            self.history_hparams = []
            self.history_models = []
            for h in histories:
                data['dates'].append(h['date'])
                data['funcs'].append(h['metric_name'])
                data['values'].append(str(h['metric_value']))
                data['times'].append(str(h.get('total_time', -1.0)))
                self.history_hparams.append(h['hparams'])
                self.history_models.append(h['model_file'])
        except Exception as e:
            traceback.print_exc()
            print(f'Exception occured during History.__init__: {e}')

        self.history_table.source.data = data

    def activate_history_interface(self):
        if self.task is None:
            return
        self.history_table.source.selected.indices = [0]
        self.menu_button.label = "История"
        self.buttons_interface.children = self.history_buttons
        self.window_interface.children = self.history_interfaces
        self.logs_interface.children = [self.history_logs]
        self.history_download_button.disabled = True
        self.history_apply_button.disabled = True
        for widget in [*self.train_params, *self.optimizer_params]:
            widget.disable()

    def init_interface(self):

        for field in ['datasets', 'task_type', 'task_objects',
                      'task_func', 'task_value', 'task_maximize']:
            widget = Div(align='start', visible=False, height=50, height_policy='fixed',
                         sizing_mode='stretch_width')
            setattr(self, f'{field}_info', widget)

        menu = [
            ("Разметка", "labeling"),
            ("База данных", "database"),
            ("Задача", "task"),
            ("Обучение", "train"),
            ("История", "history")
        ]
        self.menu_button = Dropdown(label="База данных", button_type="primary", menu=menu,
                                    width=200, margin=(5, 5, 5, 10))

        def menu_button_handler(event):
            print("Goto ", event.item, "interface")
            getattr(self, f'activate_{event.item}_interface')()
        self.menu_button.on_event("menu_item_click", menu_button_handler)
        self.menu_button.js_on_click(CustomJS(
            args=dict(task=self._task, datasets=self._datasets),
            code='''let msg;
                    if ( this.item != "database" ) {
                        if ( this.item == "task") {
                            msg = "создание задачи невозможно"
                        } else if ( this.item == "train") {
                            msg = "запуск обучения невозможен"
                        } else if ( this.item == "history") {
                            msg = "просмотр истории обучений невозможен"
                        } else {
                            msg = ""
                        }
                        if ( datasets.name == "" ) {
                            alert("Датасеты не заданы, " + msg + "!");
                        } else if ( task.name == "" && this.item != "task" ) {
                            alert("Задача не создана, " + msg + "!");
                        }
                    }'''))


        self.buttons_interface = Row(margin=(5, 5, 5, 5))
        self.window_interface = Row()
        self.logs_interface = Row(margin=(-5, 5, 5, 5))

        self.interface = \
            Column(Row(self.datasets_info, self.task_type_info, self.task_objects_info,
                       self.task_func_info, self.task_value_info, self.task_maximize_info,
                       spacing=10, min_width=1200, sizing_mode='stretch_width',
                       margin=(-10, 5, -5, 5)),
                   self.buttons_interface, self.window_interface, self.logs_interface, spacing=5)

        self.menu_labeling_button = Button("Разметка", self.on_click_labeling_menu,
                                           button_type='default',
                                           css_classes=['ann-active-head-btn'])
        self.menu_database_button = Button("База данных", self.on_click_database_menu,
                                           button_type='default',
                                           css_classes=['ann-active-head-btn'])
        self.menu_task_button = Button("Задача", self.on_click_task_menu,
                                       button_type='default',
                                       css_classes=['ann-inactive-head-btn'])
        self.menu_task_button.js_on_click(CustomJS(
            args=dict(task=self._task, datasets=self._datasets),
            code='''let msg = "создание задачи невозможно";
                    if ( datasets.name == "" ) {
                        alert("Датасеты не заданы, " + msg + "!");
                    } '''))
        self.menu_train_button = Button("Обучение", self.on_click_train_menu,
                                        button_type='default',
                                        css_classes=['ann-inactive-head-btn'])
        self.menu_train_button.js_on_click(CustomJS(
            args=dict(task=self._task, datasets=self._datasets),
            code='''let msg = "запуск обучения невозможен";
                    if ( datasets.name == "" ) {
                        alert("Датасеты не заданы, " + msg + "!");
                    } else if ( task.name == "" ) {
                        alert("Задача не создана, " + msg + "!");
                    }'''))
        self.menu_history_button = Button("История", self.on_click_history_menu,
                                          button_type='default',
                                          css_classes=['ann-inactive-head-btn'])
        self.menu_history_button.js_on_click(CustomJS(
            args=dict(task=self._task, datasets=self._datasets),
            code='''let msg = "просмотр истории обучений невозможен";
                    if ( datasets.name == "" ) {
                        alert("Датасеты не заданы, " + msg + "!");
                    } else if ( task.name == "" ) {
                        alert("Задача не создана, " + msg + "!");
                    }'''))

        self.menu_buttons = Row(self.menu_labeling_button,
                                self.menu_database_button, self.menu_task_button,
                                self.menu_train_button, self.menu_history_button)

    def on_click_labeling_menu(self, event):
        print("Goto labeling interface")
        self.activate_labeling_interface()
    
    def on_click_database_menu(self, event):
        print("Goto database interface")
        self.activate_database_interface()

    def on_click_task_menu(self, event):
        print("Goto task interface")
        self.activate_task_interface()

    def on_click_train_menu(self, event):
        print("Goto train interface")
        self.activate_train_interface()

    def on_click_history_menu(self, event):
        print("Goto history interface")
        self.activate_history_interface()

gui = NNGui()

interface = pn.template.MaterialTemplate(
    title="Ann Automl",
    # sidebar=[pn.pane.Markdown("## Settings")],
    header=[gui.menu_buttons],
    main=[
        gui.interface
        # pipeline.stage,
        # pn.layout.Divider(margin=(50, 0, 50, 0)), pn.Row(pipeline.network, pipeline.buttons)
    ],
    modal=[pn.pane.Markdown("## Modal")],
    header_background="#f57c00",
    favicon="https://static.centro.org.uk/img/wmca/favicons/mstile-150x150.png"
)
