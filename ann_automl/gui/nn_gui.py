import sys
import time

import panel as pn
import param
import pandas as pd
import datetime as dt
import bokeh
from typing import Any, Callable

from ann_automl.core.db_module import DBModule
from ann_automl.core.solver import Task
from ..utils.process import process
from .params import hyperparameters
from ann_automl.gui.transition import Transition
import ann_automl.gui.tensorboard as tb
from ..core.nn_solver import NNTask, recommend_hparams
from ..core.nnfuncs import nnDB as DB, StopFlag, train
from ..core import nn_rules_simplified

# Launch TensorBoard
tb.start("--logdir ./logs --port 0")

css = '''
.bk.panel-widget-box {
    background: #fafafa;
    border-radius: 0px;
    border: 1px solid #dcdcdc;
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
            for db in [DB.get_all_datasets_info(full_info=True)] for ds in db.values()
        },
        'title': 'База данных'
    },
    'dataset': {
        'default': None,
        'title': 'Текущий датасет'
    },
    'selected_datasets': {
        'default': [],
        'title': 'Выделенный список датасетов'
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
            'widget': 'Select'
        }
    },
    'task_objects': {
        'values': [],
        'default': [],
        'title': 'Категории изображений',
        'gui': {
            'group': 'Task',
            'widget': 'MultiChoice'
        }
    },
    'task_target': {
        'values': ['loss', 'metrics'],
        'default': 'loss',
        'title': 'Целевой функционал',
        'gui': {
            'group': 'Task',
            'widget': 'Select'
        }
    },
    'task_target_value': {
        'range': [0, 1], 
        'default': 0.7, 
        'step': 0.05, 
        'scale': 'lin',
        'title': 'Желаемое значение целевого функционала',
        'gui': {
            'group': 'Task',
            'widget': 'Slider'
        }
    },
    'task_maximize_target': {
        'default': True,
        'title': 'Максимизировать целевой функционал после достижения желаемого значения',
        'gui': {
            'group': 'Task',
            'widget': 'Checkbox'
        }
    },
    'tune': {
        'default': False,
        'title': 'Оптимизировать гиперпараметры обучения нейронной сети',
        'gui': {
            'group': 'General',
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
        print(f"{self.__class__.__name__}.__init__ called")

    def close(self):
        self.ready=True

    def param_widget(self, name: str, change_callback: Callable[[Any, Any, Any], None]):
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
        return (widget.name not in gui_params or
                'cond' not in gui_params[widget.name] or 
                all (self.params[par] in values for par, values in gui_params[widget.name]['cond']))



class Database(Window):

    next_window = param.Selector(default='Task', objects=['DatasetLoader', 'Task'])
    
    def __init__(self, **params):
        super().__init__(**params)

        def changeDatasetCallback(attr, old, new):
            new_ds = new[0]
            self.init_dataset_info_interface(new_ds)
            self.dataset_info.visible = True
            self.dataset_apply_button.disabled = False

        self.dataset_selector = bokeh.models.MultiSelect(
            options=list(self.params['db'].keys()),  
            max_width=450, width_policy='min', height_policy="max", margin=(5,15,5,5)
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

        self.dataset_description = bokeh.models.Div()
        self.dataset_url = bokeh.models.Div()
        self.dataset_contributor = bokeh.models.Div()
        self.dataset_data = bokeh.models.Div()
        self.dataset_version = bokeh.models.Div()
        self.dataset_supercategories_selector = \
            bokeh.models.Select(width=250, width_policy='fixed')
        self.dataset_supercategories_selector.on_change('value', 
            changeDatasetSupercategoriesCallback)
        self.dataset_categories_selector = \
            bokeh.models.Select(width=250, width_policy='fixed')
        self.dataset_categories_selector.on_change('value', 
            changeDatasetCategoriesCallback)
        self.dataset_categories_info = bokeh.models.Div(align='center')

        ds = None
        if len(self.dataset_selector.options) > 0:
            ds = self.dataset_selector.options[0]
            self.init_dataset_info_interface(ds)

        self.dataset_info = pn.Column(
            self.dataset_description,
            self.dataset_url,
            self.dataset_contributor,
            self.dataset_data,  
            self.dataset_version,
            pn.Row(
                bokeh.models.Div(text="<b>Категории изображений:</b>", min_width=160),
                self.dataset_supercategories_selector,
                self.dataset_categories_selector,
                self.dataset_categories_info
            ),
            visible=ds is not None
        )

        self.selected_datasets = bokeh.models.Div(
            text = f"<b>Используемые датасеты:</b> {', '.join(self.params['selected_datasets'])}",
            visible=ds is not None, margin=(5,5,10,10)
        )

        self.dataset_load_button=pn.widgets.Button(
            name='Добавить датасет', button_type='primary', 
            align='start', width=120)
        self.dataset_load_button.on_click(self.on_click_dataset_load)

        self.dataset_apply_button=pn.widgets.Button(
            name='Использовать выбранные датасеты', button_type='primary', 
            align='end', width=220, disabled=True)
        self.dataset_apply_button.on_click(self.on_click_dataset_apply)

        self.next_button=pn.widgets.Button(
            name='Далее', button_type='primary', 
            align='end', width=100, disabled=len(self.params['selected_datasets']) == 0)
        self.next_button.on_click(self.on_click_next)


    def get_dataset_supercategories(self, ds):
        return list(self.params['db'][ds]['categories'].keys())


    def get_dataset_categories(self, ds, supercategory):
        return list(self.params['db'][ds]['categories'][supercategory].keys())


    def get_dataset_category_info(self, ds, supercategory, category):
        num = int(self.params['db'][ds]['categories'][supercategory][category])
        suf = "штук" if 5 <= num % 10 and num % 10 <= 10 or 11 <= num and num <= 14 else \
              "штуки" if 2 <= num % 10 and num % 10 <= 4 else \
              "штука"
        return f"{str(num)} {suf}"


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
        self.dataset_data.text=f"<b>Дата создания:</b> {data}"
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

        self.dataset_apply_button.disabled = True
        self.next_button.disabled = False

    def on_click_next(self, event):
        self.next_window = 'Task'
        self.close()

    def panel(self):
        return pn.Column(
            '# База данных изображений',
            bokeh.models.Div(text="<b>Доступные датасеты:</b>", margin=(-10,0,0,10)),
            pn.Row(self.dataset_selector, self.dataset_info, margin=(0,5,5,5)),
            self.selected_datasets,
            pn.Row(self.dataset_load_button, self.dataset_apply_button, self.next_button),
        )


class DatasetLoader(Window):

    next_window = param.Selector(default='Database', objects=['Database'])

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

        self.apply_button=pn.widgets.Button(
            name='Загрузить', button_type='primary', 
            align='end', width=100)
        self.apply_button.on_click(self.on_click_apply)

        self.back_button=pn.widgets.Button(
            name='Назад', button_type='primary', 
            align='start', width=100)
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


class Task(Window):

    next_window = param.Selector(default='Params', 
                                 objects=['Database', 'Params', 'TrainedModels'])
    
    def __init__(self, **params):
        super().__init__(**params)
        
        def changeTaskParamCallback(attr, old, new):
            self.apply_button.disabled = False

        for widget in self.group_params_widgets('Task', changeTaskParamCallback):
            setattr(self, f"{widget.name}_selector", widget)
        
        def changeTaskObjects(attr, old, new):
            changeTaskParamCallback(attr, old, new)
            self.task_objects_checker.text = ""
        self.task_objects_selector.on_change('value', changeTaskObjects)

        self.task_interface = pn.Column(
            '# Задача анализа изображений',
            self.task_category_selector,
            self.task_type_selector,
            self.task_objects_selector,
            self.task_target_selector,
            self.task_target_value_selector,
            self.task_maximize_target_selector
        )
        
        self.task_objects_checker = bokeh.models.Div(align="center", margin=(5, 5, 5, 25))

        self.checkbox = pn.widgets.Checkbox(name='Подобрать готовые модели')

        self.apply_button=pn.widgets.Button(
            name='Создать задачу', button_type='primary', 
            align='start', width=100, disabled=self.params['task'] is not None)
        self.apply_button.on_click(self.on_click_apply)

        self.next_button=pn.widgets.Button(
            name='Далее', button_type='primary', 
            align='end', width=100, disabled=self.params['task'] is None)
        self.next_button.on_click(self.on_click_next)

        self.back_button=pn.widgets.Button(
            name='Назад', button_type='primary', 
            align='start', width=100)
        self.back_button.on_click(self.on_click_back)

    def on_click_apply(self, event):

        if len(self.task_objects_selector.value) < 2:
            self.task_objects_checker.text = \
                '<font color=red>Выберите не менее двух категорий изображений</font>'

        else:
            # CORE:
            self.params['task'] = NNTask(
                task_ct=self.params['task_category'],
                task_type=self.params['task_type'],
                objects=self.params['task_objects'],
                metric=self.params['task_target'],
                target=self.params['task_target_value'],
                goals={'maximize': self.params['task_maximize_target']}
            )
            hparams = recommend_hparams(self.params['task'], trace_solution=True)
            self.params['recommended_hparams'] = hparams
            for k, v in hparams.items():
                key = 'train.' + k
                self.params[key] = v

            self.apply_button.disabled = True
            self.next_button.disabled = False

    def on_click_next(self, event):
        if self.checkbox.value:
            self.next_window = 'TrainedModels'
        self.close()

    def on_click_back(self, event):
        self.next_window = 'Database'
        self.close()

    def panel(self):
        return pn.Column(
            '# Задача анализа изображений',
            self.task_category_selector,
            self.task_type_selector,
            self.task_objects_selector,
            self.task_target_selector,
            self.task_target_value_selector,
            self.task_maximize_target_selector,
            pn.Row(self.apply_button, self.task_objects_checker),
            pn.Spacer(height=10),
            self.checkbox,
            pn.Row(self.back_button, self.next_button)
        )


class Params(Window):

    next_window = param.Selector(default='Task', objects=['Task', 'Training'])

    def __init__(self, **params):
        super().__init__(**params)

        self.params_widgets = [
            ("Общие параметры", self.group_params_widgets('General', lambda: None)),
            ("Параметры автонастройки", self.group_params_widgets('Tune', lambda: None)),
            ("Параметры обучения", self.group_params_widgets('Learning', lambda: None)),
            ("Параметры оптимизатора", self.group_params_widgets('Optimizer', lambda: None))
        ]

        def to_column(widgets):
            return bokeh.models.Column( bokeh.models.Spacer(height=10), *widgets,
                sizing_mode="stretch_width", height=500, height_policy="fixed", css_classes=['scrollable'])

        self.tabs = bokeh.models.Tabs(
            tabs=[bokeh.models.Panel(title=title, child=to_column(widgets)) for title, widgets in self.params_widgets])

        def panelActive(attr, old, new):
            if self.tabs.active == 3:
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

    next_window = param.Selector(default='Task', objects=['Task'])

    def __init__(self, **params):
        super().__init__(**params)

        self.stop_button=pn.widgets.Button(name='Стоп', align='end', width=100, button_type='primary')
        self.stop_button.on_click(self.on_click_stop)

        self.back_button=pn.widgets.Button(name='Назад', align='start', width=100, button_type='primary', disabled=True)
        self.back_button.on_click(self.on_click_back)

        self.stop = StopFlag()
        hparams = self.params.get('recommended_hparams', {})
        hparams.update({gui_params[k]['param_key']: v for k, v in self.params.items()
                        if gui_params.get(k, {}).get('param_from', '') == 'train'})
        self.process = process(train)(nn_task=self.params['task'], stop_flag=self.stop, hparams=hparams, start=False)
        self.process.set_handler('print', lambda *args, **kwargs: self.msg(*args, **kwargs))
        self.process.set_handler('train_callback', lambda *args, **kwargs: self.on_train_callback(*args, **kwargs))
        self.process.on_finish = lambda _: self.on_process_finish()
        print('Запуск процесса обучения')
        # create timer to call self.update function to update panel widgets

        self.timer = None
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.process.start()

    def update_plot(self, *args, **kwargs):
        # TODO: сделать здесь по-нормальному обновление графиков (через поток данных)
        if len(self.epochs) > self.last_epoch+1:
            ll = len(self.epochs)-1
            # update self.plot
            self.plot.line(list(self.epochs[self.last_epoch:]), list(self.losses[self.last_epoch:]), legend_label='Loss', line_color='red')
            self.plot.line(list(self.epochs[self.last_epoch:]), list(self.accuracies[self.last_epoch:]), legend_label='Accuracy', line_color='green')
            self.last_epoch = ll

    def msg(self, *args, file=None, end='\n', **kwargs):
        text = self.output.value
        if file is None:
            text += ''.join(map(str, args)) + end
            self.output.value = text
        else:
            print(*args, **kwargs, file=file, end=end)

    def on_train_callback(self, tp, batch=None, epoch=None, logs=None, model=None):
        if tp == 'epoch':
            self.msg(f'Эпоха {epoch}: {logs}')
            self.add_plot_point(epoch, logs['loss'], logs['acc'])
            time.sleep(0.1)
        elif tp == 'finish':
            self.msg('Обучение завершено')

    def on_click_stop(self, event):
        self.stop()
        self.stop_button.disabled = True

    def on_click_back(self,event):
        self.close()

    def panel(self):
        #self.output = pn.WidgetBox('### Output', min_width=500, height=500)
        self.output = pn.widgets.TextAreaInput(min_width=500, height=500, value='### Output', disabled=True)
        # create plot widget for loss and accuracy
        self.plot = bokeh.plotting.figure(title='Loss and Accuracy', x_axis_label='Epoch', y_axis_label='Loss/Accuracy',
                                          plot_width=500, plot_height=500)
        self.timer = bokeh.plotting.curdoc().add_periodic_callback(self.update_plot, 1000)
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.last_epoch = 0
        return pn.Column(
            '# Меню обучения модели',
            pn.Row(self.output, self.plot),
            pn.Card(tb.interface(), title="TensorBoard", collapsed=True),
            pn.Row(self.back_button, self.stop_button)
        )

    def add_plot_point(self, epoch, loss, accuracy):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def on_process_finish(self):
        self.back_button.disabled = False
        self.stop_button.disabled = True
        if self.timer is not None:
            self.plot.document.remove_periodic_callback(self.timer)
            self.timer = None



class TrainedModels(Window):

    next_window = param.Selector(default='Task', objects=['Task'])

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
        ('Database', Database),
        ('DatasetLoader', DatasetLoader),
        ('Task', Task),
        ('Params', Params),
        ('Training', Training),
        ('TrainedModels', TrainedModels)
    ],
    graph={
        'Database': ('DatasetLoader', 'Task'),
        'DatasetLoader': 'Database',
        'Task': ('Database', 'Params', 'TrainedModels'),
        'Params': ('Task', 'Training'),
        'Training': 'Task',
        'TrainedModels': 'Task'
    },
    root='Database',
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
