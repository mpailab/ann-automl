#####################################################################
# Adjustable hyperparameters for the neural network training process
# This dictionary contains types of hyperparameters and their range or list of values, and default values
import math
from collections import defaultdict

import ipywidgets
from IPython.core.display import display

from ..core.solver import solve, Task
from ..utils.process import VarWaiter, process

# !!! гиперпараметры и их значения сгенерированы автоматически !!!
# TODO: проверить их на корректность
hyperparameters = {
    'batch_size': {'type': 'int', 'range': [1, 128], 'default': 32, 'exponential_step': 2, 'name': "размер батча"},
    'epochs': {'type': 'int', 'range': [1, 100], 'default': 10, 'step': 1, 'name': "количество эпох"},
    'optimizer': {'type': 'str', 'values': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta',
                                            'Adam', 'Adamax', 'Nadam'], 'default': 'Adam', 'name': "оптимизатор"},
    'learning_rate': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'exponential_step': 10**0.1,
                      'name': "скорость обучения"},
    'momentum': {'type': 'float', 'range': [0, 1], 'default': 0.9, 'name': 'момент для SGD',
                 'enable_if': {'optimizer': 'SGD'}},  # момент для SGD
    'decay': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'name': 'декремент скорости обучения'},
    'nesterov': {'type': 'bool', 'default': False, 'name': 'Nesterov momentum',
                 'enable_if': {'optimizer': 'SGD'}},  # использовать ли Nesterov momentum (только для SGD)
    'rho': {'type': 'float', 'range': [0, 1], 'default': 0.9, 'name': 'rho для RMSprop',
            'enable_if': {'optimizer': 'RMSprop'}},
    'epsilon': {'type': 'float', 'range': [1e-8, 1e-1], 'default': 1e-8, 'exponential_step': 10**0.1,
                'name': 'epsilon для RMSprop', 'enable_if': {'optimizer': 'RMSprop'}},  # epsilon для RMSprop
    'beta_1': {'type': 'float', 'range': [0, 1], 'default': 0.9, 'name': 'beta_1 для Adam',
               'enable_if': {'optimizer': 'Adam'}},
    'beta_2': {'type': 'float', 'range': [0, 1], 'default': 0.999, 'name': 'beta_2 для Adam',
               'enable_if': {'optimizer': 'Adam'}},
    'amsgrad': {'type': 'bool', 'default': False, 'name': 'amsgrad для Adam',
                'enable_if': {'optimizer': 'Adam'}},  # использовать ли amsgrad для Adam
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
    'dropout': {'type': 'float', 'range': [0, 1], 'default': 0.0, 'name': 'dropout'},  # доля нейронов, которые отключаются при обучении
    'kernel_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                     'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                     'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                     'he_uniform'],
                           'default': 'glorot_uniform', 'name': 'инициализатор весов'},
    'bias_initializer': {'type': 'str', 'values': ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform',
                                                   'truncated_normal', 'orthogonal', 'identity', 'lecun_uniform',
                                                   'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal',
                                                   'he_uniform'], 'default': 'zeros', 'name': 'инициализатор смещений'},
    'kernel_regularizer': {'type': 'str', 'values': ['disabled', 'l1', 'l2', 'l1_l2'],
                           'default': 'disabled', 'name': 'регуляризатор весов'},
    'bias_regularizer': {'type': 'str', 'values': ['disabled', 'l1', 'l2', 'l1_l2'],
                         'default': 'disabled', 'name': 'регуляризатор смещений'},
    'activity_regularizer': {'type': 'str', 'values': ['disabled', 'l1', 'l2', 'l1_l2'],
                             'default': 'disabled', 'name': 'регуляризатор активации'},
    'kernel_constraint': {'type': 'str', 'values': ['disabled', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                          'default': 'disabled', 'name': 'ограничение весов'},
    'bias_constraint': {'type': 'str', 'values': ['disabled', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm'],
                        'default': 'disabled', 'name': 'ограничение смещений'},
}


# ipywidget для выбора гиперпараметров
# Функция, создающая ipywidget для задания гиперпараметра по его описанию
def create_hyperparameter_widget(description):
    name = description['name']
    label = ipywidgets.Label(name)
    tp = description['type']

    def get_value(value):
        return value
    def set_val(value):
        return value

    value_label = ipywidgets.Label()

    if tp in ['int', 'float']:
        is_exp = 'exponential_step' in description
        if is_exp:
            step = 1
            mn, mx = description['range']
            exp_step = description['exponential_step']
            rng = (0, int(math.ceil(math.log(mx / mn, exp_step))))
            def_value = math.log(description['default'] / mn, exp_step)
            if tp == 'int':
                def_value = int(round(def_value))
                def get_value(value):
                    return min(int(round(mn * exp_step**value+0.5)), mx)
                def set_val(value):
                    return max(min(int(round(math.log(value / mn, exp_step))), rng[1]), rng[0])
            else:
                def get_value(value):
                    return min(mn * exp_step**value, mx)
                def set_val(value):
                    return max(min(math.log(value / mn, exp_step), rng[1]), rng[0])
        else:
            rng = description['range']
            def_value = description['default']
            step = description.get('step', 1 if description['type'] == 'int' else 0.01*(rng[1] - rng[0]))

        if tp == 'int':
            def set_value(value):
                value_label.value = str(get_value(value))
            control = ipywidgets.IntSlider(
                value=def_value,
                min=rng[0],
                max=rng[1],
                step=step,
                description='',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=False,
            )
        else:  # in this case description['type'] == 'float'
            def set_value(value):
                if is_exp:
                    value_label.value = f'{get_value(value):.2e}'
                else:
                    step_log = int(math.floor(math.log10(step)))
                    if step_log < 0:
                        value_label.value = f'{get_value(value):.{-step_log}f}'
                    else:
                        value_label.value = f'{get_value(value):.0f}'

            control = ipywidgets.FloatSlider(
                value=def_value,
                min=rng[0],
                max=rng[1],
                step=step,
                description='',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=False,
            )
        control.observe(lambda change: set_value(change['new']), names='value')
        getter = lambda: get_value(control.value)
        setter = lambda value: control.set_trait('value', set_val(value))
        set_value(def_value)
    else:
        if tp == 'str':
            control = ipywidgets.Dropdown(
                options=description['values'],
                value=description['default'],
                description='',
                disabled=False,
            )
        elif tp == 'bool':
            control = ipywidgets.Checkbox(
                value=description['default'],
                description='',
                disabled=False,
            )
        else:
            raise ValueError('Unknown hyperparameter type: {}'.format(description['type']))
        getter = lambda: control.value
        setter = lambda value: control.set_trait('value', value)
    return [label, control, value_label], getter, setter


def create_check_enabled(enable_if, w, getters, param, en_set):
    enable_if = list(enable_if.items())

    def check_enabled():
        enabled = param in en_set[0]
        for p, val in enable_if:
            enabled = enabled and getters[p]() == val
        # for w in line:
        w.layout.display = None if enabled else 'none'

    return check_enabled


def create_checks(checks):
    def check_i(_):
        for c in checks:
            c()

    return check_i


def create_enable_func(w):
    def enable_func(en):
        w.layout.display = None if en else 'none'

    return enable_func


# Функция, создающая ipywidget для задания гиперпараметров
def create_hyperparameters_widget(hyperparameters_description, selected_hyperparameters):
    subwidgets = []
    getters = {}
    setters = {}
    w = {}
    control_other = defaultdict(lambda: [])
    val_widgets = []
    enabled = [set()]

    cond, noncond = [], []
    for p in selected_hyperparameters:
        if 'enable_if' in hyperparameters_description[p]:
            cond.append(p)
        else:
            noncond.append(p)

    selected = noncond + [''] + cond

    # Make parameter description aligned in all widgets
    for p in selected:
        if p == '':
            if cond:
                dop = ipywidgets.Label(value='Дополнительные параметры',
                                       layout=ipywidgets.Layout(display='flex',
                                                                justify_content='center',
                                                                width='100%'))
                subwidgets.append(dop)
            continue
        description = hyperparameters_description[p]
        line, getters[p], setters[p] = create_hyperparameter_widget(description)
        widget = ipywidgets.GridspecLayout(1, 3)
        widget[0, 0] = line[0]
        widget[0, 1] = line[1]
        widget[0, 2] = line[2]
        val_widgets.append(line[1])
        subwidgets.append(widget)
        w[p] = widget
        if 'enable_if' in description:
            for q in description['enable_if']:
                control_other[q].append(create_check_enabled(description['enable_if'], widget, getters, p, enabled))

    all_checks = []
    for i, subwidget in enumerate(subwidgets):
        checks = tuple(control_other[selected[i]])
        if checks:
            check_i = create_checks(checks)
            val_widgets[i].observe(check_i, names='value')
            check_i(None)
            all_checks.append(check_i)

    def enable(en):
        enabled[0] = set(en)
        en_dop = False
        for p, pw in w.items():
            pw.layout.display = None if p in enabled[0] else 'none'
            if p in cond:
                en_dop = en_dop or p in enabled[0]
        if cond:
            dop.layout.display = None if en_dop else 'none'
        for check in all_checks:
            check(None)

    return ipywidgets.VBox(subwidgets), getters, setters, enable


def create_checkboxes(hp=None):
    """Создание чекбоксов для выбора гиперпараметров"""
    checkboxes = []
    enable = {}
    if hp is None:
        hp = hyperparameters.keys()

    hp = tuple(hp)

    for h in hp:
        name = hyperparameters[h]['name']
        checkboxes.append(ipywidgets.Checkbox(value=True, description=name))
        enable[h] = create_enable_func(checkboxes[-1])

    def get_checked():
        return [h for h, c in zip(hp, checkboxes) if c.value]

    def set_enabled(h):
        h = set(h)
        for c in hp:
            enable[c](c in h)

    return ipywidgets.VBox(checkboxes), get_checked, set_enabled


class SolverGUI:
    def create_main_screen(self):
        style = {'description_width': '200px', 'width': '500px'}
        ct_in = ipywidgets.Dropdown(
            options=['train', 'test', 'database', 'history'],
            value='train',
            description='Категория задачи:',
            style=style
        )

        type_in = ipywidgets.Dropdown(
            options=['classification', 'segmentation', 'detection'],
            value='classification',
            description='Тип задачи:',
            style=style
        )

        cat_in = ipywidgets.Text(
            value='cat, dog',
            description='Категория объектов интереса:',
            style=style
        )

        target_in = ipywidgets.Dropdown(
            options=['loss', 'metrics'],
            value='metrics',
            description='Цель:',
            style=style
        )

        target_val_in = ipywidgets.BoundedFloatText(
            value=0.7,
            min=0,
            max=1,
            step=0.05,
            description='Значение цели:',
            style=style
        )

        def on_finish(p):
            start.disabled = False
            if not isinstance(p._exn, Exception):
                self.tab.selected_index = 4
                print(f'\n======== Process finished ========')

        def on_start_clicked(b):
            self._log_widget.clear_output()
            with self._log_widget:
                print("========= Process output ==========")
            obj = [x.strip() for x in str(cat_in.value).split(",")]
            task = Task(ct_in.value, task_type=type_in.value, obj_set=obj, goal={target_in.value: target_val_in.value})
            self.proc = process(solve)(task, debug_mode=True, start=False, output_context=self._log_widget)
            self.proc.handlers.update(self.handlers)
            self.proc.start()
            self.proc.on_finish = on_finish
            start.disabled = True

        start = ipywidgets.Button(description='Запустить систему',
                                  # style = {'button_color': 'rgb(0,140,255)'},
                                  layout=ipywidgets.Layout(width='500px', height='28px'))
        start.on_click(on_start_clicked)

        return ipywidgets.VBox([ct_in, type_in, cat_in, target_in, target_val_in, start])

    def create_strategy_found(self):
        a11 = ipywidgets.Dropdown(
            options=[('Стратегия 1', 1), ('Стратегия 2', 2), ('Стратегия 3', 3)],
            value=1,
            description='Найдены подходящие стратегии:',
            style={'description_width': '200px', 'width': '600px'}
        )

        btn_yes = ipywidgets.Button(description='Да')
        btn_no = ipywidgets.Button(description='Нет')

        def on_btn_yes(b):
            self.tab.selected_index = 4

        def on_btn_no(b):
            self.tab.selected_index = 0

        btn_yes.on_click(on_btn_yes)
        btn_no.on_click(on_btn_no)

        HB = ipywidgets.HBox(
            [ipywidgets.Label('Хотите обучить новую модель или закончить работу? : '), btn_yes, btn_no])
        return ipywidgets.VBox([a11, HB])

    def create_retrain_widget(self):
        style = {}  # 'button_color': 'rgb(0,140,255)'}
        c1 = ipywidgets.Button(
            description='Выбрать другую стратегию (из истории системы)',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='500px', height='28px'),
            style=style
        )
        c2 = ipywidgets.Button(
            description='Вручную задать параметры обучения',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='500px', height='28px'),
            style=style
        )
        c3 = ipywidgets.Button(
            description='Обучение по сетке',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='500px', height='28px'),
            style=style
        )
        c4 = ipywidgets.Button(
            description='Изменить архитектуру модели (случайно\по истории\по спец. технологии)',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='500px', height='28px'),
            style=style
        )
        c5 = ipywidgets.Button(
            description='изменить расписание скорости обучения',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='500px', height='28px'),
            style=style
        )
        d1 = ipywidgets.VBox([ipywidgets.Label('Приёмы:'), c1, c2, c3, c4, c5])
        e1 = ipywidgets.FloatText(
            value=0.5,
            disabled=False,
            layout=ipywidgets.Layout(width='80px', height='28px')
        )
        e2 = ipywidgets.FloatText(
            value=0.5,
            disabled=False,
            layout=ipywidgets.Layout(width='80px', height='28px')
        )
        e3 = ipywidgets.FloatText(
            value=0.5,
            disabled=False,
            layout=ipywidgets.Layout(width='80px', height='28px')
        )
        e4 = ipywidgets.FloatText(
            value=0.5,
            disabled=False,
            layout=ipywidgets.Layout(width='80px', height='28px')
        )
        e5 = ipywidgets.FloatText(
            value=0.5,
            disabled=False,
            layout=ipywidgets.Layout(width='80px', height='28px')
        )
        d2 = ipywidgets.VBox([ipywidgets.Label('Текущее распределение вероятности «успеха»'), e1, e2, e3, e4, e5])
        d3 = ipywidgets.HBox([d1, d2])
        c6 = ipywidgets.Button(
            description='Случайный прием',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='200px', height='50px'),
            # style = {'button_color': 'rgb(204,0,255)'}
        )
        c7 = ipywidgets.Button(
            description='Закончить обучение',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='200px', height='50px'),
            # style = {'button_color': 'rgb(255,119,0)'}
        )
        d4 = ipywidgets.HBox([c6, c7])
        d5 = ipywidgets.VBox([d3, d4])
        return d5

    def create_manual_params(self):
        a7 = ipywidgets.Dropdown(
            options=[('classification', 1), ('segmentation', 2), ('detection', 3)],
            value=2,
            description='Параметры базы данных:',
            style={'description_width': '200px', 'width': '500px'}
        )
        a8 = ipywidgets.Dropdown(
            options=[('classification', 1), ('segmentation', 2), ('detection', 3)],
            value=2,
            description='Параметры обучения:',
            style={'description_width': '200px', 'width': '500px'}
        )
        a9 = ipywidgets.Dropdown(
            options=[('classification', 1), ('segmentation', 2), ('detection', 3)],
            value=2,
            description='Архитектура модели:',
            style={'description_width': '200px', 'width': '500px'}
        )
        a10 = ipywidgets.Button(
            description='Прекратить обучение',
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='200px'),
            # style = {'button_color': 'rgb(0,178,255)'}
        )
        return ipywidgets.VBox([a7, a8, a9, a10])

    def create_train_choose(self):
        manual = ipywidgets.Button(
            description='Вручную задать параметры обучения',
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='640px'),
            # style = {'button_color': 'rgb(255,255,0)'}
        )
        history = ipywidgets.Button(
            description='Обучить одну модель с т.зр. лучшей стратегии по сохраненной в системе истории обучения моделей',
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='640px'),
            # style = {'button_color': 'rgb(255,255,0)'}
        )
        grid = ipywidgets.Button(
            description='Обучение по сетке',
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=ipywidgets.Layout(width='640px'),
            # style = {'button_color': 'rgb(255,255,0)'}
        )

        buttons = ipywidgets.VBox([manual, history, grid])

        def create_switch_func(w1, w2):
            def switch(_):
                w1.layout.display = 'none'
                w2.layout.display = None

            return switch

        hp, getters, setters, params_en = create_hyperparameters_widget(hyperparameters, list(hyperparameters.keys()))
        set_params = ipywidgets.Button(description='Задать параметры')
        hp_back = ipywidgets.Button(description='Назад')
        params = ipywidgets.VBox([hp, ipywidgets.HBox([set_params, hp_back])])
        params.layout.display = 'none'

        manual.on_click(create_switch_func(buttons, params))
        hp_back.on_click(create_switch_func(params, buttons))

        gp, grid_getter, grid_en = create_checkboxes()
        set_grid = ipywidgets.Button(description='Задать параметры')
        gp_back = ipywidgets.Button(description='Назад')
        grid_params = ipywidgets.VBox([gp, ipywidgets.HBox([set_grid, gp_back])])
        grid_params.layout.display = 'none'

        grid.on_click(create_switch_func(buttons, grid_params))
        gp_back.on_click(create_switch_func(grid_params, buttons))

        all_buttons = [hp_back, set_grid, gp_back, set_params, manual, history, grid]

        def choose_strategy(current_hyperparameters, history_available, all_parameters):
            params.layout.display = 'none'
            grid_params.layout.display = 'None'
            buttons.layout.display = None
            self.tab.selected_index = 2
            for b in all_buttons:
                b.disabled = False

            for p, val in current_hyperparameters:
                if p in setters:
                    setters[p](val)

            wait = VarWaiter()

            def close():
                set_params.on_click(manual_entered, remove=True)
                history.on_click(history_entered, remove=True)
                set_grid.on_click(grid_entered, remove=True)
                for b in all_buttons:
                    b.disabled = True

            def manual_entered(_):
                close()
                new_params = {p: get() for p, get in getters.items() if p in all_parameters}
                result = current_hyperparameters
                result.update(new_params)
                wait.value = ('manual', result)

            def history_entered(_):
                close()
                wait.value = ('from_history', None)

            def grid_entered(_):
                close()
                wait.value = ('grid_search', grid_getter())

            params_en(all_parameters)
            grid_en(all_parameters)

            set_params.on_click(manual_entered)
            history.disabled = not history_available
            history.on_click(history_entered)
            set_grid.on_click(grid_entered)

            return wait.wait_value()

        self.handlers['choose_hyperparameters'] = choose_strategy

        return ipywidgets.VBox([buttons, params, grid_params])

    def create_final_screen(self):
        label = ipywidgets.Label('Задача завершена. Результаты обучения:')
        state = ipywidgets.Label("TODO: Добавить сюда информацию о результатах обучения")
        return ipywidgets.VBox([label, state])

    def __init__(self, ptask):
        self._log = []  # логи
        self._ptask = ptask
        self.handlers = {}
        #######################
        self._main_screen = self.create_main_screen()
        self._strategy_found = self.create_strategy_found()
        self._train_choose = self.create_train_choose()
        # self._manual_train_params = self.create_manual_params()
        self._retrain_widget = self.create_retrain_widget()
        self._final_screen = self.create_final_screen()
        self._log_widget = ipywidgets.Output(
            layout=ipywidgets.Layout(height='200px', border='solid', overflow_y='scroll'))

        def set_state(state):
            self.state = state

        self.handlers["set_state"] = set_state
        #######################
        screen1 = self._main_screen
        screen2 = self._strategy_found
        # screen3 = self._manual_train_params
        screen4 = self._train_choose
        screen5 = self._retrain_widget
        self.tab = ipywidgets.Tab()
        self.tab.children = [screen1, screen2, screen4, screen5, self._final_screen]
        self.tab.titles = [str(i) for i in range(len(self.tab.children))]

        #######################
        clear_button = ipywidgets.Button(description='Очистить выдачу')

        def clear_log(b):
            self._log_widget.clear_output()

        clear_button.on_click(clear_log)

        def printlog(*args, **kwargs):
            with self._log_widget:
                print(*args, **kwargs)

        self.handlers['print'] = printlog

        self.window = ipywidgets.VBox([self.tab, self._log_widget, clear_button])
        for i in range(5):
            title = 'Screen' + str(i + 1)
            self.tab.set_title(i, title)

    def display(self):
        display(self.window)
