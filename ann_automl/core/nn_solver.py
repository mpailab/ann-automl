import os
import sys
import traceback
from copy import copy
from datetime import datetime
from pytz import timezone

from .nnfuncs import nn_hparams, tune_hparams, get_hparams
from .solver import Task, set_log_dir, printlog, _log_dir, RecommendTask
from ..utils.process import pcall


class NNTask(Task):
    """ Класс, задающий задачу на обучение нейронной сети """

    def __init__(self, task_ct, task_type=None, objects=None, goals=None):
        """
        Инициализация задачи.

        Parameters:
        ----------
        task_ct:  str
            Категория задачи: обучение модели "train", тестирование модели без предварительного обучения "test"
            "служебные" = {проверка наличия нужных классов,БД, моделей}
        task_type: str
            Тип задачи {classification, detection, segmentation}
        obj_set: list[str]
            Набор категорий объектов интереса
        goal: dict[str, Any]
            Словарь целей задачи (например {"метрика" : желаемое значение метрики })
        """
        super().__init__(goals=goals)
        self._task_ct = task_ct  # str

        # Необязательные входные параметры задачи, нужны при категориях "train" и "test"
        self._task_type = task_type  # str - тип задачи {classification, detection, segmentation}
        self._objects = objects      # set of strings - набор категорий объектов интереса
        self.input_type = 'image'    # тип объектов задачи (image, video, audio, text)
        self.object_category = 'object'  # категория объектов задачи (object, symbol)
        self.log_name = ''
        self.cur_state = 'FirstCheck'

    @property
    def task_ct(self):
        """ Возвращает категорию задачи """
        return self._task_ct

    @property
    def task_type(self):
        """ Возвращает тип задачи """
        return self._task_type

    @property
    def objects(self):
        """ Возвращает набор категорий объектов интереса """
        return self._objects

    def solve(self, solver_state=None, rules=None, state=None, global_params=None, max_num_steps=1000):
        rules, solver_state = self._init_solver_state(solver_state, global_params, rules)
        debug_mode = self.solver_state.global_params.get('debug_mode', False)
        try:
            MSK = timezone('Europe/Moscow')
            msk_time = datetime.now(MSK)
            tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')

            # создаём директорию для логов
            self.log_name = f'{_log_dir}/Logs/Experiment_log_{tt}.txt'
            if not os.path.exists(f'{_log_dir}/Logs'):
                os.makedirs(f'{_log_dir}/Logs')
            num_steps = 0

            pcall("set_state", state)
            pos = 0
            if debug_mode:
                printlog([x.__class__.__name__ for x in rules])
            while self.cur_state != "Done":
                num_steps += 1
                printlog(f'{num_steps}. Состояние: {self.cur_state}')
                if rules[pos].can_apply(state, solver_state):
                    if debug_mode:
                        printlog(f'{num_steps}. Применяем правило {rules[pos].__class__.__name__}')
                    rules[pos].apply(state, solver_state)
                pos = (pos + 1) % len(rules)
                if num_steps > max_num_steps:
                    printlog(f'Превышено максимальное число шагов ({max_num_steps})')
                    break
            return state

        except Exception as e:
            printlog(f'Ошибка при решении задачи: {e}', file=sys.stderr)
            printlog(traceback.format_exc(), file=sys.stderr)
            raise


class SelectHParamsTask(RecommendTask):
    def __init__(self, nn_task: NNTask):
        super().__init__(goals={})
        self.nn_task = nn_task
        self.hparams = {param: nn_hparams[param]['default'] for param in nn_hparams}
        self.hparams['pipeline'] = None
        self.recommendations = {}

    def set_selected_options(self, options):
        self.hparams.update(options)


def recommend_hparams(task: NNTask, **kwargs) -> dict:
    """ Рекомендует гиперпараметры для задачи """
    htask = SelectHParamsTask(task)
    htask.solve(global_params=kwargs)
    return htask.hparams
