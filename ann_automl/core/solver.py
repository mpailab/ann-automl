import os
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing import Process

from pytz import timezone

from ..utils.process import request, NoHandlerError, pcall

_log_dir = '.'


def set_log_dir(log_dir):
    """
    Set the log directory. Log directory contains log files for all experiments.
    """
    global _log_dir
    _log_dir = log_dir


_rules = []  # список приёмов


def rule(cls):
    """
    Декоратор для классов приёмов. Добавляет класс приёма в список приёмов.
    """
    global _rules
    _rules.append(cls())
    return cls


class Task:
    """
    Решаемая задача.
    """

    def __init__(self, task_ct, task_type=None, obj_set=None, goal=None):
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
        self._taskCt = task_ct  # str

        # Необязательные входные параметры задачи, нужны при категориях "train" и "test"
        self._taskType = task_type  # str - тип задачи {classification, detection, segmentation}
        self._objects = obj_set  # set of strings - набор категорий объектов интереса
        self._goal = goal  # dictionary - {"метрика" : желаемое значение метрики }

    @property
    def taskCt(self):
        """ Возвращает категорию задачи """
        return self._taskCt

    @property
    def taskType(self):
        """ Возвращает тип задачи """
        return self._taskType

    @property
    def objects(self):
        """ Возвращает набор категорий объектов интереса """
        return self._objects

    @property
    def goal(self):
        """ Возвращает цели задачи """
        return self._goal

    def __str__(self) -> str:
        return f"Категория задачи:\n    {self.taskCt}\n" + \
            f"Тип задачи:\n    {self.taskType}\n" + \
            f"Категории объектов интереса:\n    {str(self.objects)}\n" + \
            f"Цели задачи:\n    {str(self.goal)}"


class State:
    """
    Состояние решателя
    """

    def __init__(self, task, log_name, res_name):
        """
        Инициализация состояния решателя.

        Parameters:
        ----------
        task: Task
            Задача
        logName: str
            Имя лог-файла
        """
        self.task = task  # решаемая задача
        self.curState = 'Initial'  # текущее состояние решателя
        self.logName = log_name
        self.resName = res_name

        # Атрибуты решателя,котоые используются для взаимодействия с пользователем
        # state.message =  str.format    #сообщение для пользователя
        # state.actions                  #что нужно делать в той или иной ситуации по выбору пользователя


class Rule(ABC):
    """
    Базовый тип приёма
    Имеет два абязательных метода can_apply и apply
    """

    @abstractmethod
    def can_apply(self, state: State):
        """
        Проверка целесообразности применения приёма

        Returns:
        -------
        bool
            Нужно ли применять приём
        """
        pass

    @abstractmethod
    def apply(self, state: State):
        """
        Применение приёма

        Parameters:
        ----------
        state: State
            Состояние решателя
        """
        pass


_first_print = _first_error = True


def printlog(*args, **kwargs):
    try:
        request('print', *args, **kwargs)
    except NoHandlerError:
        print(*args, **kwargs)

    global _first_print, _first_error
    if kwargs.get('file', None) is None:
        with open('log.txt', 'a' if not _first_print else 'w') as f:
            print(*args, **kwargs, file=f)
            _first_print = False
    elif kwargs.get('file') is sys.stderr:
        with open('err.txt', 'a' if not _first_error else 'w') as f:
            kwargs['file'] = f
            print(*args, **kwargs)
            _first_error = False


def solve(task: Task, rules=None, max_num_steps=500, debug_mode=False):
    """
    Parameters
    ----------
    task : Task
        Task to solve
    rules : list of Rule
        Rules to use for solving the task
    max_num_steps : int
        Maximum number of tries to apply rules for solving the task
    debug_mode : bool
        If True, prints debug information

    Returns
    -------
    State
        State of the solver after solving the task
    """
    try:
        rules = rules if rules is not None else _rules
        MSK = timezone('Europe/Moscow')
        msk_time = datetime.now(MSK)
        tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')

        # создаём директорию для логов
        log_name = f'{_log_dir}/Logs/Experiment_log_{tt}.txt'
        if not os.path.exists(f'{_log_dir}/Logs'):
            os.makedirs(f'{_log_dir}/Logs')
    
        # создаём директорию для логов
        res_name = f'{_log_dir}/Results/Results_log_{tt}.txt'
        if not os.path.exists(f'{_log_dir}/Results'):
            os.makedirs(f'{_log_dir}/Results')
        with open(res_name, 'w') as f:
            f.write('Results of Experiment ' + tt + '\n')
        num_steps = 0
    


        state = State(task, log_name)
        pcall("set_state", state)
        pos = 0
        if debug_mode:
            printlog([x.__class__.__name__ for x in rules])
        while state.curState != "Done":
            num_steps += 1
            with open(log_name, 'a') as log_file:
                log_file.write(f'{state.curState}\n')
            if debug_mode:
                printlog(f'{num_steps}. Состояние: {state.curState}')
            if rules[pos].can_apply(state):
                if debug_mode:
                    printlog(f'{num_steps}. Применяем правило {rules[pos].__class__.__name__}')
                rules[pos].apply(state)
            pos = (pos + 1) % len(rules)
            if num_steps > max_num_steps:
                printlog(f'Превышено максимальное число шагов ({max_num_steps})')
            break

        return state
    except Exception as e:
        printlog(f'Ошибка при решении задачи: {e}', file=sys.stderr)
        printlog(traceback.format_exc(), file=sys.stderr)
        raise
