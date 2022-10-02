from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing import Process

from pytz import timezone


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


class State:
    """
    Состояние решателя
    """

    def __init__(self, task, log_name):
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
        self.curState = 'FirstCheck'  # текущее состояние решателя
        self.logName = log_name

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

    Returns
    -------
    State
        State of the solver after solving the task
    """
    rules = rules if rules is not None else _rules
    MSK = timezone('Europe/Moscow')
    msk_time = datetime.now(MSK)
    tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')

    log_name = './Logs/Experiment_log_' + tt + '.txt'
    num_steps = 0

    state = State(task, log_name)
    pos = 0
    if debug_mode:
        print(rules)
    while state.curState is not "Done":
        with open(log_name, 'a') as log_file:
            print(state.curState, file=log_file)
        if debug_mode:
            print(state.curState)
        if rules[pos].can_apply(state):
            # print(rules[pos])
            rules[pos].apply(state)
        pos = (pos + 1) % len(rules)
        num_steps += 1
        if num_steps > max_num_steps:
            break

    return state


def start_solving(task, rules=None, max_num_steps=500):
    """
    Запускает решение задачи.

    Parameters:
    ----------
    task: Task
        Задача

    Returns:
    -------
    Дескриптор процесса решения задачи
    """
    return Process(target=solve, args=(task,), kwargs={'rules': rules, 'max_num_steps': max_num_steps})
