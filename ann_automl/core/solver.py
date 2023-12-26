import sys
from abc import ABC, abstractmethod
from collections import defaultdict

from ..utils.process import request, NoHandlerError

_log_dir = '.'


def set_log_dir(log_dir):
    """
    Set the log directory. Log directory contains log files for all experiments.
    """
    global _log_dir
    _log_dir = log_dir


_rules = []  # список приёмов


_first_print = _first_error = True


def printlog(*args, **kwargs):
    """
    Выводит сообщение в лог. Аргументы аналогичны функции print.
    Если сообщение печатается в файл, то действие эквивалентно `print(*args, **kwargs)`

    Если сообщение печатается в консоль, то дополнительно:
     - сообщение печатается в файл log.txt (stdout) или err.txt (stderr)
     - вызавается функция request('print', ...), которая быть привязана к другим
        действиям при запуске через Web-интерфейс.
    """
    try:
        request('print', *args, **kwargs)
    except NoHandlerError:
        print(*args, **kwargs)

    global _first_print, _first_error
    if kwargs.get('file', None) is None:
        with open('log.txt', 'a' if not _first_print else 'w', encoding='utf-8') as f:
            print(*args, **kwargs, file=f)
            _first_print = False
    elif kwargs.get('file') is sys.stderr:
        with open('err.txt', 'a' if not _first_error else 'w') as f:
            kwargs['file'] = f
            print(*args, **kwargs)
            _first_error = False


class CannotSolve(Exception):
    """ Исключение, которое выбрасывается, если решение не найдено. """
    pass


class FilterFailed(Exception):
    """ Исключение, которое выбрасывается, если фильтр не прошёл (используется для внутренних целей). """
    pass


def ensure(cond: bool):
    """ Проверяет, выполняется ли условие. """
    if not cond:
        raise FilterFailed()


def defined(x):
    """ Проверяет, определена ли переменная. """
    if x is None:
        raise FilterFailed()


class Task:
    """
    Базовый тип задачи
    """

    def __init__(self, goals):
        self.goals = dict(goals or ())    # множество целей
        self.solved = False        # Флаг, что задача решена
        self._state = self         # Состояние решения задачи
        self._answer = None        # Ответ задачи
        self._solver_state = None  # Глобальное состояние решателя
        self.stages = []           # Список этапов решения задачи

    def _init_solver_state(self, solver_state, global_params, rules):
        if rules is None:
            rules = self.rules
        else:
            rules += self.rules

        if solver_state is None:
            solver_state = SolverState(self, global_params=global_params or {})

        self._solver_state = solver_state
        return rules, solver_state

    def solve(self, solver_state=None, rules=None, state=None, global_params=None):
        """
        Базовая функция решения задачи

        :param solver_state: Глобальное состояние решателя
            (если нет, то начальное состояние инициализируется по умолчанию)
        :param rules: Дополнительные правила для решения данной задачи
        :param state: Состояние решения общей задачи (может передаваться в подзадачи)
        :param global_params: Глобальные параметры решателя (флаги трассировки, отладки и т.п.)

        :returns: Ответ задачи
        """
        rules, solver_state = self._init_solver_state(solver_state, global_params, rules)

        if self._solver_state.global_params.get('trace_solution', False):
            printlog(f'##########  Start solve task of type {self.__class__.__name__}  ##########')

        stages = self.stages or [0]
        if not isinstance(rules, dict):
            rules = {stage: rules for stage in stages}

        nstate = self.prepare_state(state)
        if state is None:
            state = nstate
        self._state = state if state is not None else self

        for stage in stages:
            if self._solver_state.global_params.get('trace_solution', False):
                printlog(f'##########  Start stage {stage}  ##########')
            applied = 1
            while applied:
                applied = 0
                for r in rules.get(stage, []):
                    if r.can_apply(self, self._state):
                        if self._solver_state.global_params.get('trace_solution', False):
                            printlog(f'    Apply rule {r.__class__.__name__} ')
                        r.apply(self, self._state)
                        applied += 1
            # if not applied:
            #    raise CannotSolve(self)

        if self._solver_state.global_params.get('trace_solution', False):
            printlog(f'##########  Finish solve task of type {self.__class__.__name__}  #########')

        return self.answer

    @property
    def answer(self): return self._answer

    @answer.setter
    def answer(self, ans):
        self._answer = ans
        if ans is not None:
            self.solved = True

    @property
    def state(self):
        """ Текущее состояние решения задачи (по умолчанию объект состояния и объект задачи совпадают) """
        return self._state

    @property
    def solver_state(self): return self._solver_state

    def run_subtask(self, subtask, state=None):
        """ Запуск решения подзадачи. Обёртка вокруг SolverState.run_subtask. """
        return self.solver_state.run_subtask(subtask, state)

    def prepare_state(self, state):
        """
        Подготовка состояния перед началом решения задачи.
        Может также создавать новое состояние, если state=None
        """
        return state

    @property
    def rules(self):
        """ Правила, отнесённые к задачам данного типа, включая все правила для базовых классов задач """
        r = {}
        prev = None
        for cls in self.__class__.mro():
            rules = getattr(cls, '_rules', {})
            if rules is not prev:
                for stage, rules in getattr(cls, '_rules', {}).items():
                    r.setdefault(stage, []).extend(rules)
                prev = rules
        return r


def rule(*task_classes, stage=None):
    """
    Декоратор для классов приёмов.
    Добавляет данный приём к типам задач, указанных в качестве параметров декоратора.

    Args:
        task_classes: Классы задач, к которым применяется данный приём
        stage: Этап или список этапов, на которых применяется данный приём (если None, то применяется на всех этапах)
    """
    if len(task_classes) == 0:
        task_classes = [Task]  # По умолчанию добавляем глобальный приём

    def apply(rule_cls):
        instance = rule_cls()
        for task_cls in task_classes:
            if not hasattr(task_cls, '_rules') or task_cls._rules is getattr(task_cls.__base__, '_rules', None):
                task_cls._rules = {}
            stages = ([] if stage is None else [stage]) if not isinstance(stage, (list, tuple, set)) else list(stage)
            for s in getattr(instance, 'stages', ()):
                if s not in stages:
                    stages.append(s)
            if not stages:
                stages = [None]
            for s in stages:
                task_cls._rules.setdefault(s, [])
                task_cls._rules[s].append(instance)
        return instance

    return apply


class TaskState:
    """
    Общий класс для состояния решения задачи, который можно использовать,
     если неудобно хранить состояние в самом классе задачи
    """
    pass


class SolverState:
    """
    Класс, хранящий глобальное состояние решателя:
     - стек задач
     - глобальные параметры
    """
    def __init__(self, task=None, global_params=None):
        # self.answer = None  # ответ на решение задачи
        # self.task = task  # текущая решаемая задача
        self._task_stack = [task] if task is not None else []  # стек задач
        # self._state_stack = []
        self._global_params = {} if global_params is None else global_params  # глобальные параметры решателя

    @property
    def task_stack(self): return self._task_stack

    @property
    def global_params(self): return self._global_params

    def run_subtask(self, subtask: Task, state: TaskState = None):
        self._task_stack.append(subtask)
        ans = subtask.solve(solver_state=self, state=state)
        self._task_stack.pop()
        return ans


class Rule(ABC):
    """
    Базовый тип приёма

    Имеет два абязательных метода can_apply и apply
    """

    @abstractmethod
    def can_apply(self, task, state) -> bool:
        """
        Проверка целесообразности применения приёма

        Args:
            task (Task): Задача, для которой проверяется целесообразность применения приёма
            state (SolverState): Глобальное состояние решателя
        Returns:
            Нужно ли применять приём
        """
        pass

    @abstractmethod
    def apply(self, task, state):
        """
        Применение приёма

        Args:
            task (Task): Задача, для которой применяется приём
            state (SolverState): Глобальное состояние решателя
        """
        pass


class RuleFL(Rule):
    """
    Класс приёмов, в которых фильтры можно определять в декларативном стиле
    """
    @abstractmethod
    def filter(self, task, state) -> None:
        """
        Проверка целесообразности применения приёма в декларативном виде.
        В случае невыполнения одного из условий должна бросать исключение FilterFailed
        Успешное завершение этой функции означает, что приём можно применять
        """
        pass

    def can_apply(self, task: Task, state: SolverState):
        try:
            self.filter(task, state)
            return True
        except FilterFailed:
            return False


class FinishTask(RuleFL, ABC):
    """
    Базовый класс для приёмов, которые проверяют, что задача решена
    """
    def apply(self, task: Task, state):
        task.solved = True


class RecommendTask(Task):
    """
    Класс для задач, в которых нужно дать рекомендации по выбору каких-либо параметров в зависимости от условий
    """
    def __init__(self, goals):
        super().__init__(goals=goals)
        self.recommendations = {}
        self.votes = defaultdict(lambda: 0)
        self.stages = ['Recommend', 'Vote', 'Select', 'Finalize']

    def set_selected_options(self, options):
        self.answer = options


class Recommender(Rule, ABC):
    """
    Базовый класс для приёмов, которые рекомендуют какие-либо параметры в задаче RecommendTask
    """
    stages = ['Recommend']

    def __init__(self):
        self.key = self.__class__.__name__

    def can_recommend(self, task) -> bool:
        return True

    def can_apply(self, task: RecommendTask, state: SolverState) -> bool:
        return self.key not in task.recommendations and self.can_recommend(task)


class VoteRule(Rule, ABC):
    """
    Базовый класс для приёмов, которые голосуют за какие-либо параметры в задаче RecommendTask
    """
    stages = ['Vote']

    def __init__(self):
        self.key = self.__class__.__name__

    def can_vote(self, task) -> bool:
        return True

    def can_apply(self, task: RecommendTask, state: SolverState) -> bool:
        return self.key not in task.votes and self.can_vote(task)


def _to_immutable(val):
    if isinstance(val, (str, bytes)):
        return val
    if isinstance(val, dict):
        return tuple((k, _to_immutable(v)) for k, v in val.items())
    if hasattr(val, '__iter__'):
        return tuple(map(_to_immutable, val))
    return val


@rule(RecommendTask)
class SelectRecommendation(Rule):
    """
    Приём, который выбирает окончательные рекомендации на основе голосования
    """
    stages = ['Finalize']

    def can_apply(self, task: RecommendTask, state: SolverState) -> bool:
        return len(task.recommendations) > 0 and task.answer is None and not task.solved

    def apply(self, task: RecommendTask, state: SolverState):
        keys = {k for rn, rec in task.recommendations.items() for k, v in rec.items()}
        res = {}
        to_source = {}
        for k in keys:
            votes = defaultdict(lambda: 0.0)
            for rn, rec in task.recommendations.items():
                if k in rec:
                    immut = _to_immutable(rec[k])
                    votes[immut] += task.votes.get(rn, 1)
                    to_source[immut] = rec[k]
            res[k] = to_source[max(votes.items(), key=lambda x: x[1])[0]]
        task.set_selected_options(res)
        task.solved = True
