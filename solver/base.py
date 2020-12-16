from abc import ABC, abstractmethod


class Task:
    """
    Базовый тип задачи
    """

    def __init__(self, goals):
        self.goals = set(goals)    # множество целей
        self.solved = False        # Флаг, что задача решена
        self._state = self         # Состояние решения задачи
        self._answer = None        # Ответ задачи
        self._solver_state = None  # Глобальное состояние решателя

    def solve(self, solver_state=None, rules=None, state=None, global_params=None):
        """
        Базовая функция решения задачи
        :param solver_state: Глобальное состояние решателя (если нет, то начальное состояние инициализируется по умолчанию)
        :param rules: Дополнительные правила для решения данной задачи
        :param state: Состояние решения общей задачи (может передаваться в подзадачи)
        :param global_params: Глобальные параметры решателя (флаги трассировки, отладки и т.п.)

        :returns: Ответ задачи
        """
        if self._solver_state.global_params.get('trace_solution', False):
            print(f'##########  Start solve task of type {self.__class__.__name__}  ##########')

        if rules is None:
            rules = self.rules
        else:
            rules += self.rules

        if solver_state is None:
            solver_state = SolverState(self, global_params=global_params)

        self._solver_state = solver_state

        nstate = self.prepare_state(state)
        if state is None:
            state = nstate
        self._state = state if state is not None else self

        # state = State(task)
        pos = 0
        while not self.solved:
            if self._solver_state.global_params.get('trace_solution', False):
                print(f'    Apply rule {rules[pos].__class__.__name__} ')
            if rules[pos].can_apply(self, self._state):
                rules[pos].apply(self, self._state)
            pos = (pos + 1) % len(rules)

        if self._solver_state.global_params.get('trace_solution', False):
            print(f'##########  Finish solve task of type {self.__class__.__name__}  #########')

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
        return self.solver_state.run_sub_task(subtask, state)

    def prepare_state(self, state):
        """
        Подготовка состояния перед началом решения задачи.
        Может также создавать новое состояние, если state=None
        """
        return state

    @property
    def rules(self):
        """ Правила, отнесённые к задачам данного типа, включая все правила для базовых классов задач """
        r = []
        cls = self.__class__
        while cls is not Task:
            if hasattr(cls, '_rules'):
                r += cls._rules
        if hasattr(cls, '_rules'):
            r += cls._rules
        return r


def rule(*task_classes):
    """
    Декоратор для классов приёмов.
    Добавляет данный приём к типам задач, указанных в качестве параметров декоратора.
    """
    if len(task_classes) == 0:
        task_classes = [Task]  # По умолчанию добавляем глобальный приём

    def apply(rule_cls):
        instance = rule_cls()
        for task_cls in task_classes:
            if not hasattr(task_cls, '_rules'):
                task_cls._rules = [instance]
            else:
                task_cls._rules.append(instance)
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
    def can_apply(self, task: Task, state: SolverState):
        """
        Проверка целесообразности применения приёма

        Returns:

        bool - признак целесообразности применения приёма
        """
        pass

    @abstractmethod
    def apply(self, task: Task, state: SolverState):
        """
        Применение приёма
        """
        pass
