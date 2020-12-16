from abc import ABC, abstractmethod
import numpy as np
import keras
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, AveragePooling2D, Flatten, InputLayer
from keras import Sequential

# Взодные данные для задачи
data = None

# Глобал rules задаёт список приёмов
rules = []


def rule(cls):
    global rules
    rules.append(cls())
    return cls


class Task:
    '''
    Базовый тип задачи
    '''

    def __init__(self, type, goals):
        self._type = type  # тип задачи
        self.goals = goals  # множество целей

    @property
    def type(self):
        return self._type


class ModelTask(Task):
    '''
    Тип задачи на создание модели нейронной сети

    Имеет базовую цель 'model', дающую указание приёмам
    создать модель нейронной сети и выдать её в качестве ответа
    '''

    def __init__(self, data, goals=()):
        self.data = data
        super().__init__('model', {'model'} | set(goals))


class State:
    '''
    Базовый тип состояния решения задачи
    '''

    def __init__(self, task):
        self.answer = None  # ответ на решение задачи
        self.task = task  # текущая решаемая задача


class Rule(ABC):
    '''
    Базовый тип приёма

    Имеет два абязательных метода can_apply и apply
    '''

    @abstractmethod
    def can_apply(self, state: State):
        '''
        Проверка целесообразности применения приёма

        Returns:

        bool - признак целесообразности применения приёма
        '''
        pass

    @abstractmethod
    def apply(self, state: State):
        '''
        Применение приёма
        '''
        pass


@rule
class Shuffling(Rule):
    '''
    Приём для случайного перемешивания входных данных

    Имеет смысл применять только при наличии входных данных и цели на обучение
    '''

    def can_apply(self, state):
        return ('train' in state.task.goals
                and hasattr(state.task, 'data'))

    def apply(self, state):
        np.random.shuffle(state.task.data)


@rule
class MakeSequentialModel(Rule):
    '''
    Приём для создания модели последовательного типа

    Имеет смысл применять только при наличии атрибута размерности входных данных
    и цели на создание модели
    '''

    def can_apply(self, state):
        return ('model' in state.task.goals
                and hasattr(state.task.data, 'shape'))

    def apply(self, state):
        dim = state.task.data.shape
        model = Sequential()
        model.add(InputLayer(input_shape=dim))
        model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=2, activation="relu"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=2, activation="relu"))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(1, activation="sigmoid"))
        state.model = model


@rule
class CompileModel(Rule):
    '''
    Приём для компиляции созданной модели

    Имеет смысл применять только при наличии созданной модели, при отсутствии
    признака проведенной компиляции и наличии цели на обучение
    '''

    def can_apply(self, state):
        return ('train' in state.task.goals
                and hasattr(state, 'model')
                and not hasattr(state, 'compiled'))

    def apply(self, state):
        state.model.compile(optimizer='sgd', loss='mse')
        state.compiled = state.model


@rule
class FitModel(Rule):
    '''
    Приём для обучения созданной модели

    Имеет смысл применять только при наличии цели на обучение, 
    скомпилированной модели, для которой еще не запускалосьобучение, а также 
    при наличии у входных данных атибутов 'x' и 'y'.
    '''

    def can_apply(self, state):
        return ('train' in state.task.goals
                and hasattr(state, 'model')
                and hasattr(state, 'compiled')
                and not hasattr(state, 'trained')
                and hasattr(state.task.data, 'x')
                and hasattr(state.task.data, 'y'))

    def apply(self, state):
        x = state.task.data.x
        y = state.task.data.y
        state.model.fit(x, y)
        state.trained = state.model


@rule
class CheckModelAnswer(Rule):
    '''
    Приём для проверки наличия ответа в задаче создания модели нейронной сети
    '''

    def can_apply(self, state):
        return 'model' in state.task.goals

    def apply(self, state):
        if 'train' in self.state.goals:
            if hasattr(state, 'trained'):
                state.answer = self.trained
        else:
            state.answer = state.model


def solve(task: Task, rules):
    '''
    Базовая функция решения задачи
    '''
    state = State(task)
    pos = 0
    while state.answer is None:
        if rules[pos].can_apply(state):
            rules[pos].apply(state)
        pos = (pos + 1) % len(rules)
    return state.answer


# Создаём и решаем задачу создания модели нейронной сети
task = ModelTask(data, goals={'train'})
model = solve(task, rules)
