import json
import os
import shutil
import sys
import time
import warnings
from typing import Any, Callable, Dict, List

from .nn_recommend import recommend_hparams
from .nnfuncs import nn_hparams, tune_hparams, get_hparams, train, tune
from .solver import Task, set_log_dir, printlog, _log_dir, RecommendTask

TargetFunc = Callable[[List[float]], float]


def loss_target(scores: List[float]) -> float:
    return -scores[0]


def metric_target(scores: List[float]) -> float:
    return scores[1]


class NNTask(Task):
    """ Класс, задающий задачу на обучение нейронной сети """

    def __init__(self,
                 category: str = 'train',
                 objects: List[str] = (),
                 type: str = 'classification',
                 func: TargetFunc = metric_target,
                 # TODO: кажется, надо вернуть как было -- либо accuracy,
                 #       либо конкретный вид accuracy (categorial, binary, sparse_categorical, etc.)
                 target: float = 0.9,
                 goals: Dict[str, Any] = None,
                 time_limit: int = 60 * 60 * 24):
        """
        Инициализация задачи.

        Parameters:
        ----------
        category: str
            Категория задачи:
            train - обучение модели,
            test - тестирование модели без предварительного обучения 
            "служебные" - {проверка наличия нужных классов,БД, моделей}
        type: str
            Тип задачи {classification, detection, segmentation}
        objects: list[str]
            Список категорий изображений
        func: TargetFunc
            Целевой функционал {loss_target, metric_target}
        target: str
            Значение целевого функционала {float from [0,1]}
        goal: dict[str, Any]
            Словарь целей задачи (например {"метрика" : желаемое значение метрики})
        """
        super().__init__(goals=goals or {})
        self._category = category  # str

        # Необязательные входные параметры задачи, нужны при категориях "train" и "test"
        self._type = type  # str - тип задачи {classification, detection, segmentation}
        self._objects = objects or []  # set of strings - набор категорий объектов интереса
        self.input_type = 'image'  # тип объектов задачи (image, video, audio, text)
        self.object_category = 'object'  # категория объектов задачи (object, symbol)
        self.func = func
        self.target = target
        self.log_name = ''
        self.time_limit = time_limit

    @property
    def category(self):
        """ Возвращает категорию задачи """
        return self._category

    @property
    def type(self):
        """ Возвращает тип задачи """
        return self._type

    @property
    def objects(self):
        """ Возвращает набор категорий объектов интереса """
        return self._objects

    @property
    def metric(self):
        """ Возвращает название целевого функционала """
        return 'accuracy' if self.func.__name__ == 'metric_target' else self.func.__name__
        # TODO: это пока костыль для поиска по истории (не хорошо передавать имя функции в качестве метрики)


def train_classification_model(classes,
                               target_accuracy,
                               optimize_over_target=True,
                               stop_flag=None,
                               verbosity=1,
                               time_limit=60 * 60 * 24):
    """
    Создает модель для классификации изображений

    Args:
        classes: список классов
        target_accuracy: требуемая точность
        optimize_over_target: оптимизировать ли модель после достижения целевой точности, пока не достигнется лимит времени
        stop_flag: флаг, показывающий, что нужно остановить обучение (для запуска из другого процесса)
        verbosity: уровень подробности печатаемой информации
        time_limit: лимит времени на обучение модели (в секундах)

    Returns:
        Информацию об обученной модели, включая путь к файлу с сохранённой моделью и значения метрик
    """
    start_time = time.time()
    task = NNTask(category='train', type='classification', objects=classes, func=metric_target, target=target_accuracy,
                  goals={'maximize': optimize_over_target}, time_limit=time_limit)
    hparams = recommend_hparams(task)
    metrics, simple_params = train(task, hparams, stop_flag=stop_flag)
    simple_val = task.func(metrics)
    simple_time = time.time() - start_time
    if simple_val >= task.target or time.time() - start_time > task.time_limit:
        return simple_params, simple_val

    best_params = simple_params
    best_val = simple_val
    first_run = True
    while time.time() + simple_time - start_time < task.time_limit:
        start_time = time.time()
        hparams = recommend_hparams(task)
        p, score, params = tune(task, tuned_params='all', method='grid', hparams=hparams, stop_flag=stop_flag,
                                verbosity=verbosity, timeout=task.time_limit - (time.time() - start_time),
                                start='auto' if first_run else 'random')
        val = task.func(metrics)
        if val > best_val:
            best_params = params
            best_val = val
            simple_time = time.time() - start_time
        if val >= task.target and not optimize_over_target:
            break
        first_run = False

    return best_params, best_val


def copy_classify_script(out_dir):
    printlog('Копируем скрипт для классификации')
    script_path = os.path.join(os.path.dirname(__file__), f'../scripts/')
    shutil.copy(os.path.join(script_path, 'classify.py'), out_dir)
    shutil.copy(os.path.join(script_path, 'tf_funcs.py'), out_dir)
    shutil.copy(os.path.join(script_path, 'torch_funcs.py'), out_dir)


def create_classification_model(classes,
                                target_accuracy,
                                output_dir,
                                optimize_over_target=True,
                                stop_flag=None,
                                script_type='tf',
                                verbosity=1,
                                time_limit=60 * 60 * 24):
    """
    Создает модель и скрипт для классификации изображений

    Args:
        classes: список классов
        target_accuracy: требуемая точность
        output_dir: директория для сохранения модели и скрипта (если имеет расширение .zip, то создаётся архив)
        optimize_over_target: оптимизировать ли модель после достижения целевой точности, пока не достигнется лимит времени
        stop_flag: флаг, показывающий, что нужно остановить обучение (для запуска из другого процесса)
        script_type: тип скрипта (tf, torch или None). Если None, то скрипт не создаётся
        verbosity: уровень подробности печатаемой информации
        time_limit: лимит времени на обучение модели (в секундах)
    """
    def log(*args, level=1, **kwargs):
        if verbosity >= level:
            print(*args, **kwargs)
    log(f'Запускаем создание модели, это может занять длительное время.\n'
        f'Установленный лимит времени на обучение модели: {time_limit} секунд.\n'
        f'Это ориентировочный лимит, в действительности процесс может занять больше или меньше времени.\n'
        f'Установленная целевая точность: {target_accuracy}.\n')
    model_info, val = train_classification_model(classes, target_accuracy, optimize_over_target,
                                                 stop_flag, verbosity, time_limit)
    if val < target_accuracy:
        warnings.warn(f'Не удалось достичь требуемой точности. Полученная точность: {val}')
    elif verbosity > 0:
        log(f'Модель готова. Полученная точность на тестовой выборке: {val}')

    model_path = model_info['model_path']  # directory with model
    save_dir = output_dir if not output_dir.endswith('.zip') else 'tmp/zip_out'
    # clear save_dir if it exists or create it
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    # copy model (best_weights.h5) to save_dir
    shutil.copy(os.path.join(model_path, 'best_weights.h5'), os.path.join(save_dir, 'model.h5'))
    info = dict(model_path='model.h5', classes=classes, test_accuracy=val)
    if script_type is not None:
        if script_type not in ('tf', 'torch'):
            log(f'Указан неправильный тип скрипта: {script_type}, должен быть tf или torch', file=sys.stderr)
            log('Используем тип tf', file=sys.stderr)
            script_type = 'tf'
        copy_classify_script(save_dir)
        info['backend'] = script_type

    with open(os.path.join(save_dir, 'model.json'), 'w') as f:
        json.dump(info, f)

    if output_dir.endswith('.zip'):
        log(f'Создаём архив {output_dir} ...')
        # check whether path to zip file exists
        out_dir = os.path.dirname(output_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        shutil.make_archive(output_dir[:-4], 'zip', save_dir)
        shutil.rmtree(save_dir, ignore_errors=True)
        log(f'Архив {output_dir} создан')
