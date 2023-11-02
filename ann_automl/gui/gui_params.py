from .params import hyperparameters, widget_type

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

chatbot_params = {
    'langmodel' : { 'title': 'Используемая чат-ботом языковая модель',
                 'type': 'str', 'default': 'Lama_1',
                 'values': ['Lama_1', 'Lama_2', 'Lama_3'] },
}

labeling_params = {
    'images_path': { 'title': 'Путь/Ссылка к каталогу/zip-архиву изображений', 'type': 'str', 'default': '/auto/projects/brain/ann-automl-gui/datasets/test1/Images example.zip' },
    'nn_core': { 'title': 'Ядро разметчика (используемая нейросеть)',
                 'type': 'str', 'default': 'yolov5s',
                 'values': ['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x'] },
}

dataset_params = {
    'description': { 'title': 'Название датасета', 'type': 'str', 'default': '' },
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
    **{
        f'chatbot_{k}': {**v, 'gui': {'group': 'chatbot', 'widget': widget_type(v)}}
        for k,v in chatbot_params.items()
    },
    **hyperparameters
}