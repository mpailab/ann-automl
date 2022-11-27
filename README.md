# ANN-AutoML

Библиотека для автоматического построения моделей машинного обучения для задачи классификации изображений.
Позволяет автоматически подбирать гиперпараметры модели и обучать (или дообучать) модели на различных наборах данных.

Позволяет обучать нейронные сети без знания Python через веб-интерфейс.

## Базовые примеры использования

### Запуск графического интерфейса

Из корневой папки проекта запустить команду:

```bash
panel serve launch.py --show
```
В открывшемся окне браузера есть несколько вкладок, в которых можно задавать параметры для обучения нейронной сети и
наблюдать за ее обучением.

1. Вкладка "Базы данных" позволяет выбрать одну или несколько баз данных, 
которые будут использоваться для обучения нейронной сети.
2. Вкладка "Параметры обучения" позволяет просматривать и менять параметры обучения нейронной сети.
3. Вкладка "Обучение" позволяет запустить обучение нейронной сети, просматривать логи и графики обучения.
4. Вкладка "История" позволяет просматривать историю обучения нейронных сетей и скачивать обученные модели.

### Запуск обучения из Python

```python
from ann_automl.core.nn_auto import create_classification_model
categories = ['car', 'airplane', 'bicycle']
model = create_classification_model(categories, output_dir='classifier', target_accuracy=0.9, time_limit=3600*24)
```

Это запустит обучение нейронной сети, которая будет классифицировать 
изображения на 3 класса: 'car', 'airplane', 'bicycle'.
Обучение и подбор гиперпараметров будет продолжаться до тех пор, 
пока точность не достигнет 0.9 или не пройдет 1 день.
После обучения модель и скрипт для запуска будут сохранены в папку 'classifier'.

После того, как модель обучена, ее можно использовать из папки classifier (которая была указана в параметре output_dir) 
для классификации изображений:

1. Классификация изображения из файла:
    ```bash
    python3 classifier.py --image_path /path/to/image.jpg
    ```
   На экран будет выведено имя класса, к которому относится изображение, 
   а также уверенность принадлежности к этому классу.

2. Сортировка изображений из директории по классам:
    ```bash
    python3 classifier.py --image_path /path/to/images/ --out_dir /path/to/output/ --clear_out_dir --threshold 0.6
    ```
   В этом случае изображения из директории '/path/to/images/', которые классифицируются с уверенностью больше 0.6,
   будут отсортированы по классам в директорию '/path/to/output/'. 

Для получения списка всех доступных параметров запустите:
```bash
python3 classifier.py --help
```

Другие примеры и описание API см. в [документации]().

## Структура проекта

В проект входят следующие компоненты:
1. Пакет ann_automl и система установки, к ней относятся файлы setup.py, setup.cfg, MANIFEST.in,
requirements.txt.
2. Набор тестов в папке **tests**
3. Набор примеров в папке **examples**
4. Данные для обучения нейронных сетей в папке **data**, включая:
   1. стандартные архитектуры нейронных сетей (data/architectures) 
   2. обученные нейронные сети (data/trainedNN), в этой папке:
      - для каждого запуска обучения создаётся подпапка со следующими файлами:
        - Обученная модель (best_weights.h5)
        - Её схема (.png)
        - История обучения (.csv)
        - Результаты тестирования (.csv)
        - Конфигурация обучения (.json)
      - для каждого запуска подбора гиперпараметров создаётся подпапка со следующими файлами:
        - История подбора гиперпараметров (.csv)
        - Конфигурация подбора гиперпараметров (.json)
        - Подпапки с результатами обучения нейронных сетей, созданными в ходе подбора гиперпараметров
5. Вспомогательные скрипты в папке **scripts** для подготовки баз данных и их преобразованию в удобный формат  
6. Файлы для генерации документации в папке **docs**

Для работы системы также необходимо наличие следующих файлов и папок:
1. Папка **datasets** с файлами для обучения нейронных сетей.
   Для работы системы рекомендуется, чтобы были доступны 3 базы:
   - Kaggle_CatsVSDogs
   - COCO dataset
   - imagenet
2. Папка **data/architectures** с файлами с дополнительных архитектур нейронных сетей, которые планируется использовать.
   и которые не входят в стандартный набор keras.applications.
   
Пакет ann_automl имеет следующие подмодули:
1. **core** -- ядро системы, работа с нейронными сетями и обучающими выборками
2. **nnplot** -- модуль для визуализации нейронных сетей, рисования различных графиков, связанных с обучением нейронных сетей
3. **gui** -- графический веб-интерфейс для работы с системой
4. **jupyter** -- различные функции для упрощённого использования функций пакета из jupyter notebook
5. **utils** -- библиотека различных вспомогательных функций общего назначения 

## Установка пакета

Для установки пакета нужно запустить команду 
```bash
 python setup.py install
```
