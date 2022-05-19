# ANN-AutoML

Automated machine learning for artificial neural networks

## Структура проекта

В проект входят следующие компоненты:
1. Пакет ann_automl и система установки, к ней относятся файлы setup.py, setup.cfg, MANIFEST.in,
requirements.txt.
2. Набор тестов в папке **tests**
3. Набор примеров в папке **examples**
4. Данные для обучения нейронных сетей в папке **data**, включая:
   1. информацию об обучающих выборках (data/databases)
   2. стандартные архитектуры нейронных сетей (data/architectures) 
   3. историю обучения нейронных сетей (data/history)
   4. обученные нейронные сети (data/trainedNN)
5. Вспомогательные скрипты в папке **scripts** для подготовки баз данных и их преобразованию в удобный формат   

## Установка пакета

Для установки пакета нужно запустить команду 
```bash
 python setup.py install
```

