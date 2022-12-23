import datetime
import multiprocessing
import sys
import warnings
import getpass

from sqlalchemy import *
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
import pandas as pd
import json
import ast
import cv2
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import os
import math
from PIL import Image
import glob
import xml.etree.ElementTree as ET
import time

from tqdm import tqdm

from ann_automl.utils.text_utils import print_progress_bar


Base = declarative_base()


def check_coco_images(anno_file, image_dir):
    """
    Проверяет, существует ли директория с изображениями; если нет, загружает изображения из интернета по url из файла аннотаций

    Args:
        anno_file (str): путь к файлу аннотаций
        image_dir (str): путь к директории с изображениями
    """
    if os.path.exists(image_dir):
        return
    print(f'Downloading COCO images for test (annotations = {anno_file})')
    from pycocotools.coco import COCO
    import requests
    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
    print_progress_bar(0, len(img_ids), prefix='Loading images:', suffix='Complete', length=50)
    for i, img_id in enumerate(img_ids):
        img = coco.loadImgs([img_id])[0]
        img_data = requests.get(img['coco_url']).content
        with open(image_dir + '/' + img['file_name'], 'wb') as handler:
            handler.write(img_data)
        print_progress_bar(i, len(img_ids), prefix='Loading images:', suffix='Complete', length=50)


def _crop_image(input_file, output_file, x, y, w, h):
    # the most efficient way to crop an image
    image = cv2.imread(input_file)
    image = image[math.floor(y):math.ceil(y + h), math.floor(x):math.ceil(x + w)]
    cv2.imwrite(output_file, image)


def _crop_image_tuple(args):
    return _crop_image(*args)


_default_num_processes = 1  # multiprocessing.cpu_count()-1


def set_default_num_processes(num_processes):
    """
    Устанавливает количество процессов для параллельного обрезания изображений. По умолчанию 1

    Args:
        num_processes (int): максимальное количество процессов
    """
    global _default_num_processes
    _default_num_processes = num_processes


class num_processes_context:
    def __init__(self, num_processes):
        """
        Контекстный менеджер для установки количества процессов для параллельного обрезания изображений в блоке кода.
        По умолчанию 1.

        Args:
            num_processes (int): максимальное количество процессов
        """
        self.num_processes = num_processes

    def __enter__(self):
        self.old_num_processes = _default_num_processes
        set_default_num_processes(self.num_processes)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_num_processes(self.old_num_processes)


class DBModule:

    ############################################################
    ##########        DB ORM description     ###################
    ############################################################

    class Image(Base):
        __tablename__ = "image"

        ID = Column(Integer, primary_key=True)
        """ Идентификатор изображения в БД """
        dataset_id = Column(Integer, ForeignKey("dataset.ID"))
        """ID датасета, к которому относится это изображение"""
        license_id = Column(Integer, ForeignKey("license.ID"))
        """ID лицензии, под которой выпущено это изображение"""
        file_name = Column(String)
        """Имя файла изображения"""
        coco_url = Column(String)
        """URL файла изображения"""
        height = Column(Integer)
        """высота изображения"""
        width = Column(Integer)
        """ширина изображения"""
        date_captured = Column(String)
        """дата создания изображения"""
        flickr_url = Column(String)
        aux = Column(String)
        """Вспомогательные данные для изображения"""

        def __init__(self, file_name, width, height, date_captured, dataset_id, coco_url='', flickr_url='',
                     license_id=-1, ID=None, aux=''):
            """ Создание информации об изображении для записи в базу данных """
            self.width = width
            self.height = height
            self.file_name = file_name
            self.date_captured = date_captured
            self.dataset_id = dataset_id
            self.coco_url = coco_url
            self.flickr_url = flickr_url
            if license_id == -1:
                license_id = 1
            self.license_id = license_id
            if ID is not None:
                self.ID = ID
            self.aux = aux

    class Dataset(Base):
        __tablename__ = "dataset"
        ID = Column(Integer, primary_key=True)
        """ Идентификатор датасета в БД """
        description = Column(String)
        """Имя или описание датасета"""
        url = Column(String)
        """URL датасета (если есть)"""
        version = Column(String)
        """Версия датасета"""
        year = Column(Integer)
        """Год, когда был создан датасет"""
        contributor = Column(String)
        """Автор датасета"""
        date_created = Column(String)
        """Дата создания датасета"""
        aux = Column(String)
        """Вспомогательная информация для датасета"""
        images = relationship("Image", backref=backref("dataset"))
        """Привязка к таблице Image (one-to-many)"""

        def __init__(self, description, url, version, year, contributor, date_created, ID=None, aux=''):
            """ Создание информации о датасете для записи в базу данных """
            self.description = description
            self.url = url
            self.version = version
            self.year = year
            self.contributor = contributor
            self.date_created = date_created
            if ID is not None:
                self.ID = ID
            self.aux = aux

    class Category(Base):
        __tablename__ = "category"
        ID = Column(Integer, primary_key=True)
        """ Идентификатор категории в БД """
        supercategory = Column(String)
        """Надкатегория категории (например, 'animal', 'vehicle', 'outdoor')"""
        name = Column(String)
        """Имя категории (например, 'dog', 'car', 'tree')"""
        aux = Column(String)
        """Вспомогательная информация для категории"""
        images = relationship("Annotation", backref=backref("category"))
        """Привязка к таблице Annotation (one-to-many, существует много аннотаций для одной категории)"""
        records = relationship("CategoryToModel")
        """Привязка к таблице Model (many-to-many через CategoryToModel);
         каждая обученная модель поддерживает некоторый набор категорий, 
         а каждая категория поддерживается некоторым набором моделей"""

        def __init__(self, supercategory, name, ID=None, aux=''):
            """ Создание информации о категории для записи в базу данных """
            self.supercategory = supercategory
            self.name = name
            if ID is not None:
                self.ID = ID
            self.aux = aux

    class License(Base):
        __tablename__ = "license"
        ID = Column(Integer, primary_key=True)
        """ Идентификатор лицензии в БД """
        name = Column(String)
        """Название лицензии"""
        url = Column(String)
        """URL лицензии"""
        aux = Column(String)
        """Вспомогательная информация для лицензии"""
        images = relationship("Image")
        """Привязка к таблице Image (изображение может иметь лицензию)"""

        def __init__(self, name, url, ID=None, aux=''):
            """ Создание информации о лицензии для записи в базу данных """
            self.url = url
            self.name = name
            if ID is not None:
                self.ID = ID
            self.aux = aux

    class Annotation(Base):
        __tablename__ = "annotation"
        ID = Column(Integer, primary_key=True)
        """ Идентификатор аннотации в БД """
        image_id = Column(Integer, ForeignKey("image.ID"))
        """ID изображения, связанного с аннотацией"""
        category_id = Column(Integer, ForeignKey("category.ID"))
        """ID категории, связанной с аннотацией"""
        bbox = Column(String)
        """Ограничивающий прямоугольник для объекта в формате [x, y, width, height]"""
        segmentation = Column(String)
        """Сегментация, определенная аннотацией"""
        is_crowd = Column(Integer)

        area = Column(Float)
        """Площадь объекта, описываемого аннотацией"""
        aux = Column(String)
        """Вспомогательная информация для аннотации"""

        def __init__(self, image_id, category_id, bbox, segmentation, is_crowd, area, ID=None, aux=''):
            """ Создание информации об аннотации для записи в базу данных """
            self.image_id = image_id
            self.category_id = category_id
            self.bbox = bbox
            self.segmentation = segmentation
            self.is_crowd = is_crowd
            self.area = area
            if ID is not None:
                self.ID = ID
            self.aux = aux

    class TrainResult(Base):
        __tablename__ = "trainResult"
        ID = Column(Integer, primary_key=True)
        """ Идентификатор результата обучения в БД """
        metric_name = Column(String)
        """Название метрики (например, 'accuracy', 'loss')"""
        metric_value = Column(Float)
        """Значение метрики, достигнутое моделью на тестовых данных"""
        model_id = Column(Integer, ForeignKey("model.ID"))
        """ID модели, связанной с результатом"""
        history_address = Column(String)
        """Адрес файла истории, описывающего процесс обучения"""
        aux = Column(String)
        """Может содержать дополнительную информацию о процессе обучения"""

        def __init__(self, metric_name, metric_value, model_id, history_address='', aux='', ID=None):
            """ Создание информации о результате обучения для записи в базу данных """
            self.metric_name = metric_name
            self.metric_value = metric_value
            self.history_address = history_address
            self.model_id = model_id
            if ID is not None:
                self.ID = ID
            self.aux = aux

    class CategoryToModel(Base):
        """:meta private:"""
        __tablename__ = "categoryToModel"
        category_id = Column(Integer, ForeignKey("category.ID"), primary_key=True)
        model_id = Column(Integer, ForeignKey("model.ID"), primary_key=True)

        def __init__(self, category_id, model_id):
            """ Связывает модель с категорией, которую она может распознавать"""
            self.category_id = category_id
            self.model_id = model_id

    class Model(Base):
        __tablename__ = "model"
        ID = Column(Integer, primary_key=True)
        """ Идентификатор модели в БД """
        model_address = Column(String)
        """Адрес файла модели"""
        task_type = Column(String)
        """Тип задачи обучения нейронной сети"""
        aux = Column(String)
        """Вспомогательная информация для модели"""
        train_results = relationship("TrainResult", backref=backref("model"))
        """Связь с таблицей TrainResult (модель может иметь несколько записей TrainResult для разных метрик)"""
        categories = relationship("CategoryToModel")
        """Связь с таблицей Category (many-to-many через CategoryToModel);
            каждая модель поддерживает некоторый набор категорий"""

        def __init__(self, model_address, task_type, aux='', ID=None):
            """Создание записи о модели для сохранения в базе данных"""
            self.model_address = model_address
            self.task_type = task_type
            if ID is not None:
                self.ID = ID
            self.aux = aux

    ############################################################
    ##########        DB Module methods      ###################
    ############################################################

    def __init__(self, dbstring='sqlite:///datasets.sqlite', dbconf_file='dbconfig.json', dbecho=False):
        """Создает сеанс работы с БД по адресу,
        заданному в dbstring (по умолчанию sqlite:///datasets.sqlite, то есть локальная
        БД в файле datasets.sqlite в текущей директории)

        Args:
            dbstring (str): адрес файла sqlite БД
            dbconf_file (str): адрес файла конфигурации, в котором содержатся пути к датасетам
            dbecho (bool): выводить ли в консоль запросы к БД
        """
        if os.path.isfile(dbconf_file):  # if config file exists we take all paths from there
            with open(dbconf_file) as f:
                dbconfig = json.load(f)
                if dbconfig.get('dbstring', False):
                    dbstring = dbconfig['dbstring']
                if dbconfig.get('KaggleCatsVsDogs', False):
                    self.KaggleCatsVsDogsConfig_ = dbconfig['KaggleCatsVsDogs']
                if dbconfig.get('COCO2017', False):
                    self.COCO2017Config_ = dbconfig['COCO2017']
                if dbconfig.get('ImageNet', False):
                    self.ImageNetConfig_ = dbconfig['ImageNet']
        # first_time = not os.path.isfile(dbstring[10:])
        self.engine = create_engine(dbstring, echo=dbecho)
        Session = sessionmaker(bind=self.engine)
        self.sess = Session()
        self.dbstring_ = dbstring
        self._dbfile = dbstring[10:]
        self.ds_filter = None
        # if first_time:
        #     Base.metadata.create_all(self.engine)
        #     self.add_default_licences()

    def create_sqlite_file(self):
        """Если файл SQLite ещё не создан, его нужно создать вызовом этого метода"""
        Base.metadata.create_all(self.engine)
        self.add_default_licences()

    def fill_all_default(self):
        """Метод для заполнения всех баз данных, предполагая, что
        датасеты находятся в директориях по умолчанию (CatsDogs, COCO, ImageNet)"""
        if os.path.exists(self._dbfile):  # If file exists we suppose it is filled
            return
        self.create_sqlite_file()
        self.fill_coco()
        self.fill_kaggle_cats_vs_dogs()
        self.fill_imagenet(first_time=True)
        return

    def fill_kaggle_cats_vs_dogs(self, anno_file_name='dogs_vs_cats_coco_anno.json', 
                                 file_prefix='./datasets/Kaggle/'):
        """Метод для заполнения базы данных Kaggle CatsVsDogs.
        Предполагается, что он вызывается один раз.

        Args:
            anno_file_name (str): файл с json-аннотацией в формате COCO для кошек и собак
            file_prefix (str): префикс, добавляемый к именам файлов в файле аннотации
        Raises:
            FileNotFoundError: если директория file_prefix и файл anno_file_name не существуют
        """
        if hasattr(self, 'KaggleCatsVsDogsConfig_'):  # to init from config file
            anno_file_name = self.KaggleCatsVsDogsConfig_['anno_filename']
            file_prefix = self.KaggleCatsVsDogsConfig_['file_prefix']

        print('Start filling DB with Kaggle CatsVsDogs')
        if not os.path.isfile(anno_file_name):
            raise FileNotFoundError('Error: no annotation file', anno_file_name, 'found')
        if not os.path.isdir(file_prefix):
            raise FileNotFoundError('Error: no directory', file_prefix, 'found')
        with open(anno_file_name) as json_file:
            data = json.load(json_file)
        filepath = file_prefix + data['images'][0]['file_name'].split('.')[0] + 's/' + data['images'][0]['file_name']
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'In json reference to images that doesn`t exist (i.e. {filepath})')
        dataset_info = data['info']
        dataset = self.Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'],
                               dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit()  # adding dataset
        ###################################
        im_objects = {}
        print_progress_bar(0, len(data['images']), 
                           prefix='Adding images:', suffix='Complete', length=50)
        im_counter = 0
        for im_data in data['images']:
            image = self.Image(file_name=file_prefix + im_data['file_name'].split('.')[0] + 's/' + im_data['file_name'],
                               width=im_data['width'],
                               height=im_data['height'],
                               date_captured=im_data['date_captured'],
                               ID=dataset.ID,
                               coco_url=im_data['coco_url'],
                               flickr_url=im_data['flickr_url'],
                               license_id=im_data['license'])
            im_objects[im_data['id']] = image
            self.sess.add(image)
            im_counter += 1
            if im_counter % 10 == 0 or im_counter == len(data['images']):
                print_progress_bar(im_counter, len(data['images']),
                                   prefix='Adding images:', suffix='Complete', length=50)
        self.sess.commit()  # adding images
        ###################################

        print_progress_bar(0, len(data['annotations']), 
                           prefix='Adding annotations:', suffix='Complete', length=50)
        an_counter = 0
        for an_data in data['annotations']:
            # +1 because of json file structure for this DB only
            real_id = im_objects[an_data['image_id']].ID
            annotation = self.Annotation(image_id=real_id,
                                         category_id=an_data['category_id'] + 1,
                                         bbox=';'.join(an_data['bbox']),
                                         segmentation=';'.join(an_data['segmentation']),
                                         is_crowd=an_data['iscrowd'],
                                         area=an_data['area'])
            self.sess.add(annotation)
            an_counter += 1
            if an_counter % 10 == 0 or an_counter == len(data['annotations']):
                print_progress_bar(an_counter, len(data['annotations']),
                    prefix='Adding annotations:', suffix='Complete', length=50)
        self.sess.commit()  # adding annotations
        print('Finished with Kaggle CatsVsDogs')

    def fill_coco(self, 
                  anno_file_name='./datasets/COCO2017/annotations/instances_train2017.json', 
                  file_prefix='./datasets/COCO2017/', ds_info=None):
        """ Заполняет базу данных COCOdataset. Предполагается, что вызывается один раз.

        Для создания собственных аннотаций COCO используйте вспомогательные инструменты,
        например, https://github.com/jsbroks/coco-annotator.

        Args:
            anno_file_name (str): файл с json-аннотациями в формате COCO
            file_prefix (str): префикс, добавляемый к именам файлов в файле аннотаций
            ds_info (dict): словарь с информацией о наборе данных (по умолчанию - COCO2017). Необходимые ключи:
                description, url, version, year, contributor, date_created
        Raises:
            FileNotFoundError: если не существует директории file_prefix или файла anno_file_name
        """
        if hasattr(self, 'COCO2017Config_'):  # to init from config file
            anno_file_name = self.COCO2017Config_['anno_filename']
            file_prefix = self.COCO2017Config_['file_prefix']

        print('Start filling DB with COCO-format dataset from file', anno_file_name)
        if not os.path.isfile(anno_file_name):
            raise FileNotFoundError('No annotation file', anno_file_name, 'found')
        if not os.path.isdir(file_prefix):
            raise FileNotFoundError('No directory', file_prefix, 'found')
        coco = COCO(anno_file_name)
        cats = coco.loadCats(coco.getCatIds())
        self.add_categories(cats, True)

        #######################################################################
        if ds_info is None:
            ds_info = {"description": "COCO 2017 Dataset",
                       "url": "https://cocodataset.org",
                       "version": "1.0", 
                       "year": 2017,
                       "contributor": "COCO Consortium",
                       "date_created": "2017/09/01"}
        ds_id = self.add_dataset_info(ds_info)
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        ann_ids = coco.getAnnIds()
        anns = coco.loadAnns(ann_ids)
        print(f'Dataset description: ', ds_info["description"])
        print(f'Adding {len(anns)} annotations in COCO format to DB')
        self.add_images_and_annotations(imgs, anns, ds_id, file_prefix)

    def fill_in_coco_format(self, anno_file_name, file_prefix, ds_info, auto_download=False):
        """ Заполняет базу данных новым набором данных в формате COCO.
        Предполагается, что вызывается для каждого нового датасета.

        Args:
            anno_file_name (str): файл с json-аннотациями в формате COCO
            file_prefix (str): префикс, добавляемый к именам файлов в файле аннотаций (обычно - путь к директории с изображениями)
            ds_info (dict): словарь с информацией о наборе данных. Необходимые ключи:
                description, url, version, year, contributor, date_created
            auto_download (bool): если True и директория file_prefix не существует,
                то изображения загружаются из интернета по url, взятому из файла аннотаций
        Raises:
            FileNotFoundError: если не существует директории file_prefix или файла anno_file_name
        """
        if auto_download:
            check_coco_images(anno_file_name, file_prefix)
        if not os.path.isfile(anno_file_name):
            raise FileNotFoundError(f'No file {anno_file_name} found, stop filling dataset')
        if not os.path.isdir(file_prefix):
            raise FileNotFoundError(f'No directory {file_prefix} found, stop filling dataset')
        coco = COCO(anno_file_name)
        cats = coco.loadCats(coco.getCatIds())
        cat_names = [cat['name'] for cat in cats]
        cat_ids = self.get_cat_IDs_by_names(cat_names)
        add_cats = [cat for cat, cat_id in zip(cats, cat_ids) if cat_id < 0]
        if len(add_cats) > 0:
            self.add_categories(add_cats, respect_ids=False)
        cat_ids = self.get_cat_IDs_by_names(cat_names)
        cat_id_map = {cat['id']: cat_id for cat, cat_id in zip(cats, cat_ids)}
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        ann_ids = coco.getAnnIds()
        anns = coco.loadAnns(ann_ids)
        print(f'Dataset description: {ds_info["description"]}')
        print(f'Adding {len(anns)} annotations in COCO format to DB')
        ds_id = self.add_dataset_info(ds_info)
        self.add_images_and_annotations(imgs, anns, ds_id, file_prefix, category_id_map=cat_id_map)

    def add_tensorflow_dataset(self, tdfs_name, name=None, ds_path=None, datasets_dir='./datasets'):
        """ Добавляет в базу новый датасет через tensorflow_datasets.

        Args:
            tdfs_name (str): имя набора данных в формате TensorFlow Dataset
            name (str): имя набора данных в базе данных
            ds_path (str): путь к директории с набором данных
            datasets_dir (str): путь к директории с наборами данных
        """
        from ..utils.load_ds import download_tfds
        if name is None:
            name = tdfs_name.split('/')[-1]
        res = download_tfds(tdfs_name, ds_path, os.path.join(datasets_dir, name))
        anno_file = res['annotations']
        prefix = res['images']
        ds_info = {"description": res['name'],
                   "url": res['url'],
                   "version": res['version'],
                   "year": "",
                   "contributor": 'tensorflow_datasets',
                   "date_created": ""}
        self.fill_in_coco_format(anno_file_name=anno_file, file_prefix=prefix, ds_info=ds_info)

    def add_dataset_from_subdirs(self, root_folder, ds_name, ds_info=None, datasets_dir='./datasets'):
        """ Заполняет базу данных новым набором данных, представленным в виде поддиректорий с изображениями.
        Имя поддиректории - имя класса.

        Args:
            root_folder (str): корневая директория, содержащая директории с изображениями
            ds_name (str): название набора данных
            ds_info (dict): словарь с информацией о наборе данных. Необходимые ключи:
                description, url, version, year, contributor, date_created
            datasets_dir (str): путь к директории, в которой хранятся датасеты
        """
        from ..utils.load_ds import create_annotations
        ds_info_0 = {"description": ds_name,
                   "url": "",
                   "version": "1.0",
                   "year": f"{datetime.datetime.now():%Y}",
                   "contributor": f"{getpass.getuser()}",
                   "date_created": f"{datetime.datetime.now():%Y-%m-%d}"}
        if ds_info is not None:
            ds_info_0.update(ds_info)
        ds_info = ds_info_0
        ds_path = os.path.join(datasets_dir, ds_name)
        anno_file = os.path.join(ds_path, 'annotations.json')
        if os.path.isfile(anno_file):
            print(f'Annotations file {anno_file} already exists, skip creating')
        else:
            if not os.path.isdir(ds_path):
                os.mkdir(ds_path)
            create_annotations(root_folder, anno_file)

        self.fill_in_coco_format(anno_file_name=anno_file, file_prefix=root_folder, ds_info=ds_info)

    def fill_imagenet(self, annotations_dir='./datasets/imagenet/annotations',
                      file_prefix='./datasets/imagenet/ILSVRC2012_img_train',
                      assoc_file='imageNetToCOCOClasses.txt', first_time=False,
                      ds_info=None):
        """ Заполняет базу данных набором данных ImageNet.
        Предполагается, что вызывается один раз.

        Args:
            annotations_dir (str): путь к директории с аннотациями
            file_prefix (str): префикс к именам файлов из ImageNet (обычно - путь к директории с изображениями)
            assoc_file (str): имя файла с соответствием категорий ImageNet и COCO
            first_time (bool): если True, то добавляются категории из файла assoc_file
            ds_info (dict): словарь с информацией о наборе данных.
        Raises:
            FileNotFoundError: если не существует директории annotations_dir или file_prefix
        """
        if hasattr(self, 'ImageNetConfig_'):  # to init from config file
            annotations_dir = self.ImageNetConfig_['annotations_dir']
            file_prefix = self.ImageNetConfig_['file_prefix']
            assoc_file = self.ImageNetConfig_['categories_assoc_file']

        if not os.path.isdir(annotations_dir):
            raise FileNotFoundError('No directory', annotations_dir, 'found')
        if not os.path.isdir(file_prefix):
            raise FileNotFoundError('No directory', file_prefix, 'found')
        # If we make it for the first time, we add dataset information to DB
        if first_time:
            if ds_info is None:
                ds_info = {"description": "ImageNet 2012 Dataset", "url": "https://image-net.org/about.php",
                           "version": "1.0", "year": 2012, "contributor": "ImageNet", "date_created": "2012/01/01"}
            ds_id = self.add_dataset_info(ds_info)
        else:
            ds_id = 3  # TODO: this is just a patch for ImageNet, since in 'clear' dataset imageNet is filled last
        # Then we take all the associations from the assoc_file and create some categories if needed
        assoc = {}
        with open(assoc_file) as file:
            lines = file.readlines()
            for line in lines:
                arr = line.split(';')
                assoc[arr[0]] = [arr[1], arr[2].rstrip('\n')]
        cat_names = set()
        categories_buf_assoc = {}
        for elem in assoc:
            if assoc[elem][1] != 'None':
                cat_names.add(assoc[elem][1])
                categories_buf_assoc[elem] = assoc[elem][1]
            else:
                cat_names.add(assoc[elem][0])
                categories_buf_assoc[elem] = assoc[elem][0]
        cat_names_list = list(cat_names)
        cat_ids = self.get_cat_IDs_by_names(cat_names_list)
        new_categories = []
        for i in range(len(cat_ids)):
            cat_id = cat_ids[i]
            if cat_id < 0:
                cat_name = cat_names_list[i]
                new_categories.append({'supercategory': 'imageNetOther', 'name': cat_name})
        if len(new_categories) > 0:
            self.add_categories(new_categories, respect_ids=False)
            cat_ids = self.get_cat_IDs_by_names(cat_names_list)
            for i in range(len(cat_ids)):
                assert cat_ids[i] > 0, 'Some category could not be added'
        categories_assoc = {}
        for i in range(len(cat_ids)):
            cat_id = cat_ids[i]
            cat_name = cat_names_list[i]
            categories_assoc[cat_name] = cat_id
        # Then we iterate over all the images from dataset.
        # For each image we check if there are bbox in XML files.
        # If so - this information is added to annotation.
        # ImageNet file structure requires /*/*.JPEG
        img_files = glob.glob(file_prefix + '/*/*.JPEG', recursive=True)
        print('Len img_files:', len(img_files))
        images = []
        annotations = []
        licence_id = 1  # default value
        im_id = 0
        print_progress_bar(0, len(img_files), prefix='Processing ImageNet XML files:',
                           suffix='Complete', length=50)
        for img_file in img_files:
            im = Image.open(img_file)
            width, height = im.size
            # creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') #previous version
            creation_time = time.ctime(os.path.getctime(img_file))
            image_data = {'file_name': img_file, 'width': width, 'height': height, 'date_captured': creation_time,
                          'coco_url': '', 'flickr_url': '', 'license': licence_id,
                          'id': im_id}  # id will not be respected when added to base - needed for annotations only
            images.append(image_data)
            img_name_no_ext = os.path.splitext(os.path.basename(img_file))[0]
            anno_subdir = img_name_no_ext.split('_')[0]
            annofilename = os.path.join(annotations_dir, anno_subdir, img_name_no_ext + '.xml')
            if os.path.isfile(annofilename):
                # corresponding annotation file is found
                tree = ET.parse(annofilename)
                root = tree.getroot()
                obj_tag = root.find('object')
                if obj_tag is not None:
                    for child in obj_tag:
                        bbox = []
                        if child.tag == 'bndbox':
                            for bchild in child:
                                bbox.append(bchild.text)
                            area = np.abs((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
                            annotation_data = {'image_id': im_id, 'segmentation': '', 'bbox': json.dumps(bbox),
                                               'iscrowd': 0, 'area': area,
                                               'category_id': categories_assoc[categories_buf_assoc[anno_subdir]]}
                            annotations.append(annotation_data)
            else:
                annotation_data = {'image_id': im_id, 'segmentation': '', 'bbox': '', 'iscrowd': 0,
                                   'area': width * height,
                                   'category_id': categories_assoc[categories_buf_assoc[anno_subdir]]}
                annotations.append(annotation_data)
            im_id += 1
            if im_id % 10 == 0 or im_id == len(img_files) - 1:
                print_progress_bar(im_id, len(img_files), 
                    prefix='Processing ImageNet XML files:',
                    suffix='Complete', length=50)
        self.add_images_and_annotations(images, annotations, ds_id)
        return images, annotations

    def get_all_datasets(self, full_info=False):
        """
        Возвращает все датасеты в базе данных в виде словаря

        Args:
            full_info (bool): Если True, то возвращается полная информация о датасете,
            включая количество аннотаций для каждой категории.
            Иначе возвращается только базовая информация (название, автор, версия, дата создания, лицензия)

        Returns:
            dict: словарь вида {ID : информация о датасете}
        """
        # SQL file is stored inside project working directory
        if not os.path.exists(self._dbfile):
            self.create_sqlite_file()
        query = self.sess.query(self.Dataset)
        df = pd.read_sql(query.statement, query.session.bind)
        df_rec = df.to_dict(orient='records')
        df_dict = {}
        for el in df_rec:
            if not full_info:
                df_dict[el['ID']] = el
                del df_dict[el['ID']]['ID']
            else:
                df_dict[el['ID']] = self.get_full_dataset_info(el['ID'])
        return df_dict

    def get_all_datasets_info(self, full_info=False):
        """
        Возвращает информацию обо всех датасетах в базе данных в виде словаря

        Args:
            full_info (bool): Если True, то возвращается полная информация о датасете,
                включая количество аннотаций для каждой категории и каждой надкатегории.
                Иначе возвращается только базовая информация (название, автор, версия, дата создания, лицензия)
        Returns:
            dict: словарь вида {ID : информация о датасете}
        """
        if not os.path.exists(self._dbfile):
            self.create_sqlite_file()
        query = self.sess.query(self.Dataset)
        df = pd.read_sql(query.statement, query.session.bind)
        df_rec = df.to_dict(orient='records')
        df_dict = {}
        for el in df_rec:
            df_dict[el['ID']] = el
            if full_info:
                df_dict[el['ID']]['categories'] = self.get_dataset_categories_info(el['ID'])
            del df_dict[el['ID']]['ID']
        return df_dict

    def load_specific_datasets_annotations(self, datasets_ids, normalize_cats=False, **kwargs):
        """ Загружает аннотации из указанных датасетов по их ID

        Args:
            datasets_ids (list): список ID датасетов, из которых нужно загрузить аннотации
            normalize_cats (bool): Если True, то категории будут нормализованы,
                т.е. их ID будут приведены к диапазону [0, N-1], где N - количество категорий
        Returns:
            DataFrame: pandas dataframe с аннотациями для указанных датасетов
        """
        query = self.sess.query(self.Image.file_name, self.Annotation.category_id, 
                                self.Annotation.bbox, self.Annotation.segmentation
                                ).join(self.Annotation).filter(self.Image.dataset_id.in_(datasets_ids))
        df = pd.read_sql(query.statement, query.session.bind)
        # a fancy patch for keras to start numbers from 0
        if normalize_cats:
            df_new = pd.DataFrame(columns=['images', 'target'], data=df[[
                                  'file_name', 'category_id']].values)
            df_new['target'] = df_new['target'].astype('category').cat.codes
            return df_new
        return df

    def get_all_categories(self):
        """ Возвращает pandas dataframe со всеми доступными категориями в базе данных """
        query = self.sess.query(self.Category)
        df = pd.read_sql(query.statement, query.session.bind)
        return df

    def get_dataset_categories_info(self, ds_id) -> dict:
        """ Возвращает информацию о категориях в датасете по его ID

        Args:
            ds_id (int): ID датасета в базе данных (ID можно получить, например, методом get_all_datasets)
        Returns:
            dict: словарь вида { надкатегория : { категория : количество изображений в датасете } }
        """
        reply = self.sess.query(self.Category.supercategory, self.Category.name,
            func.count(self.Annotation.category_id)
            ).join(self.Image, self.Category
                ).filter(self.Image.dataset_id == ds_id
                    ).group_by(self.Annotation.category_id).all()
        res = {}
        for supercategory, category, number in reply:
            if supercategory not in res:
                res[supercategory] = {}
            res[supercategory][category] = number
        return res

    def _prepare_cropped_images(self, df, cropped_dir='', files_dir='', num_processes=..., skip_existing=True):
        """ Вспомогательный метод для обрезки изображений в случае, когда на одном изображении несколько аннотаций """
        cropped_dir = cropped_dir or 'buf_crops/'
        Path(cropped_dir).mkdir(parents=True, exist_ok=True)

        # iterate with progress bar
        new_images = 0
        new_rows = []
        crop_tasks = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Collecting images', file=sys.stdout,
                               mininterval=0.5, maxinterval=0.6, ncols=79):
            bbox = []
            input_file = os.path.join(files_dir, row['file_name'])
            if row['bbox'] != '':
                bbox = ast.literal_eval(row['bbox'])
            if len(bbox) != 4:
                new_rows.append([input_file, row['category_id']])
                continue
            buf_name = row["file_name"].split('.')
            filename = (buf_name[-2]).split('/')[-1]
            filepath = os.path.join(cropped_dir, f'{filename}-{row["ID"]}-{row["category_id"]}.{buf_name[-1]}')
            if not skip_existing or not os.path.exists(filepath):
                crop_tasks.append((input_file, filepath, bbox[0], bbox[1], bbox[2], bbox[3]))
                new_images += 1
            new_rows.append([filepath, row['category_id']])

        # create multiprocessing pool to speed up the process
        if num_processes is ...:
            num_processes = _default_num_processes
        if new_images:
            print(f'Cropping {new_images} with {num_processes} processes')
            if num_processes > 1:
                pool = multiprocessing.Pool(processes=num_processes)
                # run the pool with progress bar
                for _ in tqdm(pool.imap_unordered(_crop_image_tuple, crop_tasks), total=len(crop_tasks),
                              desc='Cropping images', file=sys.stdout, mininterval=0.5, maxinterval=0.6, ncols=79):
                    pass
                pool.close()
            else:
                for task in tqdm(crop_tasks, desc='Cropping images', file=sys.stdout,
                                 mininterval=0.5, maxinterval=0.6, ncols=79):
                    _crop_image_tuple(task)
            print(f'Created {new_images} new image crops, used {df.shape[0] - new_images} existing image crops')
        # convert list of dicts to dataframe
        buf_df = pd.DataFrame(new_rows, columns=['file_name', 'category_id'])
        return buf_df

    def _split_and_save(self, df, save_dir, split_points, headers_string):
        """
        Helper for storing csv files with annotations returned
        """
        """ Вспомогательный метод для сохранения csv-файлов с аннотациями """
        train_end = int(split_points[0] * len(df))
        val_end = int(split_points[1] * len(df))
        train, validate, test = np.split(df.sample(frac=1), [train_end, val_end])  # we shuffle and split
        np.savetxt(f'{save_dir}train.csv', train, delimiter=",", fmt='%s', header=headers_string, comments='')
        np.savetxt(f'{save_dir}test.csv', test, delimiter=",", fmt='%s', header=headers_string, comments='')
        np.savetxt(f'{save_dir}val.csv', validate, delimiter=",", fmt='%s', header=headers_string, comments='')
        return {'train': f'{save_dir}train.csv', 'test': f'{save_dir}test.csv', 'validate': f'{save_dir}val.csv'}

    def _process_query(self, query, cat_names, with_segmentation, crop_bbox=False, split_points=(0.6, 0.8),
                       normalize_cats=False, balance_by_min_category=False, balance_by_categories=None,
                       cur_experiment_dir='.', **kwargs):
        """ Вспомогательный метод для обработки SQL-запроса для аннотаций

        Args:
            query: запрос sqlalchemy
            cat_names: имена категорий
            with_segmentation: если True, будут возвращены только аннотации с сегментацией
            crop_bbox: если True, будет выполнено вырезание объектов по bbox, заданным в аннотациях, и
                сохранение их в отдельные файлы в директории cur_experiment_dir
            split_points: квантили для разделения на обучающую, тестовую и валидационную выборки (по умолчанию 0.6,0.8)
            normalize_cats: устанавливается только для тестовых целей
            balance_by_min_category: Если True, будет выполнено балансирование
                по минимальному количеству изображений в категории
            balance_by_categories: словарь с количеством аннотаций, которые нужно получить для каждой категории
                (например, {'cat':100,'dog':200}).
        """

        df = pd.read_sql(query.statement, query.session.bind)
        av_width = df['width'].mean()
        av_height = df['height'].mean()
        if crop_bbox:
            df = self._prepare_cropped_images(df, **kwargs)
        elif 'files_dir' in kwargs:
            df['file_name'] = df['file_name'].apply(lambda x: os.path.join(kwargs['files_dir'], x))

        if with_segmentation is False:
            df_new = pd.DataFrame(columns=['images', 'target'], 
                                  data=df[['file_name', 'category_id']].values)
            headers_string = 'images,target'
        else:
            df_new = pd.DataFrame(columns=['images', 'target', 'segmentation'], 
                                  data=df[['file_name', 'category_id', 'segmentation']].values)
            df_new.dropna(subset=['segmentation'], inplace=True)
            headers_string = 'images,target,segmentation'
        if balance_by_min_category:
            g = df_new.groupby('target', group_keys=False)
            # balance-out too large categories with random selection:
            df_new = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        cat_ids = self.get_cat_IDs_by_names(cat_names)
        if balance_by_categories:
            cat_names = [el for el in balance_by_categories]
            cat_ids_dict = {}
            for i in range(len(cat_ids)):
                if cat_ids[i] != -1:
                    cat_ids_dict[cat_ids[i]] = balance_by_categories[cat_names[i]]
            g = df_new.groupby('target', group_keys=False)
            # balance by given nums:
            df_new = g.apply(lambda x: x.sample(cat_ids_dict[x['target'].iloc[0]]).reset_index(drop=True))
        # A fancy patch for keras to start numbers from 0
        if normalize_cats:
            # change category ids to form range(0, num_cats) according to order in cat_names
            cat_ids_dict = {}
            for i in range(len(cat_ids)):
                if cat_ids[i] != -1:
                    cat_ids_dict[cat_ids[i]] = i
            df_new['target'] = df_new['target'].map(cat_ids_dict)
        if not isinstance(split_points, (list, tuple)) or len(split_points) != 2:
            raise ValueError('split_points must be a list of two elements')
        filename_dict = self._split_and_save(df_new, cur_experiment_dir + '/',
                                             split_points, headers_string)
        return df_new, filename_dict, av_width, av_height

    def load_specific_categories_annotations(self, cat_names, with_segmentation=False, **kwargs):
        """
        Метод для загрузки аннотаций из конкретных категорий, заданных их ID.

        Args:
            cat_names (list): список ID категорий, аннотации из которых нужно получить
            with_segmentation (bool): если True, будут возвращены только аннотации с сегментацией
        Returns:
            (DataFrame,dict,float,float): кортеж (df, filename_dict, av_width, av_height), где:
               - df - pandas dataframe с полными аннотациями для заданных cat_ids
               - filename_dict - словарь с именами файлов для train, test, val
               - av_width, av_height - средняя ширина и высота изображений
        """
        if self.ds_filter is not None:
            return self.load_categories_datasets_annotations(cat_names, self.ds_filter, with_segmentation, **kwargs)
        query = self.sess.query(self.Image.file_name, self.Image.coco_url, self.Annotation.category_id,
                                self.Annotation.bbox, self.Annotation.segmentation, self.Image.width, self.Image.height,
                                self.Annotation.ID
                                ).join(self.Annotation).join(self.Category).filter(self.Category.name.in_(cat_names))
        if with_segmentation:
            # lengths 0 and 1 do not work because there may be strings with empty brackets in DB
            query = query.filter(func.length(self.Annotation.segmentation) > 2)
        return self._process_query(query, cat_names, with_segmentation, **kwargs)

    def load_categories_datasets_annotations(self, cat_names, datasets_ids, with_segmentation=False, **kwargs):
        """
        Метод для загрузки аннотаций из конкретных категорий и датасетов, заданных их ID.

        Args:
            cat_names (list): список ID категорий, аннотации из которых нужно получить
            datasets_ids (list): список ID датасетов, аннотации из которых нужно получить
            with_segmentation (bool): если True, будут возвращены только аннотации с сегментацией
        Returns:
            (DataFrame,dict,float,float): кортеж (df, filename_dict, av_width, av_height), где:
                - df - pandas dataframe с полными аннотациями для заданных cat_ids
                - filename_dict - словарь с именами файлов для train, test, val
                - av_width, av_height - средняя ширина и высота изображений
        """
        datasets_ids = [self.get_dataset_id(ds) for ds in datasets_ids]
        query = self.sess.query(self.Image.file_name, self.Image.coco_url, self.Annotation.category_id,
                                self.Annotation.bbox, self.Annotation.segmentation, self.Image.width, self.Image.height,
                                self.Annotation.ID
                                ).join(self.Annotation).join(self.Category).filter(self.Category.name.in_(cat_names)).filter(self.Image.dataset_id.in_(datasets_ids))
        if with_segmentation:
            # lengths 0 and 1 do not work since there are records with empty brackets
            query = query.filter(func.length(self.Annotation.segmentation) > 2)
        return self._process_query(query, cat_names, with_segmentation, **kwargs)

    def load_specific_categories_from_specific_datasets_annotations(self, dataset_categories_dict, normalize_cats=False,
                                                                    with_segmentation=False, split_points=(0.6, 0.8),
                                                                    cur_experiment_dir='.',
                                                                    **kwargs):
        """
        Метод для загрузки аннотаций из конкретных датасетов, отфильтрованных по конкретным категориям
        (отличается от load_categories_datasets_annotations).

        Args:
            dataset_categories_dict: словарь соответствия категорий, структура: {datasetID1 : [cat1,cat2], datasetID2: [cat1,cat2,cat3], ...}
            normalize_cats: если True, ID категорий будут нормализованы в диапазоне(0, num_cats)
            with_segmentation: если True, будут возвращены только аннотации с сегментацией
            split_points: список из двух элементов, точки разделения train, val, test
            cur_experiment_dir: директория для сохранения файлов train, test, val
            kwargs: дополнительные параметры.
        Returns:
            (DataFrame,dict,float,float): кортеж (df, filename_dict, av_width, av_height), где:
                - df - pandas dataframe с полными аннотациями для заданных cat_ids
                - filename_dict - словарь с именами файлов для train, test, val
                - av_width, av_height - средняя ширина и высота изображений
        """
        result = None
        sum_width = 0
        sum_height = 0

        for key in dataset_categories_dict:
            cat_names = dataset_categories_dict[key]
            datasets_ids = [key]
            result_vec = self.load_categories_datasets_annotations(cat_names, datasets_ids, normalize_cats=False,
                                                                   with_segmentation=with_segmentation, **kwargs)
            if result is None:
                result = result_vec[0]
            else:
                result = pd.concat([result, result_vec[0]])
            sum_width += len(result.index) * result_vec[2]
            sum_height += len(result.index) * result_vec[3]

        if normalize_cats:
            result['target'] = result['target'].astype('category').cat.codes
        filename_dict = self._split_and_save(result, cur_experiment_dir + '/',
                                             split_points, ','.join(list(result.columns)))
        return result, filename_dict, sum_width / len(result.index), sum_height / len(result.index)

    def load_specific_images_annotations(self, image_names, normalize_cats=True) -> pd.DataFrame:
        """
        Метод для загрузки аннотаций из конкретных изображений, заданных их именами.

        Args:
            image_names (list of str): список имен изображений, для которых нужно получить аннотации
            normalize_cats: если True, ID категорий будут нормализованы в диапазоне(0, num_cats)
                    (например, cats,dogs(17,18) -> (0,1))
        Returns:
            DataFrame: pandas dataframe с аннотациями для заданных image_names
        """
        query = self.sess.query(self.Image.file_name, self.Annotation.category_id,
                                self.Annotation.bbox, self.Annotation.segmentation
                                ).join(self.Annotation).filter(self.Image.file_name.in_(image_names))
        df = pd.read_sql(query.statement, query.session.bind)
        if normalize_cats:
            df_new = pd.DataFrame(columns=['images', 'target'],
                                  data=df[['file_name', 'category_id']].values)
            df_new['target'] = df_new['target'].astype('category').cat.codes
            return df_new
        return df

    def add_categories(self, categories, respect_ids=True):
        """
        Метод для добавления заданных категорий в базу данных.

        Args:
            categories (list of dicts): список словарей с необходимыми полями: supercategory, name, id
            respect_ids (bool): указывает, сохраняются ли идентификаторы из словаря в БД
        """
        for category in categories:
            _supercategory = category['supercategory']
            _name = category['name']
            _id = None
            if respect_ids is True:
                _id = category['id']
            try:
                self.sess.query(self.Category).filter_by(
                    supercategory=_supercategory, name=_name, ID=_id,).one()
            except NoResultFound:
                new_cat = self.Category(_supercategory, _name, _id)
                self.sess.add(new_cat)
        self.sess.commit()  # adding categories in db

    def add_default_licences(self):
        """Метод для добавления лицензий по умолчанию в БД. Ожидается вызываться один раз"""
        licenses = [{"url": "https://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                     "name": "Attribution-NonCommercial-ShareAlike License"},
                    {"url": "https://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
                     "name": "Attribution-NonCommercial License"},
                    {"url": "https://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
                     "name": "Attribution-NonCommercial-NoDerivs License"},
                    {"url": "https://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"},
                    {"url": "https://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
                     "name": "Attribution-ShareAlike License"},
                    {"url": "https://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
                     "name": "Attribution-NoDerivs License"},
                    {"url": "https://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"},
                    {"url": "https://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}]
        for license in licenses:
            lic = self.License(license['name'], license['url'], license['id'])
            self.sess.add(lic)
        self.sess.commit()  # adding licenses from dogs_vs_cats.json

    def add_dataset_info(self, dataset_info):
        """Метод для добавления информации о новом датасете
        
        Args:
            dataset_info (dict): словарь с информацией о датасете
        Returns:
            int: ID добавленного датасета
        """
        dataset = self.Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'],
                               dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit()  # adding dataset
        return dataset.ID

    def add_images_and_annotations(self, images, annotations, dataset_id, file_prefix='',
                                   respect_ids=False, category_id_map=None):
        """
        Метод для добавления части изображений и их аннотаций в БД.

        Args:
            images (list of dict): массив словарей с атрибутами:
                license, file_name, coco_url, height, width, date_captured, flickr_url, id
            annotations (list of dict): массив словарей с атрибутами:
                segmentation, area, iscrowd, image_id, bbox, category_id, id
            dataset_id (int): ID датасета, из которого изображения
            file_prefix (str): префикс, добавляемый к именам файлов (обычно путь к папке с изображениями)
            respect_ids (bool): указывает, сохраняются ли идентификаторы из словаря в БД
        """

        if file_prefix != '' and file_prefix[-1] not in ['\\', '/']:
            file_prefix += '/'
        if not os.path.isfile(file_prefix + images[0]['file_name']):
            print('Error in json file, missing images stored on disc (i.e.',
                  file_prefix + images[0]['file_name'], ')')
            raise FileNotFoundError(f'Error in json file, missing images stored on disc (i.e. {file_prefix + images[0]["file_name"]})')
        buf_images = {}
        for im_data in tqdm(images, desc='Adding images', total=len(images), file=sys.stdout, ncols=79):
            im_id = None
            if respect_ids is True:
                im_id = im_data['id']
            image = self.Image(file_name=file_prefix + im_data['file_name'],
                               width=im_data['width'],
                               height=im_data['height'],
                               date_captured=im_data['date_captured'],
                               dataset_id=dataset_id,
                               coco_url=im_data['coco_url'],
                               flickr_url=im_data['flickr_url'],
                               license_id=im_data['license'],
                               ID=im_id)
            buf_images[im_data['id']] = image
            self.sess.add(image)
        self.sess.commit()  # adding images
        for an_data in tqdm(annotations, desc='Adding annotations', total=len(annotations), file=sys.stdout, ncols=79):
            anno_id = None
            if respect_ids is True:
                anno_id = an_data['id']
            cur_image_id = buf_images[an_data['image_id']].ID
            seg_str = json.dumps(an_data['segmentation'])
            bbox_str = json.dumps(an_data['bbox'])
            cat_id = an_data['category_id'] if category_id_map is None else category_id_map[an_data['category_id']]
            annotation = self.Annotation(image_id=cur_image_id,
                                         category_id=cat_id,
                                         bbox=bbox_str,
                                         segmentation=seg_str,
                                         is_crowd=an_data['iscrowd'],
                                         area=an_data['area'],
                                         ID=anno_id)
            self.sess.add(annotation)
        self.sess.commit()  # adding annotations

    def add_model_record(self, task_type, categories, model_address, metrics, history_address=''):
        """
        Добавляет записи о результатах обучения модели.
        Если модель уже существует, метод не создает новую модель.
        Если ключ update_metrics установлен в True, то записи о метриках будут обновлены, если они уже существуют

        Args:
            task_type (str): тип задачи, для которой обучалась модель
            categories (list): список категорий, которые может классифицировать модель
            model_address (str): путь к файлу модели
            metrics (dict): словарь со значениями метрик
            history_address (str): путь к файлу истории
        Raises:
            TypeError: если один из аргументов имеет неправильный тип
            ValueError: если некоторые категории отсутствуют в БД
        """
        if not isinstance(task_type, str):
            raise TypeError(f'task_type must be a string but {type(task_type)} was given')
        if not isinstance(categories, list):
            raise TypeError(f'categories must be a list but {type(categories)} was given')
        if not isinstance(model_address, str):
            raise TypeError(f'model_address must be a string but {type(model_address)} was given')
        if not isinstance(metrics, dict):
            raise TypeError(f'metrics must be a dictionary with metrics but {type(metrics)} was given')

        abs_model_address = os.path.abspath(model_address)
        abs_history_address = os.path.abspath(history_address)
        # model should be identified by its address uniquely
        if self.sess.query(self.Model).filter(self.Model.model_address == abs_model_address).first() is not None:
            warnings.warn(f"model {abs_model_address} was already added in DB. Check model_address for correctness.")
            return

        # Model not in DB - add it (that's OK)
        new_model = self.Model(model_address=abs_model_address, task_type=task_type)
        self.sess.add(new_model)
        self.sess.commit()
        for cat_name in categories:  # categories should not change for the model - they are attached once
            category_from_db = self.sess.query(self.Category).filter(self.Category.name == cat_name).first()
            if category_from_db is None:
                # That's a very bad case - we cannot simply add new category, DB may become inconsistent
                raise ValueError(f'Category {cat_name} does not exist in DB (possibly incorrect categories list)')
            new_cat_record = self.CategoryToModel(category_from_db.ID, new_model.ID)
            self.sess.add(new_cat_record)
        self.sess.commit()

        # We do not check if metric is valid - module user should keep track on consistency of these records
        for key, value in metrics.items():
            new_train_result = self.TrainResult(metric_name=key,
                                                metric_value=value,
                                                model_id=new_model.ID,
                                                history_address=abs_history_address)
            self.sess.add(new_train_result)
        self.sess.commit()

    def update_train_result_record(self, model_address, metric_name, metric_value, history_address=''):
        """
        Обновляет запись о результатах обучения модели.

        Args:
            model_address (str): путь к файлу с сохраненной моделью
            metric_name (str): название метрики
            metric_value (float): значение метрики
            history_address (str): путь к файлу с историей обучения
        Returns:
            True, если запись была успешно обновлена, иначе False (например, если запись не существует)
        Raises:
            TypeError: если model_address не является строкой или metric_name не является строкой
        """
        if not isinstance(model_address, str):
            raise TypeError(f'model_address must be a string but {type(model_address)} was given')
        if not isinstance(metric_name, str):
            raise TypeError(f'metric_name must be a string but {type(metric_name)} was given')

        abs_model_address = os.path.abspath(model_address)
        abs_history_address = os.path.abspath(history_address)
        model_from_db = self.sess.query(self.Model).filter(self.Model.model_address == abs_model_address).first()
        if model_from_db is None:
            print('ERROR: Model does not exist in database')
            return False

        for trRes in model_from_db.train_results:
            if trRes.metric_name == metric_name:
                trRes.metric_value = metric_value
                trRes.history_address = abs_history_address
                self.sess.commit()
                return True
        # no such metric was found - add new train result
        print('Adding new metric for this model, metric name:', metric_name)
        new_train_result = self.TrainResult(metric_name=metric_name,
                                            metric_value=metric_value,
                                            model_id=model_from_db.ID,
                                            history_address=abs_history_address)
        self.sess.add(new_train_result)
        self.sess.commit()
        return True

    def delete_train_result_record(self, model_address, metric_name):
        """
        Удаляет запись о результате обучения из БД с указанным именем метрики для указанной модели

        Args:
            model_address (str): путь к файлу с сохраненной моделью
            metric_name (str): название метрики
        Returns:
            True, если запись была успешно удалена, иначе False (например, если запись не существует)
        Raises:
            TypeError: если аргументы имеют неправильный тип
        """
        if not isinstance(model_address, str):
            raise TypeError(f'model_address must be a string but {type(model_address)} was given')
        if not isinstance(metric_name, str):
            raise TypeError(f'metric_name must be a string but {type(metric_name)} was given')

        abs_model_address = os.path.abspath(model_address)
        model_from_db = self.sess.query(self.Model).filter(self.Model.model_address == abs_model_address).first()
        if model_from_db is None:
            print('ERROR: Model does not exist in database')
            return False

        for trRes in model_from_db.train_results:
            if trRes.metric_name == metric_name:
                self.sess.query(self.TrainResult).filter_by(ID=trRes.ID).delete()
                self.sess.commit()
                return True
        print('ERROR: Such train result record was not found')
        return False

    def get_models_by_filter(self, filter_dict, exact_category_match=False):
        """
        Возвращает список моделей, которые соответствуют filter_dict

        Args:
            filter_dict (dict): словарь, который содержит параметры поиска модели.
                Специфика этой структуры может меняться со временем.

                В настоящее время поддерживаются пары ключ-значение:
                  - 'min_metrics': {'metric_name': min_value}
                  - 'categories': ['list','of','categories','names']
            exact_category_match (bool):
                Если True, то будут возвращены только модели, которые имеют точно такие же категории.
        Returns:
            DataFrame: pandas DataFrame с информацией о моделях
        """
        model_query = self.sess.query(self.Model, self.TrainResult).join(self.TrainResult).join(self.CategoryToModel).join(self.Category)
        if 'categories_ids' in filter_dict:
            filter_dict['categories'] = self.get_cat_names_by_IDs(filter_dict['categories_ids'])  # TODO: this is just a patch
            print(filter_dict['categories'])
        if 'categories' in filter_dict:
            model_query = model_query.filter(self.Category.name.in_(filter_dict['categories']))
            # TODO: maybe exact category matches need to be done later
        if 'min_metrics' in filter_dict:
            if not isinstance(filter_dict['min_metrics'], dict):
                raise ValueError('min_metrics should be a dict')
            for key, value in filter_dict['min_metrics'].items():
                model_query = model_query.filter(and_(self.TrainResult.metric_value >= value,
                                                      self.TrainResult.metric_name == key))
        # I didn't yet find a way to make this better in performance
        model_query = model_query.group_by(self.Model.model_address)
        if 'categories' in filter_dict and len(filter_dict['categories']) != 0:
            cands = model_query.all()
            good_IDs = []
            for cand_buf in cands:
                cand = cand_buf[0]
                model_cand = self.sess.query(self.Model).filter(self.Model.ID == cand.ID).first()
                model_categories = set()
                cat_ids = []
                for cat in model_cand.categories:
                    cat_ids.append(cat.category_id)
                mod_cats = self.sess.query(self.Category).filter(self.Category.ID.in_(cat_ids))
                for mod_cat in mod_cats:
                    model_categories.add(mod_cat.name)
                if model_categories.issubset(set(filter_dict['categories'])):
                    good_IDs.append(model_cand.ID)
            model_query = model_query.filter(self.Model.ID.in_(good_IDs))
        df = pd.read_sql(model_query.statement, model_query.session.bind)
        return df

    def get_cat_IDs_by_names(self, cat_names):
        """
        Args:
            cat_names (list of str): список имен категорий.
        Returns:
            list: список ID категорий, соответствующих именам (в том же порядке).
                Если категория отсутствует, возвращается -1 на ее позиции.
        Raises:
            TypeError: если cat_names не является списком
        """
        if not isinstance(cat_names, list):
            raise TypeError(f'Cat_names, must be a list, but {type(cat_names)} was given')
        result = []
        for cat_name in cat_names:
            query = self.sess.query(self.Category.ID).filter(self.Category.name == cat_name).first()
            if query is None:
                result.append(-1)
            else:
                result.append(query[0])
        return result

    def get_cat_names_by_IDs(self, cat_ids):
        """
        Args:
            cat_ids (list of int): список ID категорий.
        Returns:
            list: список имен категорий, соответствующих ID (в том же порядке).
                Если категория отсутствует, возвращается "" на ее позиции.
        Raises:
            TypeError: если cat_ids не является списком
        """
        if not isinstance(cat_ids, list):
            raise TypeError(f'Cat_names, must be a list, but {type(cat_ids)} was given')
        result = []
        for cat_ID in cat_ids:
            query = self.sess.query(self.Category.name).filter(self.Category.ID == cat_ID).first()
            if query is None:
                result.append("")
            else:
                result.append(query[0])
        return result

    def get_dataset_id(self, dataset_id_or_name):
        """
        Args:
            dataset_id_or_name (int or str): ID или имя датасета.
                Если dataset_id_or_name - int, то считается, что это ID, иначе считается, что это имя.
        Returns:
            int: ID датасета с заданным именем или ID. Если не найден, возвращает -1.
        """
        if not isinstance(dataset_id_or_name, str):
            return int(dataset_id_or_name)
        query = self.sess.query(self.Dataset.ID).filter(self.Dataset.description == dataset_id_or_name).first()
        if query is None:
            return -1
        return query[0]

    def get_full_dataset_info(self, ds_id):
        """
        Args:
            ds_id (int): ID датасета в базе данных (ID можно получить, например, методом get_all_datasets)
        Returns:
            dict: полная информация о датасете по его ID в базе данных в виде словаря
              Словарь содержит следующие ключи:
                - 'dataset_info' -> краткая информация о датасете
                - 'categories' -> число конкретных категорий в этом датасете
        """
        result = {}
        query = self.sess.query(self.Dataset).filter(self.Dataset.ID == ds_id)
        df = pd.read_sql(query.statement, query.session.bind)
        result = df.to_dict('list')
        for key in result:
            # to be consistent with ds_info default dict
            result[key] = result[key][0]
        cat_query = self.sess.query(self.Annotation.category_id) \
            .join(self.Image).filter(self.Image.dataset_id == ds_id)
        cat_counts = self.sess.query(self.Annotation.category_id, func.count(self.Annotation.category_id)) \
            .join(self.Image).filter(self.Image.dataset_id == ds_id).group_by(self.Annotation.category_id).all()
        cat_counts_dict = {}
        for record in cat_counts:
            cat_counts_dict[record[0]] = record[1]
        categories_in_ds = cat_query.group_by(
            self.Annotation.category_id).all()
        categories_in_ds = list(np.array(categories_in_ds).flatten())
        cats_df = self.get_all_categories()
        result['categories'] = {}
        for cat_id in categories_in_ds:
            cat_name = cats_df.loc[cats_df['ID'] == cat_id].values[0][2]
            cat_count = cat_counts_dict[cat_id]
            result['categories'][cat_name] = cat_count
        return result

    def close(self):
        """ Закрывает соединение с базой данных """
        self.sess.close()
        self.engine.dispose()

    def get_all_category_names(self) -> "list[str]":
        """
        Returns:
            list: имена всех категорий в базе данных
        """
        query = self.sess.query(self.Category.name)
        df = pd.read_sql(query.statement, query.session.bind)
        return df['name'].tolist()
