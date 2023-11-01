import ann_automl.core.db_module as db
from ann_automl.core.smart_labeling import labeling

# make temporary directory for test database files
import tempfile
import os
from datetime import datetime, date
import pytest

@pytest.fixture(scope='function')
def db_dir():
    db_dir = tempfile.mkdtemp()
    yield db_dir

@pytest.mark.usefixtures('db_dir')
def test_upload_from_drive(db_dir, monkeypatch):
    """
            Создает словарь в формате, пригодном для использования разметчиком qsl.
            Args:
                db_dir (str): путь к временному каталогу, в котором хранится база данных
                monkeypatch: необходим для передачи в тест параметров, вместо входного потока

            В path_to_images указывается путь до директории с добавляемыми изображениями. В качестве итерируемого
            объекта подается список параметров: 1) способ добавления изображений
                                                2) Название датасета
                                                3) Нейросеть, используемая для разметки
                                                4) Путь до директории с изображениями
    """
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "https://drive.google.com/file/d/1aDlgJaDw4RPNxySbrluWJMsc5XHJ_JiA/view?usp=sharing"
    input_data = iter(["3", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert len(mydb.get_all_categories()['name']) != 0
    assert set(mydb.get_all_categories()['name']) == {'bird', 'truck', 'stop sign', 'cat', 'suitcase', 'car', 'bed',
                                                      'motorcycle', 'dog', 'person', 'traffic light', 'handbag'}
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_upload_from_folder(db_dir, monkeypatch):
    """
            Создает словарь в формате, пригодном для использования разметчиком qsl.
            Args:
                db_dir (str): путь к временному каталогу, в котором хранится база данных
                monkeypatch: необходим для передачи в тест параметров, вместо входного потока

            В path_to_images указывается путь до директории с добавляемыми изображениями. В качестве итерируемого
            объекта подается список параметров: 1) способ добавления изображений
                                                2) Название датасета
                                                3) Нейросеть, используемая для разметки
                                                4) Путь до директории с изображениями
    """
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "C:\\Users\\Alexander\\Documents\\tests\\pictures"
    input_data = iter(["1", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert len(mydb.get_all_categories()['name']) != 0
    assert set(mydb.get_all_categories()['name']) == {'bird', 'truck', 'stop sign', 'cat', 'suitcase', 'car', 'bed',
                                                      'motorcycle', 'dog', 'person', 'traffic light', 'handbag'}
    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_upload_from_zip_archive(db_dir, monkeypatch):
    """
            Создает словарь в формате, пригодном для использования разметчиком qsl.
            Args:
                db_dir (str): путь к временному каталогу, в котором хранится база данных
                monkeypatch: необходим для передачи в тест параметров, вместо входного потока

            В path_to_images указывается путь до директории с добавляемыми изображениями. В качестве итерируемого
            объекта подается список параметров: 1) способ добавления изображений
                                                2) Название датасета
                                                3) Нейросеть, используемая для разметки
                                                4) Путь до директории с изображениями
    """
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "C:\\Users\\Alexander\\Documents\\tests\\Image_examples.zip"
    input_data = iter(["2", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert len(mydb.get_all_categories()['name']) != 0
    assert set(mydb.get_all_categories()['name']) == {'bird', 'truck', 'stop sign', 'cat', 'suitcase', 'car', 'bed',
                                                      'motorcycle', 'dog', 'person', 'traffic light', 'handbag'}
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_mixed_upload(db_dir, monkeypatch):
    """
            Создает словарь в формате, пригодном для использования разметчиком qsl.
            Args:
                db_dir (str): путь к временному каталогу, в котором хранится база данных
                monkeypatch: необходим для передачи в тест параметров, вместо входного потока

            В path_to_images указывается путь до директории с добавляемыми изображениями. В качестве итерируемого
            объекта подается список параметров: 1) способ добавления изображений
                                                2) Название датасета
                                                3) Нейросеть, используемая для разметки
                                                4) Путь до директории с изображениями
    """
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    local_path_to_images = "C:\\Users\\Alexander\\Documents\\tests\\pictures"
    local_path_to_zip = "C:\\Users\\Alexander\\Documents\\tests\\Image_examples.zip"
    google_drive_path = "https://drive.google.com/file/d/1aDlgJaDw4RPNxySbrluWJMsc5XHJ_JiA/view?usp=sharing"
    input_data = iter(["2", local_path_to_zip, "My_dataset", "yolov5x",
                       "1", local_path_to_images, "My_dataset_2", "yolov5x",
                       "3", google_drive_path, "My_dataset_3", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    assert len(mydb.get_all_datasets()) == 3
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert len(mydb.get_all_categories()['name']) != 0
    assert set(mydb.get_all_categories()['name']) == {'bird', 'truck', 'stop sign', 'cat', 'suitcase', 'car', 'bed',
                                                      'motorcycle', 'dog', 'person', 'traffic light', 'handbag'}
    mydb.close()

def test_different_nn(db_dir, monkeypatch):
    """
            Создает словарь в формате, пригодном для использования разметчиком qsl.
            Args:
                db_dir (str): путь к временному каталогу, в котором хранится база данных
                monkeypatch: необходим для передачи в тест параметров, вместо входного потока

            В path_to_images указывается путь до директории с добавляемыми изображениями. В качестве итерируемого
            объекта подается список параметров: 1) способ добавления изображений
                                                2) Название датасета
                                                3) Нейросеть, используемая для разметки
                                                4) Путь до директории с изображениями
    """
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    local_path_to_images = "C:\\Users\\Alexander\\Documents\\tests\\pictures"
    input_data = iter(["1", local_path_to_images, "My_dataset", "yolov5s",
                       "1", local_path_to_images, "My_dataset_2", "yolov5l",
                       "1", local_path_to_images, "My_dataset_3", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)
    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    len_one_dataset = len(mydb.get_all_categories()['name'])
    assert  len_one_dataset != 0

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)
    assert len(mydb.get_all_datasets()) == 2
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    len_two_datasets = len(mydb.get_all_categories()['name'])
    assert len_two_datasets != 0
    assert len_two_datasets > len_one_dataset

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)
    assert len(mydb.get_all_datasets()) == 3
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    len_three_datasets = len(mydb.get_all_categories()['name'])
    assert len_three_datasets != 0
    assert len_three_datasets > len_two_datasets
    assert set(mydb.get_all_categories()['name']) == {'bird', 'truck', 'stop sign', 'cat', 'suitcase', 'car', 'bed',
                                                      'motorcycle', 'dog', 'person', 'traffic light', 'handbag',
                                                      'backpack'}
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_presented_dataset(db_dir, monkeypatch):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "C:\\Users\\Alexander\\Documents\\tests\\Image_examples.zip"
    input_data = iter(["2", path_to_images, "My_dataset", "yolov5x", "2", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)

    with pytest.raises(FileNotFoundError, match='The dataset My_dataset being added is already presented'):
        dataset_dir = labeling(db_dir)
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_empty_folder(db_dir, monkeypatch):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "C:\\Users\\Alexander\\Documents\\tests\\empty_folder"
    input_data = iter(["2", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    error = "No image files in C:\\Users\\Alexander\\Documents\\tests\\empty_folder directory"
    with pytest.raises(FileNotFoundError, match=error):
        dataset_dir = labeling(db_dir)
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_incorrect_path(db_dir, monkeypatch):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "D"
    input_data = iter(["2", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    error = 'No image files in ' + path_to_images + ' directory'
    with pytest.raises(ValueError, match=f"Incorrect path {path_to_images} to images"):
        dataset_dir = labeling(db_dir)
    mydb.close()


