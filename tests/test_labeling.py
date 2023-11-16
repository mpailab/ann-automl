import ann_automl.core.db_module as db
from ann_automl.core.smart_labeling import labeling, upload_images, extract_id, get_confirm_token, save_response_content


# make temporary directory for test database files
import tempfile
import os
import requests
from datetime import datetime, date
import pytest

@pytest.fixture(scope='function')
def db_dir():
    db_dir = tempfile.mkdtemp()
    yield db_dir

def check_small_test(data_dir):
    if os.path.exists(data_dir):
        return
    zip_url = "https://drive.google.com/file/d/15uH2-TyAh6sS18BhbVsRFelVIfl38iQd/view?usp=drive_link"
    upload_images(zip_url, data_dir)

def check_big_test(data_dir):
    if os.path.exists(data_dir):
        return
    zip_url = "https://drive.google.com/file/d/1QCwvdN3tvsFou7sA2XHPd0jDCbYg9vIw/view?usp=sharing"
    upload_images(zip_url, data_dir)

def check_zip(data_dir):
    if os.path.exists(data_dir):
        return
    zip_url = "https://drive.google.com/file/d/15uH2-TyAh6sS18BhbVsRFelVIfl38iQd/view?usp=drive_link"
    URL = "https://docs.google.com/uc?export=download&confirm=1"
    file_id = extract_id(zip_url)
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, data_dir)

def check_empty_folder(data_dir):
    if os.path.exists(data_dir):
        return
    os.makedirs(data_dir)

@pytest.mark.usefixtures('db_dir')
def test_check(db_dir):
    check_small_test("data/small_test")
    check_big_test("data/big_test")
    check_zip("data/pictures")
    check_empty_folder("data/empty_folder")
    assert os.path.isfile("data/pictures/pictures.zip")


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
    input_data = iter([path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test = True)
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

    path_to_images = "data/small_test"
    input_data = iter([path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test = True)
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

    path_to_images = "data/pictures/pictures.zip"
    input_data = iter([path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test = True)
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

    local_path_to_images = "data/small_test"
    local_path_to_zip = "data/pictures/pictures.zip"
    google_drive_path = "https://drive.google.com/file/d/1aDlgJaDw4RPNxySbrluWJMsc5XHJ_JiA/view?usp=sharing"
    input_data = iter([local_path_to_zip, "My_dataset", "yolov5x",
                       local_path_to_images, "My_dataset_2", "yolov5x",
                       google_drive_path, "My_dataset_3", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test = True)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    dataset_dir = labeling(db_dir, test = True)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    dataset_dir = labeling(db_dir, test = True)
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

    local_path_to_images = "data/small_test"
    input_data = iter([local_path_to_images, "My_dataset", "yolov5s",
                       local_path_to_images, "My_dataset_2", "yolov5l",
                       local_path_to_images, "My_dataset_3", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test = True)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)
    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    len_one_dataset = len(mydb.get_all_categories()['name'])
    assert  len_one_dataset != 0

    dataset_dir = labeling(db_dir, test = True)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)
    assert len(mydb.get_all_datasets()) == 2
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    len_two_datasets = len(mydb.get_all_categories()['name'])
    assert len_two_datasets != 0
    assert len_two_datasets > len_one_dataset

    dataset_dir = labeling(db_dir, test = True)
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

    path_to_images = "data/small_test"
    input_data = iter([path_to_images, "My_dataset", "yolov5x", path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test = True)

    with pytest.raises(FileNotFoundError, match='The dataset My_dataset being added is already presented'):
        dataset_dir = labeling(db_dir, test = True)
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_empty_folder(db_dir, monkeypatch):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "data/empty_folder"
    input_data = iter([path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    error = f"No image files in {path_to_images} directory"
    with pytest.raises(FileNotFoundError, match=error):
        dataset_dir = labeling(db_dir, test = True)
    mydb.close()

@pytest.mark.usefixtures('db_dir')
def test_incorrect_path(db_dir, monkeypatch):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "D"
    input_data = iter([path_to_images, "My_dataset", "yolov5x"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    error = 'No image files in ' + path_to_images + ' directory'
    with pytest.raises(ValueError, match=f"Incorrect path {path_to_images} to images"):
        dataset_dir = labeling(db_dir, test = True)
    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_big_dataset(db_dir, monkeypatch):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    path_to_images = "data/big_test"
    input_data = iter([path_to_images, "My_dataset", "yolov5s"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir, test=True)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations_file.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    assert len(mydb.get_all_datasets()) == 1
    mydb.close()

