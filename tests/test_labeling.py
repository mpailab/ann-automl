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
def test_fill_in_coco_format(db_dir, monkeypatch):
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

    path_to_images = "C:/Users/Alexander/Documents/tests/pictures"
    input_data = iter(["1", "My_dataset", "yolov5x", path_to_images])
    monkeypatch.setattr('builtins.input', lambda _: next(input_data))

    dataset_dir = labeling(db_dir)
    dataset_info = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                    "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    mydb.fill_in_coco_format(dataset_dir + '/annotations/annotations.json',
                             file_prefix=dataset_dir + '/images/', ds_info=dataset_info)

    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert len(mydb.get_all_categories()['name']) != 0
    assert set(mydb.get_all_categories()['name']) == {'bird', 'truck', 'stop sign', 'cat', 'suitcase', 'car', 'bed',
                                                      'motorcycle', 'dog', 'person', 'traffic light', 'handbag'}
    mydb.close()