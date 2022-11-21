import ann_automl.core.db_module as db
from ann_automl.utils.text_utils import print_progress_bar

# make temporary directory for test database files
import tempfile
import os
import shutil
import pytest


# current file path
file_path = os.path.dirname(os.path.abspath(__file__))
# path to test datasets
test_datasets_path = os.path.join(file_path, 'datasets')

# At first run download images from coco dataset
db.check_coco_images(f'{test_datasets_path}/test1/annotations1.json', f'{test_datasets_path}/test1/images')
db.check_coco_images(f'{test_datasets_path}/test2/annotations/train.json', f'{test_datasets_path}/test2/images')


@pytest.fixture(scope='module')
def db_dir():
    db_dir = tempfile.mkdtemp()
    yield db_dir
    shutil.rmtree(db_dir)


# Test database creation in temporary directory
@pytest.mark.usefixtures('db_dir')
def test_db(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite", dbconf_file=f"{test_datasets_path}/dbconf.json")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir+'/test.sqlite')

    mydb.fill_coco(f'{test_datasets_path}/test1/annotations1.json',
                   file_prefix=f'{test_datasets_path}/test1/', first_time=True)
    mydb.close()

    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert set(mydb.get_all_categories()['name']) == {'bicycle', 'airplane'}

    mydb.fill_in_coco_format(f'{test_datasets_path}/test2/annotations/train.json',
                             file_prefix=f'{test_datasets_path}/test2/',
                             ds_info={'description': 'test2',
                                      'url': f'{test_datasets_path}/test2/',
                                        'version': '1.0',
                                        'year': 2020,
                                        'contributor': 'test',
                                        'date_created': '2020-01-01'})
    assert len(mydb.get_all_datasets()) == 2
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert set(mydb.get_all_categories()['name']) == {'bicycle', 'airplane', 'cat', 'dog'}
    ids = mydb.get_cat_IDs_by_names(["cat", "dog", "elephant"])
    assert ids == [17, 18, -1]
    cat_names = mydb.get_cat_names_by_IDs(ids)
    assert cat_names == ['cat', 'dog', '']


if __name__ == '__main__':
    if os.path.isdir('tmp/db'):
        shutil.rmtree('tmp/db')
    os.makedirs('tmp/db', exist_ok=True)
    test_db('tmp/db')
