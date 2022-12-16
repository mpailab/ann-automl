from ann_automl.core.nn_auto import create_classification_model

import ann_automl.core.db_module as db

import pytest
import os
import shutil
import tempfile

from ann_automl.core.nnfuncs import set_db, db_context

# current file path
file_path = os.path.dirname(os.path.abspath(__file__))
# path to test datasets
test_datasets_path = os.path.join(file_path, 'datasets')


@pytest.fixture(scope='module')
def db_dir():
    db_dir = tempfile.mkdtemp()
    with db.num_processes_context(1):
        yield db_dir
        shutil.rmtree(db_dir)


@pytest.mark.usefixtures('db_dir')
def test_create_model(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite", dbconf_file=f"{test_datasets_path}/dbconf.json")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir+'/test.sqlite')

    mydb.fill_in_coco_format(f'{test_datasets_path}/test3/a.json',
                             file_prefix=f'{test_datasets_path}/test3/images/',
                             ds_info={'description': 'test3',
                                      'url': f'{test_datasets_path}/test3/',
                                      'version': '1.0',
                                      'year': 2020,
                                      'contributor': 'test',
                                      'date_created': '2020-01-01'},
                             auto_download=True)

    with db_context(mydb):
        classes = ['cat', 'dog', 'elephant']
        target_accuracy = 0.9
        create_classification_model(classes, target_accuracy, os.path.join(db_dir, 'output'), time_limit=20)



if __name__ == '__main__':
    db_dir = tempfile.mkdtemp()
    test_create_model(db_dir)
    shutil.rmtree(db_dir)

