from ann_automl.core.nn_recommend import recommend_hparams
from ann_automl.core.nn_task import NNTask
from ann_automl.core.nnfuncs import train, tune, set_emulation, multithreading_mode, params_from_history, set_db, \
    db_context
from ann_automl.core import nn_recommend
import ann_automl.core.db_module as db

import pytest
import os
import shutil
import tempfile


# current file path
file_path = os.path.dirname(os.path.abspath(__file__))
# path to test datasets
test_datasets_path = os.path.join(file_path, 'datasets')


@pytest.fixture(scope='module')
def db_dir():
    db_dir = tempfile.mkdtemp()
    yield db_dir
    shutil.rmtree(db_dir)


def run_train(categories):
    task2 = NNTask('train', categories, target=0.95)
    hparams = recommend_hparams(task2, trace_solution=True)
    hparams['epochs'] = 5

    result, _ = train(task2, hparams=hparams)
    print("Training result:")
    print(f"  loss:     {result[0]:.4f}")
    print(f"  accuracy: {result[1]:.4f}")


@pytest.mark.usefixtures('db_dir')
def test_train(db_dir):
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
                                      'date_created': '2020-01-01'})
    with db.num_processes_context(1):
        with db_context(mydb):
            print('====================')
            print('Test binary classification')
            run_train(['dog', 'elephant'])

            print('====================')
            print('Test multiclass classification')
            run_train(['cat', 'dog', 'elephant'])


if __name__ == '__main__':
    db_dir = tempfile.mkdtemp()
    test_train(db_dir)
    shutil.rmtree(db_dir)
