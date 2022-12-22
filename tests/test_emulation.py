import os
import shutil
import tempfile

import pytest

from ann_automl.core.nn_recommend import recommend_hparams, NNTask
from ann_automl.core.nnfuncs import train, tune, set_emulation, multithreading_mode, params_from_history, db_context
import ann_automl.core.db_module as db


def test_recommend():
    task = NNTask('train', ['cat', 'dog', 'elephant'], target=0.95)
    hparams = recommend_hparams(task, trace_solution=True)
    print("Recommended hparams:")
    for k, v in hparams.items():
        print(f'{k}: {v}')


@pytest.fixture(scope='module')
def db_dir():
    file_path = os.path.dirname(os.path.abspath(__file__))
    test_datasets_path = os.path.join(file_path, 'datasets')
    db_dir = tempfile.mkdtemp()
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
            yield db_dir
    shutil.rmtree(db_dir)


@pytest.mark.usefixtures('db_dir')
def test_train_emulation(db_dir):
    task = NNTask('train', ['cat', 'dog', 'elephant'], target=0.95)
    hparams = recommend_hparams(task, trace_solution=True)
    print("Recommended hparams:")
    for k, v in hparams.items():
        print(f'{k}: {v}')

    set_emulation(True)
    result, _ = train(task, hparams=hparams)
    print("Training result:")
    print(f"  loss:     {result[0]:.4f}")
    print(f"  accuracy: {result[1]:.4f}")


@pytest.mark.usefixtures('db_dir')
def test_tune_emulation(db_dir):
    task = NNTask('train', ['dog', 'elephant'], target=0.95)
    hparams = recommend_hparams(task, trace_solution=True)
    print("Recommended hparams:")
    for k, v in hparams.items():
        print(f'{k}: {v}')

    set_emulation(True)
    tune(task, ['lr/batch_size', 'batch_size', 'optimizer', 'nesterov'], 'grid', hparams=hparams)


@pytest.mark.usefixtures('db_dir')
def run_multithreading(db_dir):
    import threading
    task = NNTask('train', ['dog', 'elephant'], target=0.95)
    with multithreading_mode():
        hparams = recommend_hparams(task, trace_solution=False)
        print("Recommended hparams:")
        for k, v in hparams.items():
            print(f'{k}: {v}')

        set_emulation(True)
        cparams = threading.Thread(target=params_from_history, args=[task])
        cparams.start()
        res = threading.Thread(target=train, args=[task, hparams])
        res.start()
        cparams.join()
        res.join()


if __name__ == '__main__':
    test_recommend()
    test_train_emulation()
    test_tune_emulation()
    # test_multithreading()
