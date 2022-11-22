from ann_automl.core.nn_solver import recommend_hparams, NNTask
from ann_automl.core.nnfuncs import train, tune, set_emulation, multithreading_mode, params_from_history
from ann_automl.core import nn_recommend


def test_recommend():
    task = NNTask('train', ['cat', 'dog', 'elephant'], target=0.95)
    hparams = recommend_hparams(task, trace_solution=True)
    print("Recommended hparams:")
    for k, v in hparams.items():
        print(f'{k}: {v}')


def test_train_emulation():
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


def test_tune_emulation():
    task = NNTask('train', ['dog', 'elephant'], target=0.95)
    hparams = recommend_hparams(task, trace_solution=True)
    print("Recommended hparams:")
    for k, v in hparams.items():
        print(f'{k}: {v}')

    set_emulation(True)
    tune(task, ['learning_rate', 'batch_size', 'optimizer', 'nesterov'], 'grid', hparams=hparams)


def test_multithreading():
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
    test_multithreading()
