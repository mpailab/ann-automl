from ann_automl.core.nn_recommend import recommend_hparams, NNTask
from ann_automl.core.nnfuncs import set_data_dir, train
from ann_automl.core.solver import set_log_dir


def run_train():
    set_data_dir('data')
    set_log_dir('log')

    task = NNTask('train', ['cat', 'dog', 'elephant'], target=0.95)
    hparams = recommend_hparams(task, trace_solution=True)
    print("Recommended hparams:")
    for k, v in hparams.items():
        print(f'{k}: {v}')

    result = train(task, hparams=hparams)
    print("Training result:")
    print(f"  loss:     {result[0]:.4f}")
    print(f"  accuracy: {result[1]:.4f}")


if __name__ == '__main__':
    run_train()
