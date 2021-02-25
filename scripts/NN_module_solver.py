from typing import Dict, Any

from solver.base import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
import pandas as pd
from matplotlib import pyplot as plt

keras.backend.set_image_data_format('channels_last')


class NNState:
    def __init__(self):
        self.data = None
        self.model = None
        self.test_result = None
        self.core_architecture = None
        self.model_name = None
        self.model_path = None
        self.aug_param_names = None
        self.parameters = None
        self.trained = False
        self.plotted = False
        self.trained_plotted = False


class DataPrepare(Task):
    def __init__(self, data_base_address, goals=()):
        super().__init__(list(goals) + ['data_load'])
        self.data_base_address = data_base_address

    def prepare_state(self, nn: NNState):
        nn.data = None


@rule(DataPrepare)
class LoadData(Rule):
    def __init__(self):
        pass

    def can_apply(self, task: DataPrepare, nn: NNState):
        return 'data_load' in task.goals and nn.data is None

    def apply(self, task: DataPrepare, nn: NNState):
        os_symbol = "/"
        # dictionary with information about the database
        # full paths to  data folders
        nn.data = {'train_catalog_name':      task.data_base_address + os_symbol + "train",
                   'test_catalog_name':       task.data_base_address + os_symbol + "test",
                   'validation_catalog_name': task.data_base_address + os_symbol + "validation"}

        info = pd.read_csv(task.data_base_address + os_symbol + 'Info.csv', sep=',')  # loading database information
        nn.data['classes_list'] = list(info.classes)
        nn.data['Image_resol'] = (info.av_resol_height[0], info.av_resol_width[0])
        task.answer = nn.data


class Preprocessing(Task):
    def __init__(self, data_catalog_name, goals=()):
        super().__init__(list(goals) + ['augment'])
        self.data_catalog_name = data_catalog_name

    def prepare_state(self, nn: NNState):
        nn.aug_param_names = None


@rule(Preprocessing)
class DefaultAugmentation(Rule):
    def __init__(self):
        pass

    def can_apply(self, task: Preprocessing, nn: NNState):
        return 'augment' in task.goals and nn.aug_param_names is None

    def apply(self, task: Preprocessing, nn: NNState):
        nn.aug_param_names = {}  # augmentation parameters dictionary
        nn.aug_param_names['rescale'] = None  # 1./255
        nn.aug_param_names['preprocessing_function'] = keras.applications.resnet.preprocess_input
        task.solved = True


class NNInit(Task):
    def __init__(self, file_name_of_h5_type, goals=()):
        super().__init__(list(goals) + ['plot_model'])
        self.file_name_of_h5_type = file_name_of_h5_type

    def prepare_state(self, nn: NNState):
        nn.core_architecture = None
        nn.parameters = None
        nn.model_name = None
        nn.model_path = None
        nn.model = None


@rule(NNInit)
class LoadPretrained(RuleFL):
    def filter(self, task: NNInit, nn: NNState):
        defined(task.file_name_of_h5_type)
        ensure(nn.core_architecture is None)

    def apply(self, task: NNInit, nn: NNState):  # (self, file_name_of_h5_type):
        nn.core_architecture = keras.models.load_model(task.file_name_of_h5_type)


@rule(NNInit)
class NNDefaultTrainParams(Rule):
    def can_apply(self, task: NNInit, nn: NNState):
        return nn.parameters is None

    def apply(self, task: NNInit, nn: NNState):
        nn.parameters = {}
        nn.parameters['optimizer'] = 'Adam'
        nn.parameters['optimizer_parameters'] = {}
        nn.parameters['optimizer_parameters']['lr'] = 0.001
        if len(nn.data['classes_list']) == 2:
            func = 'binary_crossentropy'
        else:
            func = '"categorical_crossentropy"'
        nn.parameters['loss'] = func
        nn.parameters['metrics'] = 'accuracy'
        nn.parameters['batch_size'] = 32
        nn.parameters['epoch'] = 1


@rule(NNInit)
class NNPrepareDirectories(Rule):
    def can_apply(self, task: NNInit, state: NNState):
        return state.model_path is None

    def apply(self, task: NNInit, state: NNState):
        # creating a model folder with information about it
        model_name = 'New_Network'
        state.model_path = "./TrainedNN" + '/' + model_name
        state.model_name = state.model_path + '/' + model_name
        if not os.path.exists(state.model_path):
            os.makedirs(state.model_path)


@rule(NNInit)
class NNUpgradeModel(RuleFL):
    def filter(self, task: NNInit, nn: NNState):
        ensure(nn.model is None)
        defined(nn.parameters)
        defined(nn.model_path)
        defined(nn.core_architecture)
        defined(nn.data)

    def apply(self, task: NNInit, nn: NNState):
        # model creating
        x = keras.layers.Input(shape=(nn.data['Image_resol'][0], nn.data['Image_resol'][1], 3))
        y1 = nn.core_architecture(x)
        clas_type = len(nn.data['classes_list'])
        if clas_type == 2:
            y1 = keras.layers.Dense(1)(y1)
            func = 'sigmoid'
        else:
            y1 = keras.layers.Dense(clas_type)(y1)
            func = 'softmax'
        y = keras.layers.Activation(func)(y1)

        # model saving
        nn.model = keras.models.Model(inputs=x, outputs=y)
        nn.model.summary()
        nn.model.save(nn.model_name + '.h5')
        nn.plotted = False


@rule(NNInit)
class NNPlotModel(RuleFL):
    def filter(self, task: NNInit, nn: NNState):
        defined(nn.model)
        ensure('plot_model' in task.goals and not nn.plotted)

    def apply(self, task: NNInit, nn: NNState):
        keras.utils.plot_model(nn.model, to_file=(nn.model_name + '.png'), rankdir='TB', show_shapes=True)
        nn.plotted = True


@rule(NNInit)
class FinishNNInit(FinishTask):
    def filter(self, task: NNInit, nn: NNState):
        if 'plot_model' in task.goals:
            ensure(nn.plotted)
        defined(nn.model)


class NNTraining(Task):
    def __init__(self, goals=()):  # data_catalog_name, file_name_of_h5_type):
        super().__init__(goals)

    def prepare_state(self, nn: NNState):
        nn.trained = False
        nn.plot_saved = False


@rule(NNTraining)
class TrainNN(Rule):
    def can_apply(self, task: NNTraining, nn: NNState):
        return not nn.trained and 'nn_train' in task.goals

    def apply(self, task: NNTraining, nn: NNState):
        print('\n' + 'Training' + '\n')
        if len(nn.data['classes_list']) == 2:
            mode = 'binary'
        else:
            mode = 'categorical'

        # creating generators
        data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=nn.aug_param_names['rescale'],
                                                                      preprocessing_function=nn.aug_param_names[
                                                                          'preprocessing_function'])

        train_generator = data_generator.flow_from_directory(directory=nn.data['train_catalog_name'],
                                                             target_size=nn.data['Image_resol'], class_mode=mode,
                                                             batch_size=nn.parameters['batch_size'])
        validation_generator = data_generator.flow_from_directory(directory=nn.data['validation_catalog_name'],
                                                                  target_size=nn.data['Image_resol'], class_mode=mode,
                                                                  batch_size=nn.parameters['batch_size'])

        # preparation for training
        nn.model.compile(optimizer=nn.parameters['optimizer'], loss=nn.parameters['loss'],
                         metrics=[nn.parameters['metrics']])

        C_Log = keras.callbacks.CSVLogger(nn.model_name + '.csv')
        C_Ch = keras.callbacks.ModelCheckpoint(nn.model_path + '/weights' + '-{epoch:02d}.h5', monitor='val_accuracy',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        # training
        nn.model.fit(train_generator,
                     steps_per_epoch=(len(train_generator.filenames) // nn.parameters['batch_size']),
                     epochs=nn.parameters['epoch'], validation_data=validation_generator,
                     validation_steps=(len(validation_generator.filenames) // nn.parameters['batch_size']),
                     callbacks=[C_Log, C_Ch])
        nn.trained = True
        nn.trained_plotted = False


@rule(NNTraining)
class DrawProcessPlot(Rule):
    def can_apply(self, task: NNTraining, nn: NNState):
        return nn.trained and 'training_plot' in task.goals and not nn.trained_plotted

    # TODO: Продумать возможность задавать параметр show извне
    def apply(self, task: Task, nn: NNState):  # training_process_plot(self, show=True):
        history = pd.read_csv(nn.model_name + '.csv')
        fig = plt.figure(figsize=(7, 7))

        plt.plot(history['epoch'].values, history['loss'].values, label='Loss')
        plt.plot(history['epoch'].values, history['val_loss'].values, label='Validation Loss')
        plt.legend(fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Loss function value', fontsize=15)
        # if show:
        #    plt.show()
        fig.savefig(nn.model_name + ' Loss.png')

        fig = plt.figure(figsize=(7, 7))

        plt.plot(history['epoch'].values, history[nn.parameters['metrics']].values, label='Accuracy')
        plt.plot(history['epoch'].values, history['val_' + nn.parameters['metrics']].values,
                 label='Validation Accuracy')
        plt.legend(fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Accuracy value', fontsize=15)
        # if show:
        #    plt.show()
        fig.savefig(nn.model_name + ' Accuracy.png')
        nn.trained_plotted = True


@rule(NNTraining)
class TestTrainedNN(RuleFL):
    def filter(self, task: NNTraining, nn: NNState):
        ensure('nn_test' in task.goals)
        defined(nn.model)
        if 'nn_train' in task.goals:
            defined(nn.trained)

    def apply(self, task: NNTraining, nn: NNState):
        print('\n' + 'Testing' + '\n')
        if len(nn.data['classes_list']) == 2:
            mode = 'binary'
        else:
            mode = 'categorical'
        data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=nn.aug_param_names['rescale'],
                                                                      preprocessing_function=nn.aug_param_names[
                                                                          'preprocessing_function'])
        test_generator = data_generator.flow_from_directory(nn.data['test_catalog_name'],
                                                            target_size=nn.data['Image_resol'],
                                                            class_mode=mode, batch_size=nn.parameters['batch_size'])

        nn.model.compile(optimizer=nn.parameters['optimizer'], loss=nn.parameters['loss'],
                         metrics=[nn.parameters['metrics']])
        scores = nn.model.evaluate(test_generator, steps=None, verbose=1)

        # saving testing results
        nn.test_result = pd.DataFrame({'loss': [scores[0]], 'accuracy': [scores[1]]})
        nn.test_result.to_csv(nn.model_name + ' Result_scores.csv')


@rule(NNTraining)
class FinishTrain(FinishTask):
    def filter(self, task, nn: NNState) -> None:
        if 'nn_train' in task.goals:
            defined(nn.trained)
        if 'training_plot' in task.goals:
            ensure(nn.trained_plotted)
        if 'nn_test' in task.goals:
            defined(nn.test_result)


class NNTask(Task):
    """ Общая задача создания и обучения нейронной сети """

    def __init__(self, data_catalog_name, file_name_of_h5_type, goals=()):
        super().__init__(goals)
        self.data_catalog_name = data_catalog_name
        self.file_name_of_h5_type = file_name_of_h5_type

    def prepare_state(self, state):
        return state if state is not None else NNState()


@rule(NNTask)
class NNProcess(Rule):
    def can_apply(self, task: NNTask, state: SolverState):
        return not task.solved

    def apply(self, task: NNTask, nn: NNState):
        task.run_subtask(DataPrepare(task.data_catalog_name), nn)
        task.run_subtask(Preprocessing(task.data_catalog_name, goals=task.goals), nn)
        task.run_subtask(NNInit(task.file_name_of_h5_type, goals=task.goals), nn)
        task.run_subtask(NNTraining(goals=task.goals), nn)
        task.answer = nn


def main():
    data_catalog_name = './Databases/Kaggle_CatsVSDogs'
    file_name_of_h5_type = './Architectures/ResNet50.h5'
    # nn_task = NNTask(data_catalog_name, file_name_of_h5_type, goals=['nn_test'])
    nn_task = NNTask(data_catalog_name, file_name_of_h5_type, goals=['training_plot', 'nn_train', 'nn_test'])
    p = nn_task.solve(solver_state=SolverState(global_params={'trace_solution': True}))


if __name__ == '__main__':
    main()
