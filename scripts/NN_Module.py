import keras
import os
import pandas as pd
from matplotlib import pyplot as plt

# os.chdir('C:\\')
keras.backend.set_image_data_format('channels_last')


class DataPrepare:
    def __init__(self, data_base_adress):
        self.data_load(data_base_adress)

    def data_load(self, data_catalog_name):
        os_symbol = "/"
        self.Data = {}  # dictionary with information about the database
        # full paths to  data folders
        self.Data['train_catalog_name'] = data_catalog_name + os_symbol + "train"  # full paths to  data folders
        self.Data['test_catalog_name'] = data_catalog_name + os_symbol + "test"
        self.Data['validation_catalog_name'] = data_catalog_name + os_symbol + "validation"

        info = pd.read_csv(data_catalog_name + os_symbol + 'Info.csv', sep=',')  # loading database information
        self.Data['classes_list'] = list(info.classes)
        self.Data['Image_resol'] = (info.av_resol_height[0], info.av_resol_width[0])


class Preprocessing(DataPrepare):
    def __init__(self, data_catalog_name):
        super().__init__(data_catalog_name)
        self.data_augmentation()

    def data_augmentation(self):
        self.aug_param_names = {}  # augmentation parameters dictionary
        self.aug_param_names['rescale'] = None  # 1./255
        self.aug_param_names['preprocessing_function'] = keras.applications.resnet.preprocess_input


class NN_init(Preprocessing):
    def __init__(self, data_catalog_name, file_name_of_h5_type):
        super().__init__(data_catalog_name)
        self.NN_load(file_name_of_h5_type)  # loading pretrained neural network
        self.NN_upgrade()  # neural network architecture upgrade
        self.train_parameters()  # setting training parameters

    def train_parameters(self):

        self.parameters = {}
        self.parameters['optimizer'] = 'Adam'
        self.parameters['optimizer_parameters'] = {}
        self.parameters['optimizer_parameters']['lr'] = 0.001
        if len(self.Data['classes_list']) == 2:
            func = 'binary_crossentropy'
        else:
            func = '"categorical_crossentropy"'
        self.parameters['loss'] = func
        self.parameters['metrics'] = 'accuracy'
        self.parameters['batch_size'] = 32
        self.parameters['epoch'] = 15

    def NN_upgrade(self):

        # creating a model folder with information about it
        model_name = 'New_Network'
        self.model_path = "./Neural Networks" + '/' + model_name
        self.model_name = self.model_path + '/' + model_name
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # model creating
        x = keras.layers.Input(shape=(self.Data['Image_resol'][0], self.Data['Image_resol'][1], 3))
        y1 = self.core_architecture(x)
        clas_type = len(self.Data['classes_list'])
        if clas_type == 2:
            y1 = keras.layers.Dense(1)(y1)
            func = 'sigmoid'
        else:
            y1 = keras.layers.Dense(clas_type)(y1)
            func = 'softmax'
        y = keras.layers.Activation(func)(y1)

        # model saving
        self.model = keras.models.Model(inputs=x, outputs=y)
        self.model.summary()
        self.model.save(self.model_name + '.h5')
        keras.utils.plot_model(self.model, to_file=(self.model_name + '.png'), rankdir='TB', show_shapes=True)

    def NN_load(self, file_name_of_h5_type):
        self.core_architecture = keras.models.load_model(file_name_of_h5_type)


class NN_training(NN_init):
    def __init__(self, data_catalog_name, file_name_of_h5_type):
        super().__init__(data_catalog_name, file_name_of_h5_type)
        # self.training() # neural network training
        # self.testing() # neural network testing and saving result

    def training(self):
        print('\n' + 'Training' + '\n')
        if len(self.Data['classes_list']) == 2:
            mode = 'binary'
        else:
            mode = 'categorical'

        # creating generators
        DataGenerator = keras.preprocessing.image.ImageDataGenerator(rescale=self.aug_param_names['rescale'],
                                                                     preprocessing_function=self.aug_param_names[
                                                                         'preprocessing_function'])

        Train_generator = DataGenerator.flow_from_directory(directory=self.Data['train_catalog_name'],
                                                            target_size=self.Data['Image_resol'], class_mode=mode,
                                                            batch_size=self.parameters['batch_size'])
        Validation_generator = DataGenerator.flow_from_directory(directory=self.Data['validation_catalog_name'],
                                                                 target_size=self.Data['Image_resol'], class_mode=mode,
                                                                 batch_size=self.parameters['batch_size'])

        # preparation for training
        self.model.compile(optimizer=self.parameters['optimizer'], loss=self.parameters['loss'],
                           metrics=[self.parameters['metrics']])

        C_Log = keras.callbacks.CSVLogger(self.model_name + '.csv')
        C_Ch = keras.callbacks.ModelCheckpoint(self.model_path + '/weights' + '-{epoch:02d}.h5', monitor='val_accuracy',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        # training
        self.model.fit_generator(generator=Train_generator,
                                 steps_per_epoch=(len(Train_generator.filenames) // self.parameters['batch_size']),
                                 epochs=self.parameters['epoch'], validation_data=Validation_generator,
                                 validation_steps=(
                                             len(Validation_generator.filenames) // self.parameters['batch_size']),
                                 callbacks=[C_Log, C_Ch])

        self.training_process_plot()

    def testing(self):
        print('\n' + 'Testing' + '\n')
        if len(self.Data['classes_list']) == 2:
            mode = 'binary'
        else:
            mode = 'categorical'
        DataGenerator = keras.preprocessing.image.ImageDataGenerator(rescale=self.aug_param_names['rescale'],
                                                                     preprocessing_function=self.aug_param_names[
                                                                         'preprocessing_function'])
        Test_generator = DataGenerator.flow_from_directory(self.Data['test_catalog_name'],
                                                           target_size=self.Data['Image_resol'],
                                                           class_mode=mode, batch_size=self.parameters['batch_size'])

        self.model.compile(optimizer=self.parameters['optimizer'], loss=self.parameters['loss'],
                           metrics=[self.parameters['metrics']])
        scores = self.model.evaluate_generator(Test_generator, steps=None, verbose=1)

        # saving testing results
        result = pd.DataFrame({'loss': [scores[0]], 'accuracy': [scores[1]]})
        result.to_csv(self.model_name + ' Result_scores.csv')

    def training_process_plot(self, show=True):

        history = pd.read_csv(self.model_name + '.csv')
        fig = plt.figure(figsize=(7, 7))

        plt.plot(history['epoch'].values, history['loss'].values, label='Loss')
        plt.plot(history['epoch'].values, history['val_loss'].values, label='Validation Loss')
        plt.legend(fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Loss function value', fontsize=15)
        if show:
            plt.show()
        fig.savefig(self.model_name + ' Loss.png')

        fig = plt.figure(figsize=(7, 7))

        plt.plot(history['epoch'].values, history[self.parameters['metrics']].values, label='Accuracy')
        plt.plot(history['epoch'].values, history['val_' + self.parameters['metrics']].values,
                 label='Validation Accuracy')
        plt.legend(fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Accuracy value', fontsize=15)
        if show:
            plt.show()
        fig.savefig(self.model_name + ' Accuracy.png')


def main():
    data_catalog_name = './Databases/Kaggle_CatsVSDogs'
    file_name_of_h5_type = './Architectures/ResNet50.h5'
    P = NN_training(data_catalog_name, file_name_of_h5_type)


if __name__ == '__main__':
    main()
