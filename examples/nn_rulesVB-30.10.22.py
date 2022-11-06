import itertools
import math
import sys
import time
from collections import defaultdict

import ipywidgets
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import os
from . import db_module
from datetime import datetime
from pytz import timezone
from .solver import Rule, rule, State, Task, printlog
from ..utils.process import request, NoHandlerError

myDB = db_module.dbModule(dbstring='sqlite:///tests.sqlite')  # TODO: уточнить путь к файлу базы данных

#базовый набор гиперпараметров
Default_params = {'pipeline' : 'ResNet50', 
              'modelLastLayers' : [{'Type': 'Dense', 'units': 64, 'activation': 'relu'},{'Type': 'Dense', 'units': 16, 'activation': 'relu'}],
              'augmenParams' : {'horizontal_flip': True, 'vertical_flip': None, 'width_shift_range': 0.4, 'height_shift_range': 0.4},
              'epochs' : 150, 'stoppingCriterion': 'stop_val_metr' , 'optimizer':'Adam', 'batchSize': 16, 'lr': 0.001  }


_data_dir = 'data'


def set_data_dir(data_dir):   #TODO Положить в соответствии с этой схемой данные
    """
    Set the data directory. Data directory contains the following subdirectories:  
    - architecures: contains the neural network architectures
    - datasets: contains the datasets
    - trainedNN: contains the trained neural networks
    - history: contains the training history of the neural networks
    """
    global _data_dir
    _data_dir = data_dir



class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
        if epoch == 0:
            self.start_of_train = self.epoch_time_start

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs={}):
        self.total_time = (time.time() - self.start_of_train)


# функция осмотра окрестности
def neighborhood(cp, rr):
    """
    Finds next point in the neighborhood of the current point where there is zero in rr

    Parameters
    ----------
    cp: list[int]
        Current point
    rr: ndarray
        Array of function values

    Returns
    -------
    tuple
        (neighbor, point) where neighbor=0 if neighbor is found, -1 if neighbor is not found;
        point is the next point in the neighborhood
    """
    st = (-1, 0, 1)
    neighbor = -1
    cur = []

    printlog('\n')

    for i in range(3):
        # print('i ', i)
        if (cp[0] + st[i]) < 0 or (cp[0] + st[i]) >= rr.shape[0]:
            # print('cond  i')
            continue
        for j in range(3):
            # print('j ', j)
            if (cp[1] + st[j]) < 0 or (cp[1] + st[j]) >= rr.shape[1]:
                # print('cond j')
                continue

            for k in range(3):
                # print('k ', k)
                # print (cp, rr[cp[0]+st[i], cp[1]+st[j], cp[2]+st[k] ])
                # print('grid value',  abs(rr[cp[0]+st[i], cp[1]+st[j], cp[2]+st[k] ]))
                if 0 <= (cp[2] + st[k]) < rr.shape[2] and abs(rr[cp[0] + st[i], cp[1] + st[j], cp[2] + st[k]]) < 1e-6:
                    cur = [cp[0] + st[i], cp[1] + st[j], cp[2] + st[k]]
                    # print('cond k',  cur)
                    neighbor = 0
                    break
            if neighbor == 0:
                break
        if neighbor == 0:
            break

    return neighbor, cur

'''
def find_zero_neighbor(center, table, radius=1):
    """
    Finds next point in the neighborhood of the current point where there is zero in table

    Parameters
    ----------
    center: list[int]
        Current point
    table: ndarray
        Array of function values
    radius: int
        Radius of the neighborhood

    Returns
    -------
    optional list[int]
        Next point in the neighborhood where there is zero in table or None if there is no such point
    """
    ranges = [range(max(0, center[i] - radius), min(szi, center[i] + radius + 1)) for i, szi in enumerate(table.shape)]
    for i in itertools.product(*ranges):
        if abs(table[i]) <= 1e-6:
            return list(i)
    return None


# Допустимая погрешность по достигнутой точности при выборе общей стратегии обучения
# eps = 0.1
'''


@rule
class InitialCheck(Rule):
    """ Приём для проверки доступа к БД """

    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'Initial'

    def apply(self, state: State):
        
        available_categories = list(myDB.get_all_categories()['name'])
        for cat in state.task.objects:
            ch = cat in available_categories
            if not ch:
                '''
                TODO:
                Проверка: выбрано не менее 2 категорий
                Если недоступна БД:
                TODO request(...  см. смотри 2 слайд презентации. ~ обработка ошибки
                "изменить' = перейти на главный экран
                "закончить" = закончить работу программы  = curstate 'Done' 
                
                '''
                #tmp - without request
                state.message = 'The "{a1}" category is not in the list of available categories. There are some problems accessing the database. We can get such categories:\n\n{ll}\n\nDo you want to change the category of images or log out of the system? "Change parameters" - 1; "Finish" - 0.'.format(a1 = cat, ll = available_categories)
                
                state.actions = {'0': 'Done', '1': 'Initial'} # Нужно вернуться к главному экрану
            
                printlog('\n' + state.message)
                answer = input()
                state.curState = state.actions[str(answer)]

                with open(state.logName, 'a') as log_file:
                    print(state.message + ' .Answer: ' + answer, file=log_file)
                return
                #tmp - without request
                    
        if ch:
            state.curState = 'FirstCheck'


@rule
class CheckSuitableModelExistence(Rule):
    """ Приём для поиска готовой подходящей модели в базе данных """

    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'FirstCheck'

    def apply(self, state: State):

        ch_res = myDB.get_models_by_filter({
            'min_metrics': state.task.goal,
            'task_type': state.task.taskType,
            'categories': list(state.task.objects)})

        s = []
        if ch_res.empty:
            state.curState = 'HyperParams'
        else:
            n = len(ch_res.index)
            for i in range(n):
                s.append({'model_address': ch_res.iloc[i]['model_address'], 'metric': 
                          list(state.task.goal.keys())[0], 'value': ch_res.iloc[i]['metric_value']})
                
            ''' TODO request см. презентацию, слайд №3.
                decision = request(...
                #new = "Обучить новую модель"
                #finish = "Закончить работу"
                if decision == 'new':
                    state.curState = 'HyperParams'
                elif decision == 'finish':
                    state.curState = 'Done'                                  
                    # see below tmp block ~ см if state.curState == 'Done':
                else:
                    printlog('Неверный ввод', file = sys.stderr)
                    state.curState = 'Done'
                    
            '''
            #tmp - without request
            state.message = 'There is a suitable model in our base of trained models. Would you like to train an another model? "No" - 0; "Yes" - 1. '
            state.actions = {'0': 'Done', '1': 'HyperParams'}
            
            printlog('\n' + state.message)
            answer = '1' #input()
            state.curState = state.actions[str(answer)]

            with open(state.logName, 'a') as log_file:
                print(state.message + ' .Answer: ' + answer, file=log_file)

            if state.curState == 'Done':
                with open(state.resName, 'a') as res_file:
                    print('Our system has trained models that match the required target.\n', file=res_file)
                    for i in range(n):
                        print('Model adress: ' + s[i]['model_address'] + '.\nGoal function: ' + s[i]['metric'] + '.\nTest value of goal function: ' + str(s[i]['value']) + 
                              '.\n', file=res_file)
            #tmp - without request

   
            
@rule
class SettingInitialHyperParameters(Rule):
    """ Приём, в котором сначала должен быть выбран подход к обучению, а затем полностью определен набор гиперпараметров для первого обучения + создание директории и 'истории' эксперимента"""
    
    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'HyperParams'

    def apply(self, state: State):
        '''
        
        TODO request(... как на слайд № 4 в презентации. Возвращает один из 5 вариантов: 'grid_search', 'strategy', 'manual', 'home', 'finish'
        'home' = сбросить всё и вернуться на начальный экран. Разумно делать предупреждение о сбросе
        'finish' = перейти в 'Done'
        
        Или 3 отдельные переменные: одна для подхода (один из трех вариантов), другие - флажки (вариант реализации)
        '''
        
        # Выбор подхода к обучению и заполнение набора гиперпараметров
        state.task.learningApproach = 'manual'
        if state.task.learningApproach == 'home':
            state.curState = 'Initial'
            return
        elif state.task.learningApproach == 'finish':
            state.curState = 'Done'
            return
        elif state.task.learningApproach == 'grid_search':
            state.task.HyperParams, home = grid()
        elif state.task.learningApproach == 'strategy':
            pass
            #functions for strategy
        elif state.task.learningApproach == 'manual':
            state.task.HyperParams, home = manual()
        else:
            printlog('Некорректно.')  #TODO  ~ обработка и в других местах аналогично
            return
        
        if home:
            state.curState = 'Initial'
            return
        
        if len(state.task.objects) > 2:
            state.task.HyperParams['loss'] = "categorical_crossentropy"
        else:
            state.task.HyperParams['loss'] = "binary_crossentropy"
            
        state.task.HyperParams['metrics'] = 'accuracy'


        #Создание директории эксперимента 
        tt=time.localtime(time.time())
        MSK = timezone('Europe/Moscow')
        msk_time = datetime.now(MSK)
        tt = msk_time.strftime('%Y_%m_%d_%H_%M_%S')
        
        obj_codes = myDB.get_cat_IDs_by_names(list(state.task.objects))
        obj_codes.sort()
        obj_str = '_'.join(map(str, obj_codes))    
        
        state.task.expName = state.task.taskType+'_' + obj_str + '_DT_' + tt
        state.task.expPath = "./trainedNN" + '/' + state.task.expName             #TODO Нужна фиксированная директория
        if not os.path.exists(state.task.expPath):
            os.makedirs(state.task.expPath)

        #Создание истории эксперимента - нулевая строка = условие задачи
        HST = {'taskType': state.task.taskType, 'objects': [state.task.objects], 'expName': state.task.expName, 'LearningApproach': None }
        
        for kk in state.task.HyperParams.keys():
            HST[kk] = None
           
        HST['data'] = None
        HST['Metric_achieved_result'] = state.task.goal[ str(list(state.task.goal.keys())[0])] # цель эксперимента
        HST['curTrainingSubfolder'] = None
        HST['timeStat'] = None
        HST['totalTime'] = None
        HST['Additional_params'] = None
        
 
        state.task.history = pd.DataFrame(HST)
        state.task.history.to_csv( state.task.expPath + '/' + state.task.expName + '__History.csv', index_label = 'Index')
            
        state.task.counter = 1
        
        state.curState = 'Model_Training'
        
def grid(state: State):
    '''
    TODO   request(...   см слайд №5 должны вернуться 
    1) словарь FixedParams с гиперпараметрами без диапазона
    2) словарь гиперепараметров с диапазоном (хотя бы 2) state.task.GridParams (т.е. используются в сетке) {'название гиперпараметра': [диапазон],...}
    3) словарь dep = {hyperparameter1 : hyperparameter2} Поскольку некоторые гиперпараметры м.б. взаимосвязаны (это должно учитываться на экране выбора гиперпараметров), то подобные зависимости д.б. здесь указаны 
    4) home - флажок, вернуться на главный экран (T) или нет (схема работы аналогично manual)
    
    Также при должны быть проверки на соответствие гиперпараметров их возможным диапазонам (см. manual()) и взаимосвязи
    
    '''
    #tmp - without request
    home = False
    dep = {'optimizer': 'lr'}
    FixedParams = {'pipeline' : 'ResNet50', 
              'modelLastLayers' : [{'Type': 'Dense', 'units': 64, 'activation': 'relu'},{'Type': 'Dense', 'units': 16, 'activation': 'relu'}],
              'augmenParams' : {'horizontal_flip': True, 'vertical_flip': None, 'width_shift_range': 0.4, 'height_shift_range': 0.4},
              'epochs' : 150, 'stoppingCriterion': 'stop_val_metr' }
    
    state.task.GridParams = {'optimizer':['Adam', 'RMSprop', 'SGD'], 'batchSize':[8,16,32,64], 
                             'lr' : [[0.00025,0.0005,0.001,0.002,0.004], [0.00025,0.0005,0.001,0.002,0.004], [0.0025,0.005,0.01,0.02,0.04]]}
    
    #tmp - without request
    if home:
        return _, home
    if not bool(state.task.GridParams): # если нет гиперпараметров с диапазоном, т.е. все гиперпараметры определены
        return FixedParams, False
        
    params = FixedParams
    
    #create hyperparameter grid
    
    return params, home
    
    
    
def manual(): 
    
    ''' 
    TODO request(...)  Слайд №7 презентации.
    На экране д.б. представлен набор гиперпараметров с соответствующими допустимыми значениями и ответ на вопрос "Хотим ли мы перейти на главный экран или нет" ~ переменная home
    Минимальный набор гиперпараметров
    paramsSet = { 'pipeline' : ['ResNet18', 'ResNet34', 'ResNet50'],
                 'modelLastLayers':
                 Для modelLastLayers должна быть возможность указать порядок подсоединения к pipeline
                 Есть четыре типа слоев Dense, MaxPooling2D, Conv2D, AveragePooling2D
                 У Dense параметры следующие:  'units': натуральное число, 'activation': 'relu', 'elu', 'selu', 'leaky_relu', 'gelu';
                 У Conv2D параметры следующие: 'filters': натуральное число, 'kernel_size' - натуральное число (будем считать, что 1-11), 'strides': -  None                  или натуральное число (будем считать, что 1-4), 'activation': 'relu', 'elu', 'selu', 'leaky_relu', 'gelu';
                 У MaxPooling2D и AveragePooling2D параметры следующие: pool_size - натуральное число (будем считать, что 1-7), strides - None или                            натуральное число (будем считать, что 1-4)  

                 'augmenParams': {'horizontal_flip': True, False, 'vertical_flip': True, False, None,
                 rotation_range 0 - 1.0, 'zoom_range': 0 - 1
                 width_shift_range': 0.0 - 1.0,
                 height_shift_range':  0.0 - 1.0
                 brightness_range 0.0 - 1.0

                 'epochs' - натуральное число (сколько учится сеть, наверное, имеет смысл поставить лимит, чтобы система не рухнула)
                 'stoppingCriterion': ['stop_val_metr', 'epochs']
                 'optimizer': ['Adam', 'RMSprop', 'SGD']
                 'batchSize': натуральное число (при 128 сервер иногда падал. М.б. разумно поставить лимит)
                  learning rate: При выборе оптимизатора, д.б. доступен именно его диапазон (!) 
                  Для Adam & RMSProp: [0.00025 - 0.004],  # разумный диапазон
                  Для SGD: [0.0025, - 0.04]}   # разумный диапазон


      Мне должен прийти словарь (полностью заполненный !!!), как в данном примере (в нем указаны значения по умолчанию) + естественно проверка, что пришло всё корректно:

      '''
    params = {'pipeline' : 'ResNet50', 'modelLastLayers' : [[{'Type': 'Dense', 'units': 64, 'activation': 'relu'},{'Type': 'Dense', 'units': 16, 'activation': 'relu'}]], 'augmenParams' : [{'horizontal_flip': True, 'vertical_flip': None, 'width_shift_range': 0.4, 'height_shift_range': 0.4}],
              'epochs' : 150, 'stoppingCriterion': 'stop_val_metr' , 'optimizer':'Adam', 'batchSize': 16, 'lr': 0.001  }
    home = False
    
    return params, home



@rule
class Training(Rule):
    """ Приём проверки возможности работы с базой данных """
    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'Model_Training'

    def apply(self, state: State):
        
        
        CreateDatabase(state)
        DataGenerator(state)
        ModelAssembling(state)
        FitModel(state)

####################################################



def CreateDatabase(state: State): 
    ''' Из Базы данных заргружаются списки изображений для обучающего, проверочного и тестового множеств'''
    with open(state.logName, 'a') as log_file:
        print('CreateDatabase', file = log_file)

    printlog(f'load annotations for {list(state.task.objects)}')
    tmpData = myDB.load_specific_categories_annotations(list(state.task.objects), normalize_cats=True,
                                                        splitPoints=[0.7, 0.85],
                                                        cur_experiment_dir='./', crop_bbox=False,
                                                        cropped_dir='./crops/')

    state.task.data = {'train': tmpData[1]['train'],
                       'validate': tmpData[1]['validate'],
                       'test': tmpData[1]['test'],
                       'dim': (224, 224, 3),
                       'augmenConstr': {'vertical_flip': None}}
    

def DataGenerator(state: State):
    ''' Создаются генераторы изображений + настройка аугментации данных'''
    
    printlog('DataGenerator')
    
    with open(state.logName, 'a') as log_file:
        print('DataGenerator', file = log_file)

    df_train = pd.read_csv(state.task.data['train'])
    df_validate = pd.read_csv(state.task.data['validate'])
    df_test = pd.read_csv(state.task.data['test'])
    
    HP = state.task.HyperParams['augmenParams'][0]
    HP['preprocessing_function'] = 'keras.applications.resnet.preprocess_input'

    dataGen = keras.preprocessing.image.ImageDataGenerator(HP)


    train_generator = dataGen.flow_from_dataframe(df_train, x_col=list(df_train.columns)[0],
                                                  y_col=list(df_train.columns)[1],
                                                  target_size=state.task.data['dim'][0:2],
                                                  class_mode='raw',
                                                  batch_size=state.task.HyperParams['batchSize'])
    validate_generator = dataGen.flow_from_dataframe(df_validate, x_col=list(df_validate.columns)[0],
                                                     y_col=list(df_validate.columns)[1],
                                                     target_size=state.task.data['dim'][0:2],
                                                     class_mode='raw',
                                                     batch_size=state.task.HyperParams['batchSize'])
    test_generator = dataGen.flow_from_dataframe(df_test, x_col=list(df_test.columns)[0],
                                                 y_col=list(df_test.columns)[1],
                                                 target_size=state.task.data['dim'][0:2],
                                                 class_mode='raw',
                                                 batch_size=state.task.HyperParams['batchSize'])

    state.task.generators = [train_generator, validate_generator, test_generator]


def ModelAssembling(state: State):
    ''' Собрается модель + создается её инфраструктура'''
    
    with open(state.logName, 'a') as log_file:
        print('ModelAssembling', file=log_file)

    printlog(f"Base architecture: {state.task.HyperParams['pipeline']}.h5")
    x = keras.layers.Input(shape=(state.task.data['dim']))
    y=keras.models.load_model('./architectures/'+state.task.HyperParams['pipeline']+'.h5')(x)
    
    for layer in state.task.HyperParams['modelLastLayers'][0]:
        layerParams = list(layer.keys())
        layerParams.remove('Type')
        par_str = '('
        for p in layerParams:
            if p == 'activation':
                par_str += p + ' = \'' + str(layer[p]) + '\','
            else:
                par_str += p + ' = ' + str(layer[p]) + ','
        par_str = par_str[:-1] +')(y)'
           
        y = eval('keras.layers.' + layer['Type'] + par_str)

    if len(state.task.objects) > 2:
        y=keras.layers.Dense(units = len(state.task.objects))(y)
        y = keras.layers.Activation(activation = 'softmax')(y)
    else:
        y=keras.layers.Dense(units = 1)(y)
        y = keras.layers.Activation(activation = 'sigmoid')(y)
    state.task.model = keras.models.Model(inputs=x, outputs=y)
    
   
    # Директория для обучаемого экземпляра
    state.task.curTrainingSubfolder = state.task.expName + '_' + str(state.task.counter)
    model_path = state.task.expPath + '/' + state.task.curTrainingSubfolder
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    prefix = model_path + '/' + state.task.curTrainingSubfolder
    state.task.model.save(prefix + '__Model.h5')
    tf.keras.utils.plot_model(state.task.model, to_file=prefix + '__ModelPlot.png', rankdir='TB', show_shapes=True)
    print(prefix + '__ModelPlot.png')


def FitModel(state: State):
    """
    TODO Для этой функции д.б. реализовано распараллеливание state.task.model необходимо обучить в соответствии с процедурой описанной ниже, взять среднее значение оценки обученной модели. И это есть scores. Результаты всех параллельных запусков записать в state.history['Additional_params']. Надо именно скопировать state.task.model, а не создавать новую, т.к. при создании сети параметры модели инициализируются случайно.
    """
    
    printlog('FitModel')
    
    state.task.model.compile(optimizer = eval('keras.optimizers.' + state.task.HyperParams['optimizer'] +'(learning_rate = ' + str( state.task.HyperParams['lr'] ) +')' ), loss = state.task.HyperParams['loss'], metrics=[state.task.HyperParams['metrics']])
        
    # Обучение нейронной сети

    with open(state.logName, 'a') as log_file:
        print(state.task.curTrainingSubfolder, file = log_file)
        print(state.task.HyperParams, file = log_file)
 

    model_path = f'{state.task.expPath}/{state.task.curTrainingSubfolder}'
    model_history = model_path + '/' + state.task.curTrainingSubfolder + '__History.csv'

    C_Log = keras.callbacks.CSVLogger(model_history)  # устанавливаем файл для логов
    C_Ch = keras.callbacks.ModelCheckpoint(
        model_path + '/' + state.task.curTrainingSubfolder + '_weights' + '-{epoch:02d}.h5',  # устанавливаем имя файла для сохранения весов
        monitor='val_' + state.task.HyperParams['metrics'],
        save_best_only = True, save_weights_only = False, mode = 'max', verbose = 1)
    C_ES = keras.callbacks.EarlyStopping(monitor = 'val_' + state.task.HyperParams['metrics'], min_delta = 0.001,
                                         mode = 'max', patience = 5)
    C_T = TimeHistory()
    
    '''
    #TODO экран 10 слайд в презентации, там д.б. текущие гиперпараметры обучения + вывод обучения модели = визуализация

    state.task.model.fit(x = state.task.generators[0],
                         steps_per_epoch = len(state.task.generators[0].filenames) // state.task.HyperParams['batchSize'],
                         epochs = state.task.HyperParams['epochs'] ,
                         validation_data = state.task.generators[1],
                         callbacks = [C_Log, C_Ch, C_ES, C_T],
                         validation_steps = len(state.task.generators[1].filenames) // state.task.HyperParams['batchSize'])
    '''

    scores = [0.5, 0.87 ] # state.task.model.evaluate(state.task.generators[2], steps=None, verbose=1)  # оценка обученной модели
    
    #state.task.hp_grid[tuple(state.task.cur_training_point)] = scores[1]   state.task.HyperParams['epochs']    №TODO MEMEME

    with open(state.logName, 'a') as log_file:
        print(scores, file=log_file)
        

    #сохранение результатов в историю эксперимента
    
    HST = {'Index' : state.task.counter, 'taskType': state.task.taskType, 'objects': [state.task.objects], 'expName': state.task.expName,  'LearningApproach': state.task.learningApproach }
    
    
    for kk in state.task.HyperParams.keys():
        HST[kk] = state.task.HyperParams[kk]

    HST['data'] = [state.task.data]
    HST['Metric_achieved_result'] = 1 #scores[1]
    HST['curTrainingSubfolder'] = state.task.curTrainingSubfolder
    HST['timeStat'] = None #[C_T.times]
    HST['totalTime'] = None#C_T.total_time
    HST['Additional_params'] = [{}]

    #print(HST)
    
    state.task.history = state.task.history.append((HST), ignore_index = True)

    state.task.history.to_csv( state.task.expPath + '/' + state.task.expName + '__History.csv', index = False)
    #print( state.task.expPath + '/' + state.task.expName + '__History.csv')

    #  
    if scores[1] > state.task.goal[str(list(state.task.goal.keys())[0])]:
        pass
        # request( См 8-ой слайд) Возвращает: 'Done' = закончить, 'ChangeHyperParams' = обучить
    else:
        state.curState = 'ChangeHyperParams'

              
@rule
class ChangeHyperParams(Rule):
    """Прием для изменения гиперпараметров обучения модели"""

    def can_apply(self, state: State):
        return state.task.taskCt == "train" and state.curState == 'ChangeHyperParams'

    def apply(self, state: State):


        with open(state.logName, 'a') as log_file:
            print('ChangeHyperParams', file=log_file)
            
        '''
        TODO request((  как на слайды № 9 в презентации.
        При выборе "поиск по сетке" вывести экран №5
        При выборе  "Вручную задать параметры обучения" - как на экране 7. Поля должны быть заполнены теми гиперпараметрами, что хранятся в HyperParams
        и при изменении пользователем каких-то гиперпараметров соответствующие поля д.б. выделены
        При выборе  "стратегия" - как на экране 6. Отметить текущую (если была) и уже отработанные стратегии.
        

        Возвращает один из 6 вариантов: 'grid_search', 'strategy', 'manual', 'home','finish', 'random'
        home = сбросить всё и вернуться на начальный экран. Разумно делать предупреждение о сбросе
        '''
        
        # Выбор подхода к обучению и заполнение набора гиперпараметров
        appr = ['grid_search', 'strategy', 'manual']
        home = False
        state.task.learningApproach = 'manual'
        if state.task.learningApproach == 'home':
            state.curState = 'Initial'
            return
        elif state.task.learningApproach == 'finish':
            state.curState = 'Done'
            return
        elif state.task.learningApproach == 'random':
            idx = np.random.randint(0, high=3)
            state.task.learningApproach = appr[idx]
        
        if state.task.learningApproach == 'grid_search':
            pass
            #functions for grid search
        elif state.task.learningApproach == 'strategy':
            pass
            #functions for strategy
        elif state.task.learningApproach == 'manual':
            state.task.HyperParams, home = manual()
        else:
            printlog('Некорректно.')  #TODO  ~ обработка
            return
        
        if home:
            state.curState = 'Initial'
            return
        
        if len(state.task.objects) > 2:
            state.task.HyperParams['loss'] = "categorical_crossentropy"
        else:
            state.task.HyperParams['loss'] = "binary_crossentropy"
            
        state.task.HyperParams['metrics'] = 'accuracy'
          
        state.task.counter += 1
        
        state.curState = 'Model_Training'    
        


        
