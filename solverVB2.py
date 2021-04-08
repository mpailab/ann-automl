# -*- coding: utf-8 -*-

import os
from google.colab import drive
drive.mount('/content/drive')
os.chdir('./drive/My Drive')
!ls


from abc import ABC, abstractmethod
import time
import json
import pandas as pd
import keras

# Глобальный параметр rules задаёт список приёмов
rules = []
# Допустимая погрешность по достигнутой точности при выборе общей стратегии обучения 
eps=0.1

def rule(cls):
    global rules
    rules.append(cls())
    return cls


class Task:
    '''
    Решаемая задача.
    '''

    def __init__(self, taskCt, type=None, objSet=None, goal=None):
      # Входные параметры задачи
      # Обязательный параметр - категория задачи: обучение модели "train", тестирование модели без предварительного обучения "test"
      # "служебные" = {проверка наличия нужных классов,БД, моделей}
      self._taskCt = taskCt #str

      #Необязательные входные параметры задачи, нужны при категориях "train" и "test"
      self._taskType = type  #str - тип задачи {classification, detection, segmentation}
      self._objects = objSet #set of strings - набор категорий объектов интереса
      self._goal = goal # dictionary - {"метрика" : желаемое значение метрики } 

      #Другие атрибуты задачи, которые будут использоваться при обучении моделей
      self.augmenParams=None
      self.trainParams=None
      self.curStrategy=None
      self.modelName=None
      self.curTrainingSubFolder=None
      self.generators=None
      self.model=None

      # Атрибуты, которые будут опеределены в ходе решения задачи
      #self.data=None 
      #self.model=None
      #self.suitModels
      #self.localHistory=None


    @property
    def taskCt(self):
        return self._taskCt

    @property
    def taskType(self):
        return self._taskType    

    @property
    def objects(self):
        return self._objects

    @property
    def goal(self):
        return self._goal

class State:
    '''
    Состояние решателя
    '''

    def __init__(self, task):
        self.task = task  # решаемая задача
        self.curState ='firstCheck'  # текущее состояние решателя

        #Атрибуты решателя,котоые используются для взаимодействия с пользователем
        #state.message =  str.format    #сообщение для пользователя
        #state.actions                  #что нужно делать в той или иной ситуации по выбору пользователя


class Rule(ABC):
    '''
    Базовый тип приёма
    Имеет два абязательных метода can_apply и apply
    '''

    @abstractmethod
    def can_apply(self, state: State):
        '''
        Проверка целесообразности применения приёма
        Returns:
        bool - признак целесообразности применения приёма
        '''
        pass

    @abstractmethod
    def apply(self, state: State):
        '''
        Применение приёма
        '''
        pass


#####################################################################


@rule 
class CheckSuitableModelExistence(Rule):
  '''
  Применим данный прием только в самом начале при исходном состоянии решателя state.curState == 'firstCheck'
  Проверка наличия подходящей модели в базе modelsBase.csv.
  В случае удачи переходим в состояние "UserDec" с предложением: остановиться на существующей модели или обучить новую
  В противном случае переходим в состояние "DB" для создания базы данных по объектам в атрибуте state.task.objects и типе задачи state.task.taskType  
  '''

  def can_apply(self, state: State):
    return (state.task.taskCt == "train" and state.curState == 'firstCheck')

  def apply(self, state: State):
    modelDB = pd.read_csv('./ModelsBase.csv', sep=',')
    s=[]
    for i in range(len(modelDB)):
      row=modelDB.iloc[i]
      if (row['TaskType']==task.taskType and strToSet(row['Objects']) == task.objects and DictKeysToStr(row['Metrics'])==list(state.task.goal.keys())[0] 
          and DictValToFL(row['Metrics'])>=state.task.goal[list(state.task.goal.keys())[0]]):
        s.append(row)

    if not s:
      state.curState='DB'
    else:
      state.curState='UserDec'
      state.message='There is a suitable model in our base of trained models. Would you like to train an another model? "No" - 0; "Yes" - 1. '
      state.actions={'0' : 'Done', '1' : 'DB'}
      state.task.suitModels=s



@rule 
class UserDec(Rule):

  '''
  Переходим при необходимости взаимодейсвия с пользователем.
  Переходим в состояние state.curState по ответу пользователи и предписанным в таком случае действиям.
  Если состояние == "Done" - завершить исполнение скрипта и сохранить "локальную" историю обучения в "глобальную" историю
  '''

  def can_apply(self, state):
      return ( state.task.taskCt == "train" and state.curState == 'UserDec' )

  def apply(self, state):

    print(state.message)
    answer=input()
    state.curState=state.actions[str(answer)]

    if state.curState=='Done':
      if hasattr(state.task,'localHistory'):
        state.task.localHistory.to_csv('./TrainedNN/'+state.task.modelName+ '/' + state.task.modelName + '.csv')
        GLHistory=pd.read_csv('ModelTrainingHistory.csv')
        del GLHistory['Unnamed: 0']
        for i in range(len(state.task.localHistory)-1):
          GLHistory=GLHistory.append(state.task.localHistory.iloc[i+1], ignore_index=True)
        GLHistory.to_csv('ModelTrainingHistory.csv')

      state.task.model=state.task.suitModels
      state.task.suitModels=None
    state.message=None
    state.actions=None


@rule
class CreateDataBase(Rule):
  '''
  Множество приемов, характеризующихся состоянием "DB"
  Для дальнейшей работы необходимо перейти в другое состояние (например, "Training" для дальнейшего обучения )
  Также для обучения нужно иметь словарик state.task.data с тремя .csv файлами, входным разрешением изображения и ограничениями приемов аугментации к сформированной БД(в таком виде они будут
   занесены как аргументы генераторов изображений)
  '''
  def can_apply(self, state: State):
    return (state.task.taskCt == "train" and state.curState == 'DB' )

  def apply(self, state: State):
    state.task.data={'Train' : './CD_Train.csv', 'Validation' : './CD_Validation.csv', 'Test' : './CD_Test.csv', 'Dim' : (224,224,3), 'AugmenConstr' : {'vertical_flip' : None}}
    state.curState='Training'

##########################################################################

'''
Множество приемов, характеризующихся состоянием решателя "Training". В этом наборе происходит выбор первоначальной стратегии обучения модели, создание директории и "локальной" историю для
решаемой задачи, установка необходимых для обучения параметров, создание самой модели, её обучение, тестирование, по итогам которого происходит переход в другое состояние решателя:
- в случае достижения цели задачи пользователю предлагается завершить работу модуля (при это решатель перейдет в состояние "Done") или обучить другую модель (переход в состояние "ReTraining")
- в противном случае автоматически переход в состояние "ReTraining"
'''

@rule
class ChoosingGeneralStrategy(Rule):
  '''
  Выбор первоначальной стратегии обучения модели. Загружаем "глобальную" историю и по типу задачи, количеству категорий объектов и значению достигнутой метрики (с заранее выбранной погрешностью eps)
  По выбраннной стратегии заполняются атрубуты trainParams и augmenParams, которые потом могут меняться в рамках выбранной стратегии
  '''
  def can_apply(self, state: State):
    return (state.task.taskCt == "train" and state.curState =='Training' and (state.task.curStrategy is None) and (state.task.trainParams is None) and (state.task.augmenParams is None))

  def apply(self, state: State):
    GLHistory=pd.read_csv('ModelTrainingHistory.csv')
    curStrategy=None
    ind=0
    while curStrategy is None:
      row=GLHistory.iloc[ind]
      if (row['TaskType']==task.taskType and len(strToSet(row['Objects'])) == len(task.objects) and row['Metrics'] == list(state.task.goal.keys())[0] 
          and row['Reached_metric_value'] >= state.task.goal[list(state.task.goal.keys())[0]]-eps):
        curStrategy=row
        break
      ind+=1


    if ind<len(GLHistory)-1:
      for i in range(ind+1,len(GLHistory)):
        row=GLHistory.iloc[i]
        if (row['TaskType']==task.taskType and len(strToSet(row['Objects'])) == len(task.objects) and row['Metrics'] == list(state.task.goal.keys())[0] 
            and row['Reached_metric_value'] >= curStrategy['Reached_metric_value']):
          curStrategy=row


    state.task.augmenParams=distAllAugmenParams(curStrategy['Augmen_params'], state.task.data['AugmenConstr'])
    curStrategy['Augmen_params']=[state.task.augmenParams]
    state.task.curStrategy=curStrategy
    state.task.trainParams={'Optimizer': curStrategy['Optimizer'] , 'Epochs' : curStrategy['Epochs'] , 'Loss' : curStrategy['Loss'] ,
                            'Metrics': curStrategy['Metrics'], 'Batch_size': curStrategy['Batch_size'] }


@rule
class CreateINfrastructureForTrainingModel(Rule):
  '''
  Попадаем в этот прием только, когда выбрали первоначальную стратегию обучения.
  Создается директория для хранения результатов и "локальная" история обучения моделей при решении конкретного экземпляра задачи
  '''
  def can_apply(self, state: State):
    return (state.task.taskCt == "train" and state.curState =='Training' and (not hasattr(state.task, 'localHistory'))   and  (state.task.curStrategy is not None)  )

  def apply(self, state: State):
    tt=time.localtime(time.time())
    state.task.modelName=state.task.taskType+'_'+str(state.task.objects)+'_'+str(tt.tm_year)+'_'+str(tt.tm_mon)+'_'+str(tt.tm_mday)+'_'+str(tt.tm_hour)+'_'+str(tt.tm_min)
    model_path="./TrainedNN" + '/'+state.task.modelName
    if not os.path.exists(model_path):
            os.makedirs(model_path)

    
    state.task.localHistory=pd.DataFrame({'TaskType': state.task.taskType, 'Objects': [state.task.objects], 'Pipeline' : state.task.curStrategy['Pipeline'],
      'ModelLastLayers' : [state.task.curStrategy['ModelLastLayers']], 'Augmen_params' :  [state.task.curStrategy['Augmen_params']], 'Optimizer': state.task.curStrategy['Optimizer'],
      'Batch_size' : state.task.curStrategy['Batch_size'], 'Epochs' : state.task.curStrategy['Epochs'], 'Loss' :  state.task.curStrategy['Loss'], 'Metrics':  state.task.curStrategy['Metrics'], 
      'Reached_metric_value' : 0.0, 'TrainingSubFolder' : '', 'DataUsed' : [state.task.data], 'Additional_params' : [{}] })


@rule
class DataGenerator(Rule):
  '''
  Прием для создания генераторов изображений по заданным в curStrategy параметрам аугментации
  В этот прием попадем как при первичном обучении, так и при смене параметров аугментации после обучения модели
  '''


  def can_apply(self, state):
    return (state.task.taskCt == "train" and state.curState =='Training' and  (state.task.curStrategy is not None) and  (state.task.augmenParams is not None) and (state.task.generators is None) )
  def apply(self, state):
    
    df_train=pd.read_csv(state.task.data['Train'])
    df_validation=pd.read_csv(state.task.data['Validation'])
    df_test=pd.read_csv(state.task.data['Test'])

    dataGen=keras.preprocessing.image.ImageDataGenerator(state.task.augmenParams)

    Train_generator=dataGen.flow_from_dataframe(df_train,directory='./Databases/Kaggle_CatsVSDogs', x_col="images", y_col="target",
                        target_size=state.task.data['Dim'][0:2], class_mode='raw', batch_size=state.task.curStrategy['Batch_size'])
    Validation_generator=dataGen.flow_from_dataframe(df_validation ,directory='./Databases/Kaggle_CatsVSDogs', x_col="images", y_col="target",
                        target_size=state.task.data['Dim'][0:2], class_mode='raw', batch_size=state.task.curStrategy['Batch_size'])
    Test_generator=dataGen.flow_from_dataframe(df_test,directory='./Databases/Kaggle_CatsVSDogs', x_col="images", y_col="target",
                        target_size=state.task.data['Dim'][0:2], class_mode='raw', batch_size=state.task.curStrategy['Batch_size'])     
    state.task.generators=[Train_generator,Validation_generator,Test_generator] 


@rule
class ModelAssembling(Rule):
  '''
  Прием для cоздания модели по основе сети "Pipeline" и добавленных к ней слоям, указанных в параметре "ModelLastLayers".
  Создание субдиректории для хранения архитектуры, схемы, весов и истории конкретного экземпляра модели
  В этот прием попадем при первичном обучении, и при обучении нового экземпляра модели
  '''

  def can_apply(self, state: State):
    return (state.task.taskCt == "train" and state.curState =='Training' and  (state.task.curStrategy is not None) and (state.task.modelName is not None) 
    and (state.task.model is None))

  def apply(self, state: State):

      x=keras.layers.Input(shape=(state.task.data['Dim']))
      y=keras.models.load_model('./Architectures/'+state.task.curStrategy['Pipeline']+'.h5')(x)
      #parcer
      y=keras.layers.Dense(units=64)(y)
      y=keras.layers.Activation(activation='relu')(y)
      y=keras.layers.Dense(units=1)(y)
      y=keras.layers.Activation(activation = 'sigmoid')(y)
      state.task.model=keras.models.Model(inputs=x, outputs=y)

      tt=time.localtime(time.time())
      state.task.curTrainingSubFolder=str(tt.tm_year)+'_'+str(tt.tm_mon)+'_'+str(tt.tm_mday)+'_'+str(tt.tm_hour)+'_'+str(tt.tm_min)
      model_path='./TrainedNN/'+state.task.modelName+'/'+state.task.curTrainingSubFolder
      if not os.path.exists(model_path):
              os.makedirs(model_path)


      state.task.model.save(model_path + '/Model.h5')
      keras.utils.plot_model(state.task.model, to_file=(model_path + '/ModelPlot.png'), rankdir='TB', show_shapes=True)


@rule
class FitModel(Rule):
  '''
  Прием для обучения модели. К этому моменту должны быть определены все параметры (обучения, аугментации) и созданы все необходимые объекты (генераторы, модель, директории).
  По окончании обучения происходит тестирование и проверка на достижение поставленной цели(в атрибуте "goal"). 
  Записывается результат в "локальную" историю обучения модели.
  Происходит переход в другое состояние решателя:
  - в случае достижения цели задачи пользователю предлагается завершить работу модуля (при это решатель перейдет в состояние "Done") или обучить другую модель (переход в состояние "ReTraining")
  - в противном случае автоматически переход в состояние "ReTraining"
  '''


  def can_apply(self, state):
      return (state.task.taskCt == "train" and state.curState =='Training' and (state.task.generators is not None) and (state.task.curTrainingSubFolder is not None) and
        (state.task.trainParams is not None) and  (state.task.augmenParams is not None)  and  (state.task.curStrategy is not None) )

  def apply(self, state):

      state.task.model.compile(optimizer=state.task.trainParams['Optimizer'], loss=state.task.trainParams['Loss'], 
                                metrics=[state.task.trainParams['Metrics']])

      C_Log= keras.callbacks.CSVLogger( './TrainedNN/' + state.task.modelName + '/' + state.task.curTrainingSubFolder +'/History.csv')
      C_Ch = keras.callbacks.ModelCheckpoint( './TrainedNN/' + state.task.modelName + '/' + state.task.curTrainingSubFolder + '/weights'  +'-{epoch:02d}.h5',
                                              monitor='val_'+state.task.trainParams['Metrics'], save_best_only=True,  save_weights_only=False, verbose=1)

      state.task.model.fit_generator(generator=state.task.generators[0], steps_per_epoch=(len(state.task.generators[0].filenames) // state.task.trainParams['Batch_size']), 
                epochs=state.task.trainParams['Epochs'], validation_data=state.task.generators[1], callbacks=[C_Log,C_Ch],
                validation_steps=(len(state.task.generators[1].filenames) // state.task.trainParams['Batch_size']) )
      
      
      scores = state.task.model.evaluate_generator(state.task.generators[2], steps=None,verbose=1 )
      
      if scores[1]>=state.task.goal[list(state.task.goal.keys())[0]]:
        state.curState='UserDec'
        state.message='Our system trained a model that achieved its goal. Would you like to train an another model? "No" - 0; "Yes" - 1. '
        state.actions={'0' : 'Done', '1' : 'ReTraining'}
        state.task.suitModels=state.task.model
      else:
        state.curState='ReTraining'

              
      new_row={'TaskType': state.task.taskType, 'Objects': state.task.objects, 'Pipeline' : state.task.curStrategy['Pipeline'],
          'ModelLastLayers' : state.task.curStrategy['ModelLastLayers'], 'Augmen_params' :  state.task.augmenParams, 'Optimizer': state.task.trainParams['Optimizer'],
          'Batch_size' : state.task.trainParams['Batch_size'], 'Epochs' : state.task.trainParams['Epochs'], 'Loss' :  state.task.trainParams['Loss'], 'Metrics':  state.task.trainParams['Metrics'], 
          'Reached_metric_value' : scores[1], 'TrainingSubFolder' : state.task.curTrainingSubFolder,'DataUsed' : state.task.data, 'Additional_params' : {} }

      state.task.localHistory=state.task.localHistory.append(new_row, ignore_index=True)


#############################################################

'''
Множество приемов для изменения стратегии обучения, параметров аугментации, параметров обучения ->переходим в состояние "Training" для продолжения обучения с новыми значениями атрибутов
-> переходим в состояние "DB" при неоходимости изменения базы данных

если НЕдообучение, то  state.task.model=None
если меняем статегию обучения целиком, то поменяли стратегию = задали curStrategy, TrainParams, AugmenParams, state.task.generators=None -> доступно: DataGen, ModelAssembling -> FitModel
если поменяли только параметры аугментации, то state.task.generators=None -> доступно: DataGen, ModelAssembling -> FitModel
если поменяли только параметры обучения, то -> доступно: ModelAssembling -> FitModel
'''
@rule
class ChangeParameters(Rule):


    def can_apply(self, state):
        return (state.task.taskCt == "train" and state.curState =='ReTraining' )

    def apply(self, state):

        state.curState='Training'
        state.task.curTrainingSubFolder=None # для нового обучения
        # если НЕдообучение, то  state.task.model=None
        state.task.model=None
        state.task.generators=None


###################################
'''
Вспомогательные функции-парсеры.
'''

def strToSet(sstr):
  frList=sstr.split('\'')
  s=set()
  for i in range(1,len(frList),2):
    s.add(frList[i])
  return s

def strToDict(sstr):
  json_acceptable_string = sstr.replace("'", "\"")
  dic = json.loads(json_acceptable_string)
  return dic

def DictKeysToStr(sstr, i=0):
  dic = strToDict(sstr)
  key = list(dic.keys())[i]
  return key

def DictValToFL(sstr, i=0):
  dic = strToDict(sstr)
  key = list(dic.keys())[i]
  return dic[key]

def distAllAugmenParams(hist, constrDIC):
  tmp=hist.replace
  tmp=hist.split("{")[1]
  tmp=tmp.split("}")[0]
  if '\'<' in tmp:
    tmp=tmp.replace('\'<', '<')
  if '>\'' in tmp:
    tmp=tmp.replace('>\'', '>')

  tmp=tmp.split('\'')[1:]
  keys=[]
  values=[]
  for i in range(len(tmp)):
    if (i%2==0):
      keys.append(tmp[i])
    else:
      tm=tmp[i].split(': ')[1]
      tm=tm.split(', ')[0]
      values.append(tm)

  allparams={}
  for i in range(len(keys)):
    if values[i]=='None':
      allparams[keys[i]]=None
    elif values[i]=='False':
      allparams[keys[i]]=False
    elif values[i]=='True':
      allparams[keys[i]]=True
    elif keys[i]=='fill_mode':
      allparams[keys[i]]=values[i]
    elif keys[i]=='preprocessing_function':
      allparams[keys[i]]=keras.applications.resnet.preprocess_input
    else:
      allparams[keys[i]]=float(values[i])

  all=set(allparams.keys())
  constr=set(constrDIC.keys())

  augmen_params={}
  for key in all:
    if key in constr:
      augmen_params[key]=constrDIC[key]
      constr.remove(key)
    else:
      augmen_params[key]=allparams[key]
  for key in constr:
    augmen_params[key]=constrDIC[key]
  return augmen_params

###################################


def solve(task: Task, rules):
  k=0

  state = State(task)
  pos = 0
  print(rules)
  while state.curState is not "Done":
    print(state.curState)
    if rules[pos].can_apply(state):
      print(rules[pos])
      rules[pos].apply(state)
    pos = (pos + 1) % len(rules)
    k+=1
    if k>100:
      break
  return state



# Создаём и решаем задачу создания модели нейронной сети
task = Task( "train", type="classification", objSet={"cats","dogs"}, goal={'accuracy':0.9}) 
w=solve(task, rules)

w.curState

