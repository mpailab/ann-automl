# -*- coding: utf-8 -*-

import os
'''
from google.colab import drive
drive.mount('/content/drive')

os.chdir('./drive/My Drive')
!ls
'''

from abc import ABC, abstractmethod
import time
import json
import pandas as pd
import keras

#Создание базы данных обученных моделей

df=pd.DataFrame({'TaskType': 'detection', 'Objects': [{'cats', 'mice'}], 'ModelAddress' : './TrainedNN/New_NetworkFirst', 
 'Metrics' :  [{'accuracy' : 0.91}] }  )  # обертка списком нужна только при создании БД

new_row={'TaskType': 'classification', 'Objects': {'cats', 'dogs'}, 'ModelAddress' : './TrainedNN/New_NetworkPP', 
 'Metrics' :  {'accuracy' : 0.91} }
df=df.append(new_row, ignore_index=True)

new_row={'TaskType': 'classification', 'Objects': {'cats', 'dogs'}, 'ModelAddress' : './TrainedNN/New_NetworkFirst', 
 'Metrics' :  {'accuracy' : 0.80} }
df=df.append(new_row, ignore_index=True)

new_row={'TaskType': 'classification', 'Objects': {'cats', 'mice'}, 'ModelAddress' : './TrainedNN/New_NetworkFirst', 
 'Metrics' :  {'accuracy' : 0.93} }
df=df.append(new_row, ignore_index=True)

new_row={'TaskType': 'classification', 'Objects': {'cats', 'mice'}, 'ModelAddress' : './TrainedNN/New_NetworkFirst', 
 'Metrics' :  {'loss' : 0.93} }
df=df.append(new_row, ignore_index=True)

df.to_csv('ModelsBase.csv')
df

#Создание "глобальной" истории обучения моделей


GLH=pd.DataFrame({'TaskType': 'classification', 'Objects': [{'cats', 'dogs'}], 'Pipeline' : 'ResNet50', 'ModelLastLayers' : [[ {'Type': keras.layers.Dense, 'units': 64},
                  {'Type': keras.layers.Activation, 'activation': 'relu'}, {'Type': keras.layers.Dense, 'units': 1}, {'Type': keras.layers.Activation, 'activation': 'sigmoid'}]], 
                 'Augmen_params' : [{'preprocessing_function': keras.applications.resnet.preprocess_input, 'horizontal_flip' : True}],
                 'Optimizer': 'Adam', 'Batch_size' : 16, 'Epochs' : 5 , 'Loss' : 'binary_crossentropy', 'Metrics':'accuracy','Reached_metric_value' : 0.85, 
                  'TrainingSubFolder' : './TrainedNN/New_NetworkPP/New_NetworkSUB', 'DataUsed' : [['./CD_Train.csv','./CD_Validation.csv','./CD_Test.csv']], 'Additional_params' : [{}] } )

new_row={'TaskType': 'classification', 'Objects': {'cats', 'mice'}, 'Pipeline' : 'ResNet50', 'ModelLastLayers' : [ {'Type': keras.layers.Dense, 'units': 64},
                  {'Type': keras.layers.Activation, 'activation': 'relu'}, {'Type': keras.layers.Dense, 'units': 1}, {'Type': keras.layers.Activation, 'activation': 'sigmoid'}], 
                 'Augmen_params' : {'preprocessing_function': keras.applications.resnet.preprocess_input, 'horizontal_flip' : True},
                 'Optimizer': 'Adam', 'Batch_size' : 16, 'Epochs' : 5 , 'Loss' : 'binary_crossentropy', 'Metrics':'accuracy','Reached_metric_value' : 0.95, 
                  'TrainingSubFolder' : './TrainedNN/New_NetworkPP/New_NetworkSUB', 'DataUsed' : ['./CD_Train.csv','./CD_Validation.csv','./CD_Test.csv'], 'Additional_params' : {} } 

GLH=GLH.append(new_row, ignore_index=True)

new_row={'TaskType': 'detection', 'Objects': {'cats', 'mice'}, 'Pipeline' : 'ResNet50', 'ModelLastLayers' : [ {'Type': keras.layers.Dense, 'units': 64},
                  {'Type': keras.layers.Activation, 'activation': 'relu'}, {'Type': keras.layers.Dense, 'units': 1}, {'Type': keras.layers.Activation, 'activation': 'sigmoid'}], 
          'Augmen_params' : {'preprocessing_function': keras.applications.resnet.preprocess_input, 'horizontal_flip' : True},
          'Optimizer': 'Adam', 'Batch_size' : 16, 'Epochs' : 5 , 'Loss' : 'binary_crossentropy', 'Metrics':'accuracy','Reached_metric_value' : 0.91, 
          'TrainingSubFolder' : './TrainedNN/New_NetworkPP/New_NetworkSUB', 'DataUsed' : ['./CD_Train.csv','./CD_Validation.csv','./CD_Test.csv'], 'Additional_params' : {} }


GLH=GLH.append(new_row, ignore_index=True)

GLH.to_csv('ModelTrainingHistory.csv')
GLH

'''
Пример выхода работающего модуля

[<__main__.CheckSuitableModelExistence object at 0x7f187ae94410>, <__main__.UserDec object at 0x7f187ae94e50>, <__main__.CreateDataBase object at 0x7f187ae944d0>, <__main__.ChoosingGeneralStrategy object at 0x7f187ae943d0>, <__main__.CreateINfrastructureForTrainingModel object at 0x7f187ae94750>, <__main__.DataGenerator object at 0x7f187ae94dd0>, <__main__.ModelAssembling object at 0x7f187ae94110>, <__main__.FitModel object at 0x7f187ae94290>, <__main__.ChangeParameters object at 0x7f187ae94f90>]
firstCheck
<__main__.CheckSuitableModelExistence object at 0x7f187ae94410>
UserDec
<__main__.UserDec object at 0x7f187ae94e50>
There is a suitable model in our base of trained models. Would you like to train an another model? "No" - 0; "Yes" - 1. 
1
DB
<__main__.CreateDataBase object at 0x7f187ae944d0>
Training
<__main__.ChoosingGeneralStrategy object at 0x7f187ae943d0>
Training
<__main__.CreateINfrastructureForTrainingModel object at 0x7f187ae94750>
Training
<__main__.DataGenerator object at 0x7f187ae94dd0>
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:219: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
Found 17500 validated image filenames.
Found 3750 validated image filenames.
Found 3750 validated image filenames.
Training
<__main__.ModelAssembling object at 0x7f187ae94110>
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
Training
<__main__.FitModel object at 0x7f187ae94290>
UserDec
UserDec
UserDec
<__main__.UserDec object at 0x7f187ae94e50>
Our system trained a model that achieved its goal. Would you like to train an another model? "No" - 0; "Yes" - 1. 
1
ReTraining
ReTraining
ReTraining
ReTraining
ReTraining
ReTraining
ReTraining
<__main__.ChangeParameters object at 0x7f187ae94f90>
Training
Training
Training
Training
Training
Training
<__main__.DataGenerator object at 0x7f187ae94dd0>
Found 17500 validated image filenames.
Found 3750 validated image filenames.
Found 3750 validated image filenames.
Training
<__main__.ModelAssembling object at 0x7f187ae94110>
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
Training
<__main__.FitModel object at 0x7f187ae94290>
UserDec
UserDec
UserDec
<__main__.UserDec object at 0x7f187ae94e50>
Our system trained a model that achieved its goal. Would you like to train an another model? "No" - 0; "Yes" - 1. 
0
'''