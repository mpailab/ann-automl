from ann_automl.core.nn_task import NNTask
from ann_automl.core.nnfuncs import train #create_and_train_model
from ann_automl.core.nn_recommend import recommend_hparams

categories = ['cat', 'dog']
target_accuracy=0.9
time_limit=3600*40
for_mobile=False
optimize_over_target=True

task = NNTask(category='train', type='segmentation', objects=categories, target=target_accuracy, 
              goals={'maximize': optimize_over_target}, time_limit=time_limit, for_mobile=for_mobile)
hparams = recommend_hparams(task)
hparams['model_arch']='Linknet'
hparams['input_shape']=256
hparams['epochs']=30
hparams['optimizer']='Adam'
hparams['learning_rate']=0.0005
hparams['batch_size']=8
hparams['loss']='categorical_crossentropy'
metrics, simple_params = train(task, hparams, timeout=60 * 60 * 24)

