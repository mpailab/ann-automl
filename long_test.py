from ann_automl.core.nn_auto import create_classification_model
categories = ['horse', 'elephant', 'giraffe', 'sheep', 'cow']
model = create_classification_model(categories, output_dir='tests/out_1', target_accuracy=0.9, time_limit=3600*40)

