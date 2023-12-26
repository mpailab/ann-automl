import keras_cv
import tensorflow as tf
from keras.models import load_model
from ann_automl.nnplot.vislib import show_detections
from ann_automl.core.nnfuncs import load_image
import pandas as pd
from ast import literal_eval

categories = ['cat', 'dog']
model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    num_classes=len(categories),
    bounding_box_format="xywh",
)
model.load_weights('data/trainedNN/train_17_18_DT_2023_12_11_18_18_44/best_weights-15.h5')

pascal_shape=640 
df_test = pd.read_csv('data/trainedNN/train_17_18_DT_2023_12_11_18_18_44/test.csv',delimiter=';',converters={'bbox': literal_eval})
bboxes=[]
classes=[]
image_paths=[]
i=0
for fn,group in df_test.groupby('images'):
    i+=1
    if i>10:
        print(fn,group['bbox'].to_list())
        image_paths.append(fn)
        bboxes.append(group['bbox'].to_list())
        classes.append(group['target'].to_list())
        if len(image_paths)>=8: break

inference_resizing = keras_cv.layers.Resizing(pascal_shape, pascal_shape, bounding_box_format="xywh",pad_to_aspect_ratio=True)
images=tf.zeros((0,pascal_shape,pascal_shape,3))
for i in range(8):
    image = load_image(image_paths[i])
    res=inference_resizing({'images':image,"bounding_boxes":{'boxes': bboxes[i],'classes': classes[i]}})
    bboxes[i]=res["bounding_boxes"]['boxes'].to_list()
    img=tf.expand_dims(res['images'],0)
    images=tf.concat([images,img],0)

boxes={'boxes': tf.ragged.constant(bboxes),'classes': tf.ragged.constant(classes)}
pred=model.predict(images)
show_detections(images,pred,y_true=boxes,path='result.png')

