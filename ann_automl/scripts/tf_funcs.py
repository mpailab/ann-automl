import json
import os
import sys

import numpy as np

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras


def classify(model, objects, image_path, preprocessing=None):
    """ Classifies image and returns object name and score.

    Args:
        model (keras.Model): model to use for classification
        objects (list of str): list of objects that model can classify
        image_path (str): path to image to classify
        preprocessing (str): name of preprocessing function to use
    Returns:
        pair -- object name and score
    """
    channels_first = model.input_shape[1] == 3
    target_size = model.input_shape[1:3]
    img = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img, data_format='channels_first' if channels_first else None)
    if preprocessing is not None:
        fpath = preprocessing.split('.')
        if fpath[:2] != ['keras', 'applications']:
            raise ValueError(f'Invalid preprocess function name: {preprocessing}. '
                             'Only keras applications preprocess functions are supported')
        m = keras.applications
        for attr in fpath[2:]:
            m = getattr(m, attr)
        img_array = m(img_array)

    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    if len(objects) == 2:
        index = 1 if score > 0.5 else 0
    else:
        index = np.argmax(score)

    return objects[index], score[index]


def classify_image(model_path, objects, image_path, preprocessing=None):
    """ Classifies image and returns object name and score.

    Args:
        model_path (str): path to model
        objects (list of str): list of objects that model can classify
        image_path (str): path to image to classify
        preprocessing (str): name of preprocessing function to use

    Returns:
        pair -- object name and score
    """
    model = keras.models.load_model(model_path)
    return classify(model, objects, image_path, preprocessing)


def classify_all_in_directory(model_path, objects, directory_path, image_ext='jpg;png;jpeg;bmp', preprocessing=None):
    """
    Classifies all images in directory and returns dictionary with results.

    Args:
        model_path (str): path to model
        objects (list of str): list of objects that model can classify
        directory_path (str): path to directory with images to classify
        image_ext (str): image extensions to classify
        preprocessing (str): name of preprocessing function to use

    Returns:
        dict -- dictionary with results. Key is image name, value is tuple with object name and score.
    """
    from tqdm import tqdm
    model = keras.models.load_model(model_path)
    images = os.listdir(directory_path)
    # select only images
    image_extensions = set('.'+x for x in image_ext.split(';'))
    images = [image for image in images if os.path.splitext(image)[1] in image_extensions]
    if not images:
        print(f'No images found in {directory_path}', file=sys.stderr)
    result = {}
    for image in tqdm(images, desc=f'Classifying images in {directory_path}'):
        image_path = os.path.join(directory_path, image)
        result[image] = classify(model, objects, image_path, preprocessing)
    return result
