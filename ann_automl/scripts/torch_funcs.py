import json
import os
import shutil

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def classify(image_path, classes, model):
    """
    Classifies image by path and returns object name.
    Args:
        image_path: path to image to classify
        model: model to use for classification
    """
    target_size = model.input_shape[1:3]
    img = Image.open(image_path).resize(target_size)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
    score = predictions[0]
    if len(classes) == 2:
        index = 1 if score > 0.5 else 0
    else:
        index = np.argmax(score)

    return classes[index], score[index]


def classify_image(model_path, classes, image_path):
    """ Classifies image by path and returns object name.
    Args:
        model_path: path to model
        classes: list of classes that model can classify
        image_path: path to image to classify
    """
    model = torch.load(model_path)
    return classify(image_path, classes, model)


def classify_all_in_directory(model_path, classes, directory_path, image_ext='jpg;png;jpeg;bmp'):
    """
    Classifies all images in directory and returns dictionary with results.
    Args:
        model_path: path to model
        classes: list of classes that model can classify
        directory_path: path to directory with images to classify

    Returns:
        Dictionary with results. Key is image name, value is tuple with object name and score.
    """
    from tqdm import tqdm
    model = torch.load(model_path)
    images = os.listdir(directory_path)
    # select only images
    image_extensions = set('.'+x for x in image_ext.split(';'))
    images = [image for image in images if os.path.splitext(image)[1] in image_extensions]
    if not images:
        print(f'No images found in {directory_path}', file=sys.stderr)
    result = {}
    for image in tqdm(images, desc=f'Classifying images in {directory_path}'):
        image_path = os.path.join(directory_path, image)
        result[image] = classify(image_path, classes, model)
    return result


