import json
import os
import shutil
import sys

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def classify(image_path, classes, model, preprocessing=None):
    """
    Classifies image by path and returns object name.
    Args:
        image_path: path to image to classify
        classes: list of classes that model can classify
        model: model to use for classification
        preprocessing: name of preprocessing function to use
    """
    target_size = model.input_shape[1:3]
    img = Image.open(image_path).resize(target_size)
    img = transforms.ToTensor()(img)
    if preprocessing is not None:
        preprocessing = preprocessing.split('.')[2]
        if preprocessing == 'resnet':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(img)
        elif preprocessing == 'inception':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])(img)
        elif preprocessing == 'vgg16':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(img)
        elif preprocessing == 'xception':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])(img)
        elif preprocessing == 'densenet':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(img)
        elif preprocessing == 'mobilenet':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(img)
        elif preprocessing == 'nasnet':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])(img)
        elif preprocessing == 'efficientnet':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(img)
        elif preprocessing == 'inceptionresnet':
            img = transforms.Compose([
                transforms.Lambda(lambda x: x[[2, 1, 0]]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])(img)
        else:
            raise ValueError(f'Unknown preprocessing: {preprocessing}')

    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
    score = predictions[0]
    if len(classes) == 2:
        index = 1 if score > 0.5 else 0
    else:
        index = np.argmax(score)

    return classes[index], score[index]


def classify_image(model_path, classes, image_path, preprocessing=None):
    """ Classifies image by path and returns object name.
    Args:
        model_path: path to model
        classes: list of classes that model can classify
        image_path: path to image to classify
        preprocessing: name of preprocessing function to use
    """
    model = torch.load(model_path)
    return classify(image_path, classes, model, preprocessing)


def classify_all_in_directory(model_path, classes, directory_path, image_ext='jpg;png;jpeg;bmp', preprocessing=None):
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
        result[image] = classify(image_path, classes, model, preprocessing)
    return result


