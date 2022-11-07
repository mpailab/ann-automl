import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def classify_image(model_path, objects, image_path):
    """ Classifies image by path and returns object name.
    Can classify objects: {obj_list}
    :param image_path: path to image
    """
    model = torch.load(model_path)
    target_size = model.input_shape[1:3]
    img = Image.open(image_path).resize(target_size)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
    score = predictions[0]
    if len(objects) == 2:
        index = 1 if score > 0.5 else 0
    else:
        index = np.argmax(score)

    return objects[index], score[index]


if __name__ == '__main__':
    # parse arguments:
    # one required argument - path to image,
    # one optional - path to model config in json format (default: model.json)
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('--model_config', type=str, default='model.json', help='path to model config in json format')
    args = parser.parse_args()

    # load model config
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)

    # classify image
    if not os.path.exists(args.image_path):
        print(f'File {args.image_path} does not exist')
    else:
        object_name, score = classify_image(model_config['model_path'], model_config['objects'], args.image_path)
        print(f'Object: {object_name}, score: {score}')
