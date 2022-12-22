import os
import sys
from tqdm import tqdm


def create_annotations(ds_dir, anno_file):
    """
    Args:
        ds_dir: path to the dataset folder
        anno_file: path to the output file with annotations (if None, resuls are not saved)
    Returns:
        dict: dataset info in the format of the COCO dataset
    """
    import json
    from PIL import Image
    import os
    import time

    # preparing structure for JSON COCO-like annotations file
    json_result = {
        "info": {
            "description": "Dogs vs Cats",
            "url": "https://www.kaggle.com/c/dogs-vs-cats",
            "version": "1.0",
            "year": 2013,
            "contributor": "Asirra ",
            "date_created": "2013/09/25"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            # {"supercategory": "animal", "id": 16, "name": "cat"},  # COCO-like IDs
            # {"supercategory": "animal", "id": 17, "name": "dog"}
        ],
        "segment_info": []
    }

    im_id = 0
    anno_id = 0
    cls_ids = {}
    for class_name in os.listdir(ds_dir):
        if not os.path.isdir(os.path.join(ds_dir, class_name)):
            continue
        cls_id = len(cls_ids)
        cls_ids[class_name] = cls_id
        json_result['categories'].append({"supercategory": "", "id": cls_id, "name": class_name})
        for im_name in os.listdir(os.path.join(ds_dir, class_name)):
            if not os.path.isfile(os.path.join(ds_dir, class_name, im_name)):
                continue
            im_id += 1
            anno_id += 1
            im_path = os.path.join(ds_dir, class_name, im_name)
            im = Image.open(im_path)
            json_result['images'].append({
                "license": 1,
                "file_name": os.path.join(class_name, im_name),
                "coco_url": "",
                "height": im.height,
                "width": im.width,
                "date_captured": "",
                "flickr_url": "",
                "id": im_id
            })
            json_result['annotations'].append({
                "segmentation": [],
                "area": im.height * im.width,
                "iscrowd": 0,
                "image_id": im_id,
                "bbox": [0, 0, im.width, im.height],
                "category_id": cls_id,
                "id": anno_id
            })

    if anno_file is not None:
        with open(anno_file, 'w') as f:
            json.dump(json_result, f)
    return json_result


def download_tfds(tfds_name, ds_path, output_dir):
    import tensorflow as tf
    # if tensorflow_datasets is not installed, install it
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print('Installing tensorflow_datasets...')
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow_datasets"])
        import tensorflow_datasets as tfds

    # Download the dataset and saves it as folders of images.
    ds, info = tfds.load(tfds_name, split='train', shuffle_files=True,
                         as_supervised=True, with_info=True, data_dir=ds_path)

    anno_file = os.path.join(output_dir, 'annotations.json')

    if not os.path.exists(anno_file):
        # save the dataset as folder with folders of images (one folder per class)
        ntrain = info.splits['train'].num_examples
        print(f"Number of training examples: {ntrain}")
        # create a folder for the dataset
        os.makedirs(output_dir, exist_ok=True)
        # create a folder for each class
        for i in range(info.features['label'].num_classes):
            class_name = info.features['label'].int2str(i)
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
        # copy images to the corresponding folders
        nums = [0] * info.features['label'].num_classes
        for image, label in tqdm(ds, total=ntrain, desc='Saving images to folders'):
            class_id = label.numpy()
            class_name = info.features['label'].int2str(class_id)
            nums[class_id] += 1
            image_name = f"{class_name}_{nums[class_id]:0{len(str(ntrain))}}.jpg"
            image_path = os.path.join(output_dir, class_name, image_name)
            # save in jpg format
            serialized_im = tf.image.encode_jpeg(image)
            tf.io.write_file(image_path, serialized_im)
        create_annotations(output_dir, anno_file)
    else:
        print('Dataset already downloaded')

    result = {'annotations': anno_file,
              'images': output_dir,
              'version': str(info.version),
              'name': info.name,
              'description': info.description,
              'license': info.redistribution_info.license,
              'url': info.homepage,
              'citation': info.citation}
    return result
