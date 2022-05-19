import json
from PIL import Image
import os
import time

# preparing structure for JSON COCO-like annotations file
json_result = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [],
    "segment_info": [] #No segment info for Dogs vs. Cats dataset
}

json_result["info"] = {
    "description": "Dogs vs Cats",
    "url": "https://www.kaggle.com/c/dogs-vs-cats",
    "version": "1.0",
    "year": 2013,
    "contributor": "Asirra ",
    "date_created": "2013/09/25"
}

json_result["licenses"] = [
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
]

json_result["categories"] = [
    {"supercategory": "animal", "id": 16, "name": "cat"},  # COCO-like IDs
    {"supercategory": "animal", "id": 17, "name": "dog"}
]


im_id = 0
anno_id = 0
directory = 'train/'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        im = Image.open(directory + filename)
        width, height = im.size
        ti_c = os.path.getctime(directory + filename) 
        c_ti = time.ctime(ti_c) 
        image_info_dict = {"license": 1,
                           "file_name": filename,
                           "coco_url": "",
                           "height": height,
                           "width": width,
                           "date_captured": str(c_ti),
                           "flickr_url": "",
                           "id": im_id} 
        category = 0
        class_name_str = filename.split('.')[0]
        if class_name_str == 'cat':
            category = 16
        elif class_name_str == 'dog':
            category = 17
        annotation_dict = {
            "segmentation": [], #Dogs vs Cats is not segmented
            "num_keypoints": 0,
            "area": 0,
            "iscrowd": 0,
            "keypoints": [],
            "image_id": im_id,
            "bbox": [],
            "category_id": category,
            "id": anno_id
        }
        json_result['images'].append(image_info_dict)
        json_result['annotations'].append(annotation_dict)
        im_id += 1
        anno_id += 1

with open('dogs_vs_cats_coco_anno.json', 'w') as fp:
    json.dump(json_result, fp)