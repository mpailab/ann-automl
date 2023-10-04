import os
import json
import yolov5
from PIL import Image
from datetime import datetime, date


def make_labeling_info():
    """
        Заполняет словарь информации, необходимой для разметки датасета. Необходима только для тестов,  в будущем
        подразумевается, что словарь будет заполнятся информацией от пользователя.
        upload_type - способ добавления изображений
        dataset_name - название добавляемого датасета
        nn - нейросеть, используемая для предварительной разметки датасета
    """
    labeling_info = dict()
    images_zip = input(
        "Выберите, каким образом добавить изображения: \n 1) Загрузить изображения из Директории \n "
        "2) Загрузить изображения zip-архивом")
    labeling_info["images_path"] = input("Введите путь до директории с изображениями")
    if images_zip == "1":
        labeling_info["images_zip"] = False
    else:
        labeling_info["images_zip"] = True
    labeling_info["images_name"] = input("Введите название датасета:")
    dict_of_nn = {"yolov5s": 'yolov5s', "yolov5n": 'yolov5n', "yolov5m": 'yolov5m', "yolov5l": 'yolov5l',
                  "yolov5x": 'yolov5x'}
    i = 0
    for key in dict_of_nn.keys():
        i = i + 1
        print(i, ")", key)
    nn_var = input("Выберите, какую нейросеть использовать для разметки: ")
    labeling_info["nn_core"] = dict_of_nn[nn_var]
    return labeling_info


def upload_images(upload_type, image_dir, dst_dir):
    """
        Добавляет изображения в котолог, находящийся по пути dst_dit

        Args:
            dst_dir (str): путь к каталогу, в котором будут храниться изображения
            upload_type (str): способ добавления изобрадений
    """
    if not os.listdir(image_dir):
        raise FileNotFoundError(f'Reference to empty directory ({image_dir})')
    if not upload_type:
        upload_from_folder(image_dir, dst_dir)
        return 0
    else:
        upload_from_zip(image_dir, dst_dir)
        return 0


def labeling(dst_dir=""):
    """
        Выполняет разметку изображений при помощи qsl Media labeler с последующей записью аннотаций в файл
        annotations.json.
        Args:
            dst_dir (str): путь к каталогу, в котором хранятся изображения
    """
    labeling_args = make_labeling_info()
    '''
    labeling_args = {"images_path" : "C:/Users/Alexander/Documents/tests/pictures",
                     "images_zip" : False,
                     "nn_core" : "yolov5x",
                     "images_name" : "microtest"}
    '''
    dst_dir = os.path.join(dst_dir, "data", "datasets", labeling_args["images_name"]).strip()
    labels_dict = pre_processing(labeling_args, dst_dir)
    with open(dst_dir + "/labels.json", "w") as outfile:
        json.dump(labels_dict, outfile)
    command = "qsl label " + dst_dir + "/labels.json"
    os.system(command)
    final_anno = post_processing(dst_dir)
    with open(dst_dir + "/annotations/annotations_file.json", "w") as outfile:
        json.dump(final_anno, outfile)
    return dst_dir


def pre_processing(labeling_info, dst_dir):
    """
        Запускает функции для загрузки изображений и создания словаря для разметчика.
        Args:
            dst_dir (str): путь к каталогу, в котором хранятся изображения
            labeling_info (dict): словарь, содержащий основную информацию о добавляемом датасете
    """
    if os.path.exists(dst_dir):
        raise FileNotFoundError(f'The dataset {labeling_info["images_name"]} being added is already presented')
    else:
        os.makedirs(dst_dir)
    os.makedirs(dst_dir + "/images")
    os.makedirs(dst_dir + "/annotations")
    error = upload_images(labeling_info["images_zip"], labeling_info["images_path"], dst_dir + "/images")
    if error != 0:
        raise ValueError(f'Error downloading images in {labeling_info["images_name"]} dataset')
    labeling_dict = make_anno_with_yolo(dst_dir, labeling_info["nn_core"] + ".pt")
    return labeling_dict


def make_anno_with_yolo(dst_dir, network):
    """
        Создает словарь в формате, пригодном для использования разметчиком qsl.
        Args:
            dst_dir (str): путь к каталогу, в котором хранятся изображения
            network (str): нейросеть, используемая для предварительной разметки изображений
    """
    data_dir = dst_dir + "/images"
    model = yolov5.load(network)
    list_of_images = []
    for filename in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, filename)):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                jpgfile = os.path.join(data_dir, filename).replace("\\", "/")
                list_of_images.append(jpgfile)
    result_dict = dict()
    result_dict["items"] = []
    regions = []
    for filename in list_of_images:
        result = model(filename)
        im = Image.open(filename)
        image_size = im.size
        picture_dict = {}
        picture_dict["target"] = filename
        picture_dict["labels"] = {}
        picture_dict["labels"]["polygons"] = []
        picture_dict["labels"]["masks"] = []
        picture_dict["labels"]["dimensions"] = {"width": image_size[0], "height": image_size[1]}
        boxes = []
        for col in range(result.pandas().xyxy[0].shape[0]):
            boxes.append(
                {"pt1": {"x": result.pandas().xyxy[0].iloc[col]['xmin'] / picture_dict["labels"]["dimensions"]["width"],
                         "y": result.pandas().xyxy[0].iloc[col]['ymin'] / picture_dict["labels"]["dimensions"][
                             "height"]},
                 "labels": {"Type": [result.pandas().xyxy[0].iloc[col]['name']]},
                 "pt2": {"x": result.pandas().xyxy[0].iloc[col]['xmax'] / picture_dict["labels"]["dimensions"]["width"],
                         "y": result.pandas().xyxy[0].iloc[col]['ymax'] / picture_dict["labels"]["dimensions"][
                             "height"]}})
            regions.append(result.pandas().xyxy[0].iloc[col]['name'])
        picture_dict["labels"]["boxes"] = boxes
        picture_dict["ignore"] = False
        result_dict["items"].append(picture_dict)
    options = []
    for region in set(regions):
        options.append({"name": region})
    result_dict["config"] = {"regions": [{"name": "Type", "multiple": False, "options": options}]}
    result_dict["maxCanvasSize"] = 512
    result_dict["maxViewHeight"] = 512
    result_dict["mode"] = "light"
    result_dict["batchSize"] = 1
    result_dict["allowConfigChange"] = True
    result_dict["advanceOnSave"] = True
    return result_dict


def post_processing(labels_dir):
    """
        Создает словарь аннотаций в формате COCO.
        Args:
            labels_dir (str): путь к каталогу, в котором хранится json файл, содержащий разметку.
    """
    if not os.path.isfile(labels_dir + "/labels.json"):
        raise FileNotFoundError('Error: no labels file found in', labels_dir + "/labels.json", 'directory')
    with open(labels_dir + "/labels.json", "r") as file:
        labels = json.load(file)

    anno_dict = dict()
    anno_dict["info"] = {"description": "", "url": "", "version": "", "year": datetime.now().year,
                         "contributor": "", "date_created": date.today().strftime("%B %d, %Y")}
    anno_dict["licenses"] = {"url": "", "id": 1, "name": ""}
    anno_dict["images"] = []
    anno_dict["annotations"] = []
    img_id_index = 0
    cat_dict = {}
    cat_index = 0
    for bb_name in labels["config"]["regions"][0]["options"]:
        cat_dict[bb_name["name"]] = cat_index
        cat_index = cat_index + 1
    for image in labels["items"]:
        image_dict = {}
        name = image["target"]
        name = name[name.rfind("/") + 1:]
        image_dict["license"] = ""
        image_dict["file_name"] = name
        image_dict["coco_url"] = ""
        image_dict["height"] = image["labels"]["dimensions"]["height"]
        image_dict["width"] = image["labels"]["dimensions"]["width"]
        image_dict["date_captured"] = str(datetime.now())
        image_dict["flickr_url"] = ""
        image_dict["id"] = img_id_index
        anno_dict["images"].append(image_dict)
        img_id_index = img_id_index + 1
        for box in image["labels"]["boxes"]:
            annotation_dict = {}
            annotation_dict["segmentation"] = []
            annotation_dict["num_keypoints"] = 0
            annotation_dict["area"] = 0
            annotation_dict["iscrowd"] = 0
            annotation_dict["keypoints"] = []
            annotation_dict["image_id"] = image_dict["id"]
            annotation_dict["bbox"] = [box["pt1"]["x"] * image_dict["width"], box["pt1"]["y"] * image_dict["height"],
                                       box["pt2"]["x"] * image_dict["width"], box["pt2"]["y"] * image_dict["height"]]
            annotation_dict["category_id"] = cat_dict[box["labels"]["Type"][0]]
            annotation_dict["id"] = len(anno_dict["annotations"])
            anno_dict["annotations"].append(annotation_dict)

    anno_dict["categories"] = []
    for cat in cat_dict.keys():
        anno_dict["categories"].append({"supercategory": "", "id": cat_dict[cat], "name": cat})
    return anno_dict


def upload_from_folder(image_dir, dst_dir):
    """
        Добавляет в каталог, находящийся по пути dst_dir все изобрвжения, находящиеся в пользовательском каталоге.
        Args:
            dst_dir (str): путь к каталогу, в котором будут храниться изображения
    """
    from os.path import isfile, join
    import shutil
    # print("Введите путь директории с изображениями")
    if len(os.listdir(image_dir)) == 0:
        raise FileNotFoundError('No image files in', image_dir, 'directory')
    for filename in os.listdir(image_dir):
        if isfile(join(image_dir, filename)):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                jpgfile = join(image_dir, filename)
                shutil.copy(jpgfile, dst_dir)


def upload_from_server(dst_dir):
    """
    Добавляет в каталог, находящийся по пути dst_dir изобрвжения,скаченные с указанного ресурса.
    Args:
        dst_dir (str): путь к каталогу, в котором будут храниться изображения

    TODO:
        дописать
    """
    import requests
    list_of_names = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
    print("Введите путь до файла с url")
    url_file = input()
    with open(url_file) as file:
        i = 0
        for line in file:
            print(i)
            if line == '':
                break
            else:
                img_data = requests.get(line).content
                print(dst_dir + '/' + list_of_names[i])
                with open(dst_dir + '/' + list_of_names[i], 'wb') as handler:
                    handler.write(img_data)
            i = i + 1


def upload_from_zip(image_dir, dst_dir):
    """
    Добавляет в каталог, находящийся по пути dst_dir изобрвжения, находящиеся в zip-архиве.
    Args:
        dst_dir (str): путь к каталогу, в котором будут храниться изображения
    """
    import shutil
    if not os.path.isfile(image_dir):
        raise FileNotFoundError('Error: no annotation file', image_dir, 'found')
    shutil.unpack_archive(image_dir, dst_dir)