import ann_automl.core.db_module as db
from ann_automl.utils.text_utils import print_progress_bar

# make temporary directory for test database files
import tempfile
import os
import shutil
import pytest
import requests
import xmltodict
import json

IMAGENET_API_WNID = 'http://www.image-net.org/api/imagenet.synset.geturls?wnid='


def check_coco_images(anno_file, image_dir):
    if os.path.exists(image_dir):
        return
    print(f'Downloading COCO images for test (annotations = {anno_file})')
    from pycocotools.coco import COCO

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
    # print_progress_bar(0, len(img_ids), prefix='Loading images:', suffix='Complete', length=50)
    for i, img_id in enumerate(img_ids):
        img = coco.loadImgs([img_id])[0]
        img_data = requests.get(img['coco_url']).content
        with open(image_dir + '/' + img['file_name'], 'wb') as handler:
            handler.write(img_data)
        # print_progress_bar(i, len(img_ids), prefix='Loading images:', suffix='Complete', length=50)


def check_imagenet_images(anno_dir, image_dir):
    if os.path.exists(image_dir):
        return
    os.makedirs(image_dir, exist_ok=True)
    for dirs in os.listdir(anno_dir):
        os.makedirs(image_dir + '/' + dirs, exist_ok=True)
        response = requests.get(IMAGENET_API_WNID + dirs).content.decode('utf8')
        urls = [url for url in response.splitlines()]
        count = len(os.listdir(anno_dir + dirs))
        i = 1
        for url in urls:
            if i > count:
                break
            try:
                img_data = requests.get(url)
            except Exception:
                continue
            if img_data.status_code != 200 or img_data.encoding:
                continue
            with open(anno_dir + dirs + '/' + dirs + '_' + str(i) + '.xml') as file:
                xml_data = xmltodict.parse(file.read())
                with open(image_dir + '/' + dirs + '/' + xml_data['annotation']['filename'] + '.JPEG', 'wb') as handler:
                    handler.write(img_data.content)
                    i += 1


def download_cat_images(cat_name, anno_dir, image_dir):
    with open(anno_dir + 'annotations1.json') as f:
        d = json.load(f)
        img_names = []
        for img in d['images']:
            if img['file_name'].split('.')[0] == cat_name:
                img_names.append(img['file_name'])
        with open(anno_dir + cat_name + 's_urls.txt') as urls:
            img_num = 0
            for url in urls:
                if url[-1:] == '\n':
                    url = url[:-1]
                try:
                    img_data = requests.get(url)
                except Exception:
                    continue
                if img_data.status_code != 200 or img_data.encoding:
                    continue
                with open(image_dir + '/' + img_names[img_num], 'wb') as handler:
                    handler.write(img_data.content)
                img_num += 1
        print(d)


def check_cats_vs_dogs_images(anno_dir, image_dir_1, image_dir_2):
    if os.path.exists(image_dir_1) and os.path.exists(image_dir_2):
        return
    elif os.path.exists(image_dir_1) and not os.path.exists(image_dir_2):
        os.makedirs(image_dir_2, exist_ok=True)
        download_cat_images(image_dir_2[-4:-1], anno_dir, image_dir_2)
    elif not os.path.exists(image_dir_1) and os.path.exists(image_dir_2):
        os.makedirs(image_dir_1, exist_ok=True)
        download_cat_images(image_dir_1[-4:-1], anno_dir, image_dir_1)
    else:
        os.makedirs(image_dir_1, exist_ok=True)
        download_cat_images(image_dir_1[-4:-1], anno_dir, image_dir_1)
        os.makedirs(image_dir_2, exist_ok=True)
        download_cat_images(image_dir_2[-4:-1], anno_dir, image_dir_2)


# At first run download images from coco dataset
check_coco_images('datasets/test1/annotations/annotations1.json', 'datasets/test1/images')
check_coco_images('datasets/test2/annotations/train.json', 'datasets/test2/images')
check_imagenet_images('datasets/test_imagenet/annotations/', 'datasets/test_imagenet/images')
check_cats_vs_dogs_images('datasets/test3/annotations/', 'datasets/test3/images/cats', 'datasets/test3/images/dogs')


@pytest.fixture(scope='function')
def db_dir():
    db_dir = tempfile.mkdtemp()
    yield db_dir
    shutil.rmtree(db_dir)


# Test database creation in temporary directory
@pytest.mark.usefixtures('db_dir')
def test_double_fill_coco(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')

    mydb.fill_coco('datasets/test1/annotations/annotations1.json', file_prefix='datasets/test1/images')
    mydb.close()

    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert set(mydb.get_all_categories()['name']) == {'bicycle', 'airplane'}

    mydb.fill_coco('datasets/test2/annotations/train.json', file_prefix='datasets/test2/images')
    assert len(mydb.get_all_datasets()) == 2
    # print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert set(mydb.get_all_categories()['name']) == {'bicycle', 'airplane', 'cat', 'dog'}

    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_simple_fill_coco(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_coco('datasets/test2/annotations/train.json', file_prefix='datasets/test2/images')
    assert len(mydb.get_all_datasets()) == 1
    print(f'Categories: {list(mydb.get_all_categories()["name"])}')
    assert set(mydb.get_all_categories()['name']) == {'cat', 'dog'}
    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_fill_kaggle_cats_vs_dogs(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_kaggle_cats_vs_dogs('datasets/test3/annotations/annotations1.json', file_prefix='datasets/test3/images/')
    assert len(mydb.get_all_datasets()) == 1
    assert len(mydb.get_all_categories()) == 2
    print('all data sets info : ', mydb.get_all_datasets_info(full_info=True))
    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_fill_imagenet(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_imagenet('datasets/test_imagenet/annotations', file_prefix='datasets/test_imagenet/images/',
                       assoc_file='datasets/test_imagenet/imageNetToCOCOClasses.txt', first_time=True)
    assert len(mydb.get_all_datasets()) == 1
    first_cat = list(mydb.get_all_categories()['name'])
    mydb.fill_imagenet('datasets/test_imagenet/annotations', file_prefix='datasets/test_imagenet/images/',
                       assoc_file='datasets/test_imagenet/imageNetToCOCOClasses.txt', first_time=False)
    second_cat = list(mydb.get_all_categories()['name'])
    assert first_cat == second_cat
    # assert len(mydb.get_all_datasets()) == 2
    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_check_info_methods(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_coco('datasets/test2/annotations/train.json', file_prefix='datasets/test2/images')
    assert len(mydb.get_all_datasets()) == 1
    # print('Datasets : ', mydb.get_all_datasets())
    # print('Categories info : ', mydb.get_dataset_categories_info(1))
    check_dict = make_inf_check_dict_from_json('datasets/test2/annotations/train.json')
    assert mydb.get_dataset_categories_info(1) == check_dict

    mydb.fill_coco('datasets/test1/annotations/annotations1.json', file_prefix='datasets/test1/images')
    assert len(mydb.get_all_datasets()) == 2
    # print('Datasets : ', mydb.get_all_datasets())
    # print('Categories info : ', mydb.get_dataset_categories_info(2))
    check_dict2 = make_inf_check_dict_from_json('datasets/test1/annotations/annotations1.json')
    assert mydb.get_dataset_categories_info(2) == check_dict2

    # print('Dataset info : ', mydb.get_all_datasets_info())
    assert mydb.load_specific_images_annotations(['datasets/test2/images/000000073729.jpg',
                                                  'datasets/test2/images/000000098304.jpg']).shape[0] == 2

    cat_data_dict = {1: ['cat', 'dog'], 2: ['airplane', 'bicycle']}
    assert list(mydb.load_specific_categories_from_specific_datasets_annotations(cat_data_dict)[0]['target']) == list(
        mydb.load_categories_datasets_annotations(['cat', 'dog', 'airplane', 'bicycle'], [1, 2])[0]['target'])

    print(mydb.get_full_dataset_info(1))
    assert mydb.get_full_dataset_info(1)['ID'] == 1
    assert mydb.get_full_dataset_info(1)['categories'] == get_all_cats_from_check_dict(check_dict)
    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_ids_and_names(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_coco('datasets/test2/annotations/train.json', file_prefix='datasets/test2/images')
    ids = mydb.get_cat_IDs_by_names(["cat", "dog", "elephant"])
    assert ids == [17, 18, -1]
    cat_names = mydb.get_cat_names_by_IDs(ids)
    assert cat_names == ['cat', 'dog', '']

    mydb2 = db.DBModule(f"sqlite:///{db_dir}/test2.sqlite")
    mydb2.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test2.sqlite')
    mydb2.fill_coco('datasets/test1/annotations/annotations1.json', file_prefix='datasets/test1/images')
    ids2 = mydb2.get_cat_IDs_by_names(["bicycle", "airplane", "cat", "dog"])
    assert ids2 == [2, 5, -1, -1]
    cat_names = mydb2.get_cat_names_by_IDs(ids2)
    assert cat_names == ['bicycle', 'airplane', '', '']

    mydb.fill_coco('datasets/test1/annotations/annotations1.json', file_prefix='datasets/test1/images')
    ids2 = mydb.get_cat_IDs_by_names(["bicycle", "airplane", "cat", "dog"])
    assert ids2 == [2, 5, 17, 18]
    cat_names = mydb.get_cat_names_by_IDs(ids2)
    assert cat_names == ['bicycle', 'airplane', 'cat', 'dog']

    mydb.close()
    mydb2.close()


@pytest.mark.usefixtures('db_dir')
def test_get_models_by_filter(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_coco('datasets/test2/annotations/train.json', file_prefix='datasets/test2/images')
    assert len(mydb.get_all_datasets()) == 1
    assert list(mydb.get_all_categories()['name']) == ['cat', 'dog']
    mydb.add_model_record(task_type="classification", categories=['cat', 'dog'],
                          model_address='data/trained_NN/First_model', metrics={'accuracy': 0.625},
                          history_address='data/trained_NN/First_model')
    mydb.add_model_record(task_type="classification", categories=['cat', 'dog'],
                          model_address='data/trained_NN/Second_model', metrics={'accuracy': 0.58},
                          history_address='data/trained_NN/Second_model')
    assert mydb.get_models_by_filter({'min_metrics': {'accuracy': 0.58}, 'categories': ['cat', 'dog']}).shape[0] == 2
    assert mydb.get_models_by_filter({'min_metrics': {'accuracy': 0.59}, 'categories': ['cat', 'dog']}).shape[0] == 1
    assert mydb.get_models_by_filter({'min_metrics': {'accuracy': 0.65}, 'categories': ['cat', 'dog']}).shape[0] == 0
    assert mydb.get_models_by_filter({'min_metrics': {'accuracy': 0.58}, 'categories': ['cat', 'mouse']}).shape[0] == 0

    mydb.update_train_result_record(model_address='data/trained_NN/Second_model', metric_name='accuracy',
                                    metric_value=0.6, history_address='')
    assert mydb.get_models_by_filter({'min_metrics': {'accuracy': 0.59}, 'categories': ['cat', 'dog']}).shape[0] == 2

    mydb.delete_train_result_record(model_address='data/trained_NN/Second_model', metric_name='accuracy')

    assert mydb.get_models_by_filter({'min_metrics': {'accuracy': 0.59}, 'categories': ['cat', 'dog']}).shape[0] == 1

    mydb.close()


@pytest.mark.usefixtures('db_dir')
def test_load_categories_annotations(db_dir):
    mydb = db.DBModule(f"sqlite:///{db_dir}/test.sqlite")
    mydb.create_sqlite_file()
    assert os.path.isfile(db_dir + '/test.sqlite')
    mydb.fill_coco('datasets/test1/annotations/annotations1.json', file_prefix='datasets/test1/images')
    assert len(mydb.get_all_datasets()) == 1
    mydb.load_specific_datasets_annotations([1])
    check_pd1 = mydb.load_specific_categories_annotations(['bicycle', 'airplane'])[0]
    print('\n\n\n\n\n\n', mydb.load_specific_categories_annotations(['bicycle', 'airplane'], normalize_cats=False))
    check_pd2 = mydb.load_specific_categories_annotations(['bicycle', 'airplane'], normalize_cats=True,
                                                          split_points=[0.7, 0.85])[0]
    # check_pd1 = mydb.load_specific_categories_annotations(['bicycle', 'airplane'])[0]
    # check_pd2 = mydb.load_specific_categories_annotations(['bicycle', 'airplane'], normalize_cats = True,
    #                                                       split_points = [0.7, 0.85])[0]
    assert check_pd1.shape[0] == check_pd2.shape[0]
    print()
    assoc_dict = create_assoc_dict(check_pd1)
    for row in check_pd1.index:
        assert check_pd1['images'][row] == check_pd2['images'][row]
        assert check_pd2['target'][row] == assoc_dict[check_pd1['target'][row]]

    mydb.close()


if __name__ == '__main__':
    if os.path.isdir('tmp/db'):
        shutil.rmtree('tmp/db')
    os.makedirs('tmp/db', exist_ok=True)
    test_double_fill_coco('tmp/db')


def make_inf_check_dict_from_json(file_path):
    import json
    with open(file_path, 'r') as fp:
        check = json.load(fp)
    cat_id_counter = {}
    for img in check["annotations"]:
        if img["category_id"] not in cat_id_counter:
            cat_id_counter[img["category_id"]] = 1
        else:
            cat_id_counter[img["category_id"]] += 1
    check_dict = {}
    for cat in check["categories"]:
        if cat['supercategory'] not in check_dict:
            check_dict[cat['supercategory']] = {}
        check_dict[cat['supercategory']][cat['name']] = cat_id_counter[cat['id']]
    return check_dict


def get_all_cats_from_check_dict(dict):
    cats = {}
    for supercat in dict.keys():
        for cat in dict[supercat]:
            cats[cat] = dict[supercat][cat]
    return cats


def create_assoc_dict(df):
    assoc_dict = {}
    k = 0
    for row in df.index:
        if df['target'][row] not in assoc_dict:
            assoc_dict[df['target'][row]] = k
            k += 1
    return assoc_dict
