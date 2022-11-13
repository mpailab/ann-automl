from sqlalchemy import *
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
import pandas as pd
from sqlalchemy.orm import sessionmaker
import json
import ast
import cv2
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import os
import math
from PIL import Image
import glob
import xml.etree.ElementTree as ET
import time

Base = declarative_base()


class dbModule:

    ############################################################
    ##########              Helpers          ###################
    ############################################################
    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Helper to display progress bar, source from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    ############################################################
    ##########        DB ORM description     ###################
    ############################################################

    class Image(Base):
        __tablename__ = "image"
        ID = Column(Integer, primary_key=True)
        dataset_id = Column(Integer, ForeignKey("dataset.ID"))
        license_id = Column(Integer, ForeignKey("license.ID"))
        file_name = Column(String)
        coco_url = Column(String)
        height = Column(Integer)
        width = Column(Integer)
        date_captured = Column(String)
        flickr_url = Column(String)
        aux = Column(String)

        def __init__(self, file_name, width, height, date_captured, dataset_id, coco_url='', flickr_url='',
                     license_id=-1, _id=None, aux=''):
            self.width = width
            self.height = height
            self.file_name = file_name
            self.date_captured = date_captured
            self.dataset_id = dataset_id
            self.coco_url = coco_url
            self.flickr_url = flickr_url
            if license_id == -1:
                license_id = 1
            self.license_id = license_id
            if _id is not None:
                self.ID = _id
            self.aux = aux

    class Dataset(Base):
        __tablename__ = "dataset"
        ID = Column(Integer, primary_key=True)
        description = Column(String)
        url = Column(String)
        version = Column(String)
        year = Column(Integer)
        contributor = Column(String)
        date_created = Column(String)
        aux = Column(String)
        images = relationship("Image", backref=backref("dataset"))

        def __init__(self, description, url, version, year, contributor, date_created, _id=None, aux=''):
            self.description = description
            self.url = url
            self.version = version
            self.year = year
            self.contributor = contributor
            self.date_created = date_created
            if _id is not None:
                self.ID = _id
            self.aux = aux

    class Category(Base):
        __tablename__ = "category"
        ID = Column(Integer, primary_key=True)
        supercategory = Column(String)
        name = Column(String)
        aux = Column(String)
        images = relationship("Annotation", backref=backref("category"))
        records = relationship("CategoryToModel")

        def __init__(self, supercategory, name, _id=None, aux=''):
            self.supercategory = supercategory
            self.name = name
            if _id is not None:
                self.ID = _id
            self.aux = aux

    class License(Base):
        __tablename__ = "license"
        ID = Column(Integer, primary_key=True)
        name = Column(String)
        url = Column(String)
        aux = Column(String)
        images = relationship("Image")

        def __init__(self, name, url, _id=None, aux=''):
            self.url = url
            self.name = name
            if _id is not None:
                self.ID = _id
            self.aux = aux

    class Annotation(Base):
        __tablename__ = "annotation"
        ID = Column(Integer, primary_key=True)
        image_id = Column(Integer, ForeignKey("image.ID"))
        category_id = Column(Integer, ForeignKey("category.ID"))
        bbox = Column(String)
        segmentation = Column(String)
        is_crowd = Column(Integer)
        area = Column(Float)
        aux = Column(String)

        def __init__(self, image_id, category_id, bbox, segmentation, is_crowd, area, _id=None, aux=''):
            self.image_id = image_id
            self.category_id = category_id
            self.bbox = bbox
            self.segmentation = segmentation
            self.is_crowd = is_crowd
            self.area = area
            if _id is not None:
                self.ID = _id
            self.aux = aux

    class TrainResult(Base):
        __tablename__ = "trainResult"
        ID = Column(Integer, primary_key=True)
        metric_name = Column(String)
        metric_value = Column(Float)
        model_id = Column(Integer, ForeignKey("model.ID"))
        history_address = Column(String)
        aux = Column(String)

        def __init__(self, metric_name, metric_value, model_id, history_address='', aux='', _id=None):
            self.metric_name = metric_name
            self.metric_value = metric_value
            self.history_address = history_address
            self.model_id = model_id
            if _id is not None:
                self.ID = _id
            self.aux = aux

    class CategoryToModel(Base):
        __tablename__ = "categoryToModel"
        category_id = Column(Integer, ForeignKey(
            "category.ID"), primary_key=True)
        model_id = Column(Integer, ForeignKey("model.ID"), primary_key=True)

        def __init__(self, category_id, model_id):
            self.category_id = category_id
            self.model_id = model_id

    class Model(Base):
        __tablename__ = "model"
        ID = Column(Integer, primary_key=True)
        model_address = Column(String)
        task_type = Column(String)
        aux = Column(String)
        train_results = relationship("TrainResult", backref=backref("model"))
        categories = relationship("CategoryToModel")

        def __init__(self, model_address, task_type, aux='', _id=None):
            self.model_address = model_address
            self.task_type = task_type
            if _id is not None:
                self.ID = _id
            self.aux = aux

    ############################################################
    ##########        DB Module methods      ###################
    ############################################################

    def __init__(self, dbstring='sqlite:///datasets.sqlite', dbecho=False):
        """
        Basic initialization method, creates session to the DB address given by dbstring (defult sqlite:///datasets.sqlite).
        IMPORTANT: SQLLite file is stored inside project working directory
        """
        if os.path.isfile('dbconfig.txt'):  # if config file exists we take all paths from there
            with open('dbconfig.txt') as f:
                dbconfig = json.load(f)
                if dbconfig.get('dbstring', False):
                    dbstring = dbconfig['dbstring']
                if dbconfig.get('KaggleCatsVsDogs', False):
                    self.KaggleCatsVsDogsConfig_ = dbconfig['KaggleCatsVsDogs']
                if dbconfig.get('COCO2017', False):
                    self.COCO2017Config_ = dbconfig['COCO2017']
                if dbconfig.get('ImageNet', False):
                    self.ImageNetConfig_ = dbconfig['ImageNet']
        self.engine = create_engine(dbstring, echo=dbecho)
        Session = sessionmaker(bind=self.engine)
        self.sess = Session()
        self.dbstring_ = dbstring

    def create_sqlite_file(self):
        """
        In case SQLite file not found one should call this method to create one.
        """
        Base.metadata.create_all(self.engine)

    def fill_all_default(self, annoFileName_='./datasets/coco/annotations/instances_train2017.json'):
        """
        Method to fill all at once, supposing datasets are at default locations (CatsDogs, COCO, ImageNet)
        annoFileName_ -> shows path to the default COCO2017 annotations (train subset as default)
        """
        if os.path.exists(self.dbstring_.split('/')[-1]):  # If file exists we suppose it is filled
            return
        self.create_sqlite_file()
        self.fill_coco(annoFileName=annoFileName_, firstTime=True)
        self.fill_Kaggle_CatsVsDogs()
        self.fill_imagenet(first_time=True)
        return

    def fill_Kaggle_CatsVsDogs(self, annoFileName='dogs_vs_cats_coco_anno.json', file_prefix='./datasets/Kaggle/'):
        """Method to fill Kaggle CatsVsDogs dataset into db. It is supposed to be called once.
        INPUT:
            annoFileName - file with json annotation in COCO format for cats and dogs
        OUTPUT:
            None
        """
        #print('Continue with catsVsDogs?')
        # input()
        if hasattr(self, 'KaggleCatsVsDogsConfig_'):  # to init from config file
            annoFileName = self.KaggleCatsVsDogsConfig_['anno_filename']
            file_prefix = self.KaggleCatsVsDogsConfig_['file_prefix']

        print('Start filling DB with Kaggle CatsVsDogs')
        if not os.path.isfile(annoFileName):
            print('Error: no file', annoFileName, 'found')
            print('Stop filling dataset')
            return
        if not os.path.isdir(file_prefix):
            print('Error: no directory', file_prefix, 'found')
            print('Stop filling dataset')
            return
        with open(annoFileName) as json_file:
            data = json.load(json_file)
        if not os.path.isfile(file_prefix + data['images'][0]['file_name'].split('.')[0] + 's/' + data['images'][0]['file_name']):
            print('Error in json file, missing images stored on disc (i.e.',
                  file_prefix + data['images'][0]['file_name'].split('.')[0] + 's/' + data['images'][0]['file_name'], ')')
            print('Stop filling dataset')
            return
        dataset_info = data['info']
        dataset = self.Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'],
                               dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit()  # adding dataset
        ###################################
        im_objects = {}
        self.printProgressBar(
            0, len(data['images']), prefix='Adding images:', suffix='Complete', length=50)
        im_counter = 0
        for im_data in data['images']:
            image = self.Image(file_prefix + im_data['file_name'].split('.')[0] + 's/' + im_data['file_name'], im_data['width'], im_data['height'],
                               im_data['date_captured'], dataset.ID, im_data['coco_url'], im_data['flickr_url'],
                               im_data['license'])
            if im_counter % 10 == 0 or im_counter == len(data['images']) - 2:
                self.printProgressBar(im_counter, len(
                    data['images']), prefix='Adding images:', suffix='Complete', length=50)
            im_counter += 1
            im_objects[im_data['id']] = image
            self.sess.add(image)
        self.sess.commit()  # adding images
        ###################################

        self.printProgressBar(0, len(
            data['annotations']), prefix='Adding annotations:', suffix='Complete', length=50)
        an_counter = 0
        for an_data in data['annotations']:
            # +1 because of json file structure for this DB only
            real_id = im_objects[an_data['image_id']].ID
            annotation = self.Annotation(real_id, an_data['category_id'] + 1, ';'.join(an_data['bbox']),
                                         ';'.join(an_data['segmentation']), an_data['iscrowd'], an_data['area'])
            if an_counter % 10 == 0 or an_counter == len(data['annotations']) - 2:
                self.printProgressBar(an_counter, len(
                    data['annotations']), prefix='Adding annotations:', suffix='Complete', length=50)
            an_counter += 1
            self.sess.add(annotation)
        self.sess.commit()  # adding annotations
        print('Finished with Kaggle CatsVsDogs')

    def fill_coco(self, annoFileName, file_prefix='./datasets/COCO2017/', firstTime=False, ds_info=None):
        """Method to fill COCOdataset into db. It is supposed to be called once.
        To create custom COCO annotations use some aux tools like https://github.com/jsbroks/coco-annotator
        INPUT:
            annoFileName - file with json annotation in COCO format for cats and dogs
            ds_info - dictionary with info about dataset (default - COCO2017). Necessary keys:
                description, url, version, year, contributor, date_created
        OUTPUT:
            None
        """
        if hasattr(self, 'COCO2017Config_'):  # to init from config file
            annoFileName = self.COCO2017Config_['anno_filename']
            file_prefix = self.COCO2017Config_['file_prefix']

        print('Start filling DB with COCO-format dataset from file', annoFileName)
        if not os.path.isfile(annoFileName):
            print('Error: no file', annoFileName, 'found')
            print('Stop filling dataset')
            return
        if not os.path.isdir(file_prefix):
            print('Error: no directory', file_prefix, 'found')
            print('Stop filling dataset')
            return
        coco = COCO(annoFileName)
        cats = coco.loadCats(coco.getCatIds())
        # CALL THE FOLLOWING TWO METHODS ONLY WHEN NEEDED - WE MAKE A CHECK - USER IS RESPONSIBLE
        if firstTime:
            self.add_categories(cats, True)
            self.add_default_licences()
        #######################################################################
        if ds_info is None:
            ds_info = {"description": "COCO 2017 Dataset",
                       "url": "http://cocodataset.org",
                       "version": "1.0", "year": 2017,
                       "contributor": "COCO Consortium",
                       "date_created": "2017/09/01"}
        dsID = self.add_dataset_info(ds_info)
        imgIds = coco.getImgIds()
        imgs = coco.loadImgs(imgIds)
        annIds = coco.getAnnIds()
        anns = coco.loadAnns(annIds)
        print(f'Dataset description: ', ds_info["description"])
        print(f'Adding {len(anns)} annotations in COCO format to DB')
        self.add_images_and_annotations(imgs, anns, dsID, file_prefix)
        return

    def fill_imagenet(self, annotations_dir='./datasets/imagenet/annotations',
                      file_prefix='./datasets/imagenet/ILSVRC2012_img_train',
                      assoc_file='imageNetToCOCOClasses.txt', first_time=False,
                      ds_info=None):
        """Method to fill ImageNet dataset into db. It is supposed to be called once.
            INPUT:
                annotations_dir - path to annotations directories
                file_prefix - prefix of images from ImageNet
                assoc_file - filename of ImageNet to COCO associations
                first_time - bool, put to true if called for the first time
                ds_info - some information about dataset
            OUTPUT:
                None
        """
        if hasattr(self, 'ImageNetConfig_'):  # to init from config file
            annotations_dir = self.ImageNetConfig_['annotations_dir']
            file_prefix = self.ImageNetConfig_['file_prefix']
            assoc_file = self.ImageNetConfig_['categories_assoc_file']

        if not os.path.isdir(annotations_dir):
            print('Error: no directory', annotations_dir, 'found')
            print('Stop filling ImageNet dataset')
            return
        if not os.path.isdir(file_prefix):
            print('Error: no directory', file_prefix, 'found')
            print('Stop filling ImageNet dataset')
            return
        # If we make it for the first time, we add dataset information to DB
        if first_time:
            if ds_info is None:
                ds_info = {"description": "ImageNet 2012 Dataset", "url": "https://image-net.org/about.php",
                           "version": "1.0", "year": 2012, "contributor": "ImageNet", "date_created": "2012/01/01"}
            dsID = self.add_dataset_info(ds_info)
        else:
            dsID = 3  # this is just a patch for ImageNet, since in 'clear' dataset imageNet is filled last
        # Then we take all the associations from the assoc_file and create some categories if needed
        assoc = {}
        with open(assoc_file) as file:
            lines = file.readlines()
            for line in lines:
                arr = line.split(';')
                assoc[arr[0]] = [arr[1], arr[2].rstrip('\n')]
        cat_names = set()
        categories_buf_assoc = {}
        for elem in assoc:
            if assoc[elem][1] != 'None':
                cat_names.add(assoc[elem][1])
                categories_buf_assoc[elem] = assoc[elem][1]
            else:
                cat_names.add(assoc[elem][0])
                categories_buf_assoc[elem] = assoc[elem][0]
        cat_names_list = list(cat_names)
        catIDs = self.get_cat_IDs_by_names(cat_names_list)
        new_categories = []
        for i in range(len(catIDs)):
            cat_id = catIDs[i]
            if cat_id < 0:
                cat_name = cat_names_list[i]
                new_categories.append(
                    {'supercategory': 'imageNetOther', 'name': cat_name})
        if len(new_categories) > 0:
            self.add_categories(new_categories, respect_ids=False)
            catIDs = self.get_cat_IDs_by_names(cat_names_list)
            for i in range(len(catIDs)):
                assert catIDs[i] > 0, 'Some category could not be added'
        categories_assoc = {}
        for i in range(len(catIDs)):
            cat_id = catIDs[i]
            cat_name = cat_names_list[i]
            categories_assoc[cat_name] = cat_id
        # Then we iterate over all the images from dataset.
        # For each image we check if there are bbox in XML files.
        # If so - this information is added to annotation.
        # ImageNet file structure requires /*/*.JPEG
        img_files = glob.glob(file_prefix + '/*/*.JPEG', recursive=True)
        print('Len img_files:', len(img_files))
        images = []
        annotations = []
        licence_id = 1  # default value
        im_id = 0
        self.printProgressBar(
            0, len(img_files), prefix='Processing ImageNet XML files:', suffix='Complete', length=50)
        for img_file in img_files:
            im = Image.open(img_file)
            width, height = im.size
            # creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') #previous version
            creation_time = time.ctime(os.path.getctime(img_file))
            image_data = {'file_name': img_file, 'width': width, 'height': height, 'date_captured': creation_time,
                          'coco_url': '', 'flickr_url': '', 'license': licence_id,
                          'id': im_id}  # id will not be respected when added to base - needed for annotations only
            images.append(image_data)
            img_name_no_ext = os.path.splitext(os.path.basename(img_file))[0]
            anno_subdir = img_name_no_ext.split('_')[0]
            annofilename = os.path.join(
                annotations_dir, anno_subdir, img_name_no_ext + '.xml')
            if os.path.isfile(annofilename):
                # corresponding annotation file is found
                tree = ET.parse(annofilename)
                root = tree.getroot()
                obj_tag = root.find('object')
                if obj_tag is not None:
                    for child in obj_tag:
                        bbox = []
                        if child.tag == 'bndbox':
                            for bchild in child:
                                bbox.append(bchild.text)
                            area = np.abs(
                                (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
                            annotation_data = {'image_id': im_id, 'segmentation': '', 'bbox': json.dumps(bbox),
                                               'iscrowd': 0, 'area': area,
                                               'category_id': categories_assoc[categories_buf_assoc[anno_subdir]]}
                            annotations.append(annotation_data)
            else:
                annotation_data = {'image_id': im_id, 'segmentation': '', 'bbox': '', 'iscrowd': 0,
                                   'area': width * height,
                                   'category_id': categories_assoc[categories_buf_assoc[anno_subdir]]}
                annotations.append(annotation_data)
            im_id += 1
            if im_id % 10 == 0 or im_id == len(img_files) - 1:
                self.printProgressBar(
                    im_id, len(img_files), prefix='Processing ImageNet XML files:', suffix='Complete', length=50)
        self.add_images_and_annotations(images, annotations, dsID)
        return images, annotations

    def get_all_datasets(self, full_info=False):
        # SQL file is stored inside project working directory
        if not os.path.exists(self.dbstring_.split('/')[-1]):
            self.create_sqlite_file()
        query = self.sess.query(self.Dataset)
        df = pd.read_sql(query.statement, query.session.bind)
        df_rec = df.to_dict('record')
        df_dict = {}
        for el in df_rec:
            if not full_info:
                df_dict[el['ID']] = el
                del df_dict[el['ID']]['ID']
            else:
                df_dict[el['ID']] = self.get_full_dataset_info(el['ID'])
        return df_dict

    def load_specific_datasets_annotations(self, datasets_ids, **kwargs):
        """Method to load annotations from specific datasets, given their IDs.
        INPUT:
            datasets_ids - list of datasets IDs to get annotations from
            kwargs["normalizeCats"] -> used for test purposes only, changes real categories to count from 0 (i.e. cats,dogs(17,18) -> (0,1))
        OUTPUT:
            pandas dataframe with annotations for given datasets IDs
        """
        query = self.sess.query(self.Image.file_name, self.Annotation.category_id, self.Annotation.bbox,
                                self.Annotation.segmentation).join(self.Annotation).filter(self.Image.dataset_id.in_(datasets_ids))
        df = pd.read_sql(query.statement, query.session.bind)
        # a fancy patch for keras to start numbers from 0
        if 'normalizeCats' in kwargs and kwargs['normalizeCats'] is True:
            df_new = pd.DataFrame(columns=['images', 'target'], data=df[[
                                  'file_name', 'category_id']].values)
            min_cat = df_new['target'].min()
            df_new['target'] = df_new['target'] - min_cat
            return df_new
        return df

    def get_all_categories(self):
        """
        Returns pandas dataframe with all available categories in database
        """
        query = self.sess.query(self.Category)
        df = pd.read_sql(query.statement, query.session.bind)
        return df

    def _prepare_cropped_images(self, df, kwargs):
        """
        Helper for image cropping in case of multiple annotations on one picture.
        """
        column_names = ["file_name", "category_id"]
        buf_df = pd.DataFrame(columns=column_names)
        # print(buf_df)
        # print(df)
        cropped_dir = kwargs.get('cropped_dir', '') or 'buf_crops/'
        Path(cropped_dir).mkdir(parents=True, exist_ok=True)
        files_dir = kwargs.get('files_dir', '')
        for index, row in df.iterrows():
            bbox = []
            if row['bbox'] != '':
                bbox = ast.literal_eval(row['bbox'])
            if len(bbox) != 4:
                new_row = {'file_name': row['file_name'],
                           'category_id': row['category_id']}
                buf_df = buf_df.append(
                    new_row, ignore_index=True)  # nothing to cut
                continue
            image = cv2.imread(files_dir + row['file_name'])
            crop = image[math.floor(bbox[1]):math.ceil(bbox[1] + bbox[3]),
                         math.floor(bbox[0]):math.ceil(bbox[0] + bbox[2])]
            buf_name = row["file_name"].split('.')
            filename = (buf_name[-2]).split('/')[-1]
            filepath = cropped_dir + filename + \
                "-" + str(index) + "." + buf_name[-1]
            cv2.imwrite(filepath, crop)
            new_row = pd.DataFrame(
                [[filepath, row['category_id']]], columns=column_names)
            buf_df = buf_df.append(new_row)
        return buf_df

    def _split_and_save(self, df, save_dir, split_points, headers_string):
        """
        Helper for storing csv files with annotations returned
        """
        train_end = int(split_points[0] * len(df))
        val_end = int(split_points[1] * len(df))
        train, validate, test = np.split(
            df.sample(frac=1), [train_end, val_end])  # we shuffle and split
        np.savetxt(f'{save_dir}train.csv', train, delimiter=",",
                   fmt='%s', header=headers_string, comments='')
        np.savetxt(f'{save_dir}test.csv', test, delimiter=",",
                   fmt='%s', header=headers_string, comments='')
        np.savetxt(f'{save_dir}val.csv', validate, delimiter=",",
                   fmt='%s', header=headers_string, comments='')
        return {'train': f'{save_dir}train.csv', 'test': f'{save_dir}test.csv', 'validate': f'{save_dir}val.csv'}

    def _process_query(self, query, kwargs):
        """
        Helper to process SQL query for Annotations
        query -> sqlalchemy query object
        kwargs['crop_box'] -> if True, cropping and saving will be performed
        kwargs['with_segmentation'] -> if True, only annotations with segmentation will be returned
        kwargs['split_points'] -> stores quantiles for train_test_validation split (default 0.6,0.8)
        kwargs['normalize_cats'] -> set for test purposes only, changes real categories to count from 0 (i.e. cats,dogs(17,18) -> (0,1))
        kwargs['balance_by_min_category'] -> boolean, set to True to balance by minimum amount in some category
        kwargs['balance_by_categories'] -> dictionary with number of elements of each category to query (i.e. {'cat':100,'dog':200})
        """
        df = pd.read_sql(query.statement, query.session.bind)
        av_width = df['width'].mean()
        av_height = df['height'].mean()
        if kwargs.get('crop_bbox', False):
            df = self._prepare_cropped_images(df, kwargs)
        if 'with_segmentation' in kwargs and kwargs['with_segmentation'] is False:
            df_new = pd.DataFrame(columns=['images', 'target'], data=df[[
                                  'file_name', 'category_id']].values)
            headers_string = 'images,target'
        else:
            df_new = pd.DataFrame(columns=['images', 'target', 'segmentation'], data=df[[
                                  'file_name', 'category_id', 'segmentation']].values)
            df_new.dropna(subset=['segmentation'], inplace=True)
            headers_string = 'images,target,segmentation'
        if kwargs.get('balance_by_min_category', False):
            g = df_new.groupby('target', group_keys=False)
            # code to balance-out too large categories with random selection
            df_new = g.apply(lambda x: x.sample(
                g.size().min()).reset_index(drop=True))
        if kwargs.get('balance_by_categories', False):
            cat_names = [el for el in kwargs['balance_by_categories']]
            cat_ids = self.get_cat_IDs_by_names(cat_names)
            cat_ids_dict = {}
            for i in range(len(cat_ids)):
                if cat_ids[i] != -1:
                    cat_ids_dict[cat_ids[i]
                                 ] = kwargs['balance_by_categories'][cat_names[i]]
            g = df_new.groupby('target', group_keys=False)
            df_new = g.apply(lambda x: x.sample(cat_ids_dict[x['target'].iloc[0]]).reset_index(
                drop=True))  # code to balance by given nums
        # A fancy patch for keras to start numbers from 0
        if kwargs.get('normalizeCats', False):
            min_cat = df_new['target'].min()
            df_new['target'] = df_new['target'] - min_cat
        split_points = [0.6, 0.8]
        if 'splitPoints' in kwargs and isinstance(kwargs['splitPoints'], list) and len(kwargs['splitPoints']) == 2:
            split_points = kwargs['splitPoints']
        filename_dict = self._split_and_save(df_new, kwargs.get(
            'curExperimentFolder', './'), split_points, headers_string)
        return df_new, filename_dict, av_width, av_height

    def load_specific_categories_annotations(self, cat_names, **kwargs):
        """Method to load annotations from specific categories, given their IDs.
        INPUT:
            cat_names - list of categories IDs to get annotations from
            kwargs['with_segmentation'] -> if True, only annotations with segmentation will be returned
        OUTPUT:
            pandas dataframe with full annotations for given cat_ids
            dictionary with train, test, val files
            average width, height of images
        """
        query = self.sess.query(self.Image.file_name, self.Image.coco_url, self.Annotation.category_id,
                                self.Annotation.bbox, self.Annotation.segmentation, self.Image.width, self.Image.height
                                ).join(self.Annotation).join(self.Category).filter(self.Category.name.in_(cat_names))
        if 'with_segmentation' in kwargs and kwargs['with_segmentation'] is True:
            # lengths 0 and 1 do not work because there may be strings with empty brackets in DB
            query = query.filter(func.length(self.Annotation.segmentation) > 2)
        return self._process_query(query, kwargs)

    def load_categories_datasets_annotations(self, cat_names, datasets_ids, **kwargs):
        """Method to load annotations from specific categories, given their IDs.
        INPUT:
            cat_names - list of categories IDs to get annotations from
            dataset_ids - list of dataset IDs to get annotations from
            kwargs['with_segmentation'] -> if True, only annotations with segmentation will be returned
        OUTPUT:
            pandas dataframe with full annotations for given cat_ids
            dictionary with train, test, val files
            average width, height of images
        """
        query = self.sess.query(self.Image.file_name, self.Image.coco_url, self.Annotation.category_id,
                                self.Annotation.bbox, self.Annotation.segmentation, self.Image.width, self.Image.height
                                ).join(self.Annotation).join(self.Category).filter(self.Category.name.in_(cat_names)).filter(self.Image.dataset_id.in_(datasets_ids))
        if 'with_segmentation' in kwargs and kwargs['with_segmentation'] is True:
            # lengths 0 and 1 do not work since there are records with empty brackets
            query = query.filter(func.length(self.Annotation.segmentation) > 2)
        return self._process_query(query, kwargs)

    def load_specific_categories_from_specific_datasets_annotations(self, dataset_categories_dict, **kwargs):
        """Method to load annotations from specific datasets filtered by specific categories (different from load_categories_datasets_annotations).
        INPUT:
            dataset_categories_dict - dictionary of categories correspondence, structure: {datasetID1 : [cat1,cat2], datasetID2: [cat1,cat2,cat3], ...}
            kwargs['with_segmentation'] -> if True, only annotations with segmentation will be returned
        OUTPUT:
            pandas dataframe with full annotations for given cat_ids
            dictionary with train, test, val files
            average width, height of images
        """
        result = None
        old_norm_cat_value = None
        sum_width = 0
        sum_height = 0
        if kwargs.get('normalizeCats', False):
            old_norm_cat_value = kwargs['normalizeCats']
            del kwargs['normalizeCats']
        kwargs['normalizeCats'] = False
        for key in dataset_categories_dict:
            cat_names = dataset_categories_dict[key]
            datasets_ids = [key]
            result_vec = self.load_categories_datasets_annotations(
                cat_names, datasets_ids, **kwargs)
            if result is None:
                result = result_vec[0]
            else:
                result = pd.concat([result, result_vec[0]])
            sum_width += len(result.index) * result_vec[2]
            sum_height += len(result.index) * result_vec[3]
        if old_norm_cat_value is not None:
            kwargs['normalizeCats'] = old_norm_cat_value
        if kwargs.get('normalizeCats', False):
            min_cat = result['target'].min()
            result['target'] = result['target'] - min_cat
        split_points = [0.6, 0.8]
        if 'splitPoints' in kwargs and isinstance(kwargs['splitPoints'], list) and len(kwargs['splitPoints']) == 2:
            split_points = kwargs['splitPoints']
        filename_dict = self._split_and_save(result, kwargs.get(
            'curExperimentFolder', './'), split_points, ','.join(list(result.columns)))
        return result, filename_dict, sum_width / len(result.index), sum_height / len(result.index)

    def load_specific_images_annotations(self, image_names, **kwargs):
        """Method to load annotations from specific images, given their names.
        INPUT:
            image_names - list of image names to get annotations for
            kwargs['normalize_cats'] -> set for test purposes only, changes real categories to count from 0 (i.e. cats,dogs(17,18) -> (0,1))
        OUTPUT:
            pandas dataframe with annotations for given image_names
        """
        query = self.sess.query(self.Image.file_name, self.Annotation.category_id, self.Annotation.bbox,
                                self.Annotation.segmentation).join(self.Annotation).filter(self.Image.file_name.in_(image_names))
        df = pd.read_sql(query.statement, query.session.bind)
        # This is a very fancy patch for keras, used only to start numbers of categories from 0
        if 'normalizeCats' in kwargs and kwargs['normalizeCats'] is True:
            df_new = pd.DataFrame(columns=['images', 'target'], data=df[[
                                  'file_name', 'category_id']].values)
            min_cat = df_new['target'].min()
            df_new['target'] = df_new['target'] - min_cat
            return df_new
        return df

    def add_categories(self, categories, respect_ids=True):
        """Method to add given categories to database. Expected to be called rarely, since category list is almost permanent.
        INPUT:
            categories - disctionary with necessary fields: supercategory, name, id
            respect_ids - boolean, to specify if ids from dictionary are preserved in DB
        OUTPUT:
            None
        """
        for category in categories:
            _id = None
            if respect_ids is True:
                _id = category['id']
            newCat = self.Category(
                category['supercategory'], category['name'], _id)
            self.sess.add(newCat)
        self.sess.commit()  # adding categories in db

    def add_default_licences(self):
        """Method to add default licenses to DB. Exptected to be called once"""
        licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                     "name": "Attribution-NonCommercial-ShareAlike License"},
                    {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
                     "name": "Attribution-NonCommercial License"},
                    {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
                     "name": "Attribution-NonCommercial-NoDerivs License"},
                    {"url": "http://creativecommons.org/licenses/by/2.0/",
                        "id": 4, "name": "Attribution License"},
                    {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
                     "name": "Attribution-ShareAlike License"},
                    {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
                     "name": "Attribution-NoDerivs License"},
                    {"url": "http://flickr.com/commons/usage/", "id": 7,
                        "name": "No known copyright restrictions"},
                    {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}]
        for license in licenses:
            lic = self.License(license['name'], license['url'], license['id'])
            self.sess.add(lic)
        self.sess.commit()  # adding licenses from dogs_vs_cats.json

    def add_dataset_info(self, dataset_info):
        """Method to add info about new dataset. Returns added dataset ID"""
        dataset = self.Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'],
                               dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit()  # adding dataset
        return dataset.ID

    def add_images_and_annotations(self, images, annotations, dataset_id, file_prefix='', respect_ids=False):
        """Method to add a chunk of images and their annotations to DB.

        Parameters
        ----------
        images : list[dict]
            array of dicts with attributes:
            license, file_name, coco_url, height, width, date_captured, flickr_url, id
        annotations : list[dict]
            array of dicts with attributes:
            segmentation, area, iscrowd, image_id, bbox, category_id, id
        dataset_id : any
            ID of a dataset images are from
        file_prefix : str
            prefix to be added to filenames
        respect_ids : bool
            boolean to specify if input ids are preserved in DB

        Returns
        -------
        None
        """
        if not os.path.isfile(file_prefix + images[0]['file_name']):
            print('Error in json file, missing images stored on disc (i.e.',
                  file_prefix + images[0]['file_name'], ')')
            print('Stop filling dataset')
            return
        print('Adding images')
        buf_images = {}
        self.printProgressBar(
            0, len(images), prefix='Adding images:', suffix='Complete', length=50)
        im_counter = 0
        for im_data in images:
            im_id = None
            if respect_ids is True:
                im_id = im_data['id']
            image = self.Image(file_prefix + im_data['file_name'], im_data['width'], im_data['height'],
                               im_data['date_captured'], dataset_id, im_data['coco_url'], im_data['flickr_url'], im_data['license'], im_id)
            buf_images[im_data['id']] = image
            if im_counter % 10 == 0 or im_counter == len(images) - 2:
                self.printProgressBar(im_counter, len(
                    images), prefix='Adding images:', suffix='Complete', length=50)
            im_counter += 1
            self.sess.add(image)
        self.sess.commit()  # adding images
        print('Done adding images, adding annotations')
        counter = 0
        self.printProgressBar(
            0, len(annotations), prefix='Adding annotations:', suffix='Complete', length=50)
        an_counter = 0
        for an_data in annotations:
            # print(counter)
            counter += 1
            anno_id = None
            if respect_ids is True:
                anno_id = an_data['id']
            cur_image_id = buf_images[an_data['image_id']].ID
            seg_str = json.dumps(an_data['segmentation'])
            bbox_str = json.dumps(an_data['bbox'])
            annotation = self.Annotation(
                cur_image_id, an_data['category_id'], bbox_str, seg_str, an_data['iscrowd'], an_data['area'], anno_id)
            if an_counter % 10 == 0 or an_counter == len(annotations) - 2:
                self.printProgressBar(an_counter, len(
                    annotations), prefix='Adding annotations:', suffix='Complete', length=50)
            an_counter += 1
            self.sess.add(annotation)
        self.sess.commit()  # adding annotations

    def add_model_record(self, task_type, categories, model_address, metrics, history_address=''):
        """
        Inserts records about train results for some model.
        If model already exists method does not create new model.
        If key update_metrics is set to True, then metric records will be updated if they already exist
        """
        if not isinstance(task_type, str):
            print(
                'ERROR: Bad input for global history record, expected string as task_type')
            return
        if not isinstance(categories, list):
            print('ERROR: Bad input for global history record, expected list of objects')
            return
        if not isinstance(model_address, str):
            print(
                'ERROR: Bad input for global history record, expected string for model_address')
            return
        if not isinstance(metrics, dict):
            print(
                'ERROR: Bad input for global history record, expected dictionary with merics')
            return

        abs_model_address = os.path.abspath(model_address)
        abs_history_address = os.path.abspath(history_address)
        # model should be identified by its address uniquely
        model_from_db = self.sess.query(self.Model).filter(
            self.Model.model_address == abs_model_address).first()
        if model_from_db is None:
            # Model not in DB - add it (that's OK)
            new_model = self.Model(abs_model_address, task_type)
            self.sess.add(new_model)
            self.sess.commit()
            for cat_name in categories:  # categories should not change for the model - they are attached once
                categoryFromDB = self.sess.query(self.Category).filter(
                    self.Category.name == cat_name).first()
                if categoryFromDB is None:
                    # That's a very bad case - we cannot simply add new category, DB may become inconsistent
                    print("ERROR: No category " + cat_name + " in DB")
                    return
                new_cat_record = self.CategoryToModel(
                    categoryFromDB.ID, new_model.ID)
                self.sess.add(new_cat_record)
            self.sess.commit()
            model_from_db = new_model
        else:
            print(
                "ERROR: model was already added in DB. Check model_address for correctness.")
            return

        # We do not check if metric is valid - module user should keep track on consistency of these records
        for key, value in metrics.items():
            new_train_result = self.TrainResult(
                key, value, model_from_db.ID, abs_history_address)
            self.sess.add(new_train_result)
            self.sess.commit()
        return

    def update_train_result_record(self, model_address, metric_name, metric_value, history_address=''):
        """
        Returns boolean of operation success
        """
        if not isinstance(model_address, str):
            print(
                'ERROR: Bad input for global history record, expected string for model_address')
            return False
        if not isinstance(metric_name, str):
            print(
                'ERROR: Bad input for global history record, expected string for metric_name')
            return False

        abs_model_address = os.path.abspath(model_address)
        abs_history_address = os.path.abspath(history_address)
        model_from_db = self.sess.query(self.Model).filter(
            self.Model.model_address == abs_model_address).first()
        if model_from_db is None:
            print('ERROR: Model does not exist in database')
            return False

        for trRes in model_from_db.train_results:
            if trRes.metric_name == metric_name:
                trRes.metric_value = metric_value
                trRes.history_address = abs_history_address
                self.sess.commit()
                return True
        # no such metric was found - add new train result
        print('Adding new metric for this model, metric name:', metric_name)
        new_train_result = self.TrainResult(
            metric_name, metric_value, model_from_db.ID, abs_history_address)
        self.sess.add(new_train_result)
        self.sess.commit()
        return True

    def delete_train_result_record(self, model_address, metric_name):
        """
        Returns boolean of operation success
        """
        if not isinstance(model_address, str):
            print(
                'ERROR: Bad input for global history record, expected string for model_address')
            return False
        if not isinstance(metric_name, str):
            print(
                'ERROR: Bad input for global history record, expected string for metric_name')
            return False

        abs_model_address = os.path.abspath(model_address)
        model_from_db = self.sess.query(self.Model).filter(
            self.Model.model_address == abs_model_address).first()
        if model_from_db is None:
            print('ERROR: Model does not exist in database')
            return False

        for trRes in model_from_db.train_results:
            if trRes.metric_name == metric_name:
                ret = self.sess.query(self.TrainResult).filter_by(
                    ID=trRes.ID).delete()
                self.sess.commit()
                return True
        print('ERROR: Such train result record was not found')
        return False

    def get_models_by_filter(self, filter_dict, exact_category_match=False):
        """
        Returns list of models that match filter_dict

        Parameters
        ----------
        filter_dict : dict
            filter_dict is a dictionary which contains params for model search.
            Specification for this structure can be changed in time.
            Possible key-value pairs:
                'min_metrics':
                    {
                        'metric_name': min_value
                    }
                'categories': ['list','of','categories','names']
        exact_category_match : bool
            If True, then only models that have exactly the same categories will be returned.

        Returns
        -------
        Generator
            returns pd with models info
        """
        model_query = self.sess.query(self.Model, self.TrainResult).join(
            self.TrainResult).join(self.CategoryToModel).join(self.Category)
        if 'categories_ids' in filter_dict:
            filter_dict['categories'] = self.get_cat_names_by_IDs(
                filter_dict['categories_ids'])  # this is just a patch
            print(filter_dict['categories'])
        if 'categories' in filter_dict:
            model_query = model_query.filter(
                self.Category.name.in_(filter_dict['categories']))
            # TODO: maybe exact category matches need to be done later
        if 'min_metrics' in filter_dict:
            if not isinstance(filter_dict['min_metrics'], dict):
                print('ERROR: Bad input for min_metrics - should be dict')
                return
            for key, value in filter_dict['min_metrics'].items():
                model_query = model_query.filter(and_(self.TrainResult.metric_value >= value,
                                                      self.TrainResult.metric_name == key))
        # I didn't yet find a way to make this better in performance
        model_query = model_query.group_by(self.Model.model_address)
        if 'categories' in filter_dict and len(filter_dict['categories']) != 0:
            cands = model_query.all()
            good_IDs = []
            for cand_buf in cands:
                cand = cand_buf[0]
                model_cand = self.sess.query(self.Model).filter(
                    self.Model.ID == cand.ID).first()
                model_categories = set()
                cat_ids = []
                for cat in model_cand.categories:
                    cat_ids.append(cat.category_id)
                mod_cats = self.sess.query(self.Category).filter(
                    self.Category.ID.in_(cat_ids))
                for mod_cat in mod_cats:
                    model_categories.add(mod_cat.name)
                if model_categories.issubset(set(filter_dict['categories'])):
                    good_IDs.append(model_cand.ID)
            model_query = model_query.filter(self.Model.ID.in_(good_IDs))
        df = pd.read_sql(model_query.statement, model_query.session.bind)
        return df

    def get_cat_IDs_by_names(self, cat_names):
        """
        cat_names - list of names of categories. Returns list of IDs (one-to-one correspondence).
        If category is not present, -1 is returned on its position.
        In case of bad input empty list is returned.
        """
        if not isinstance(cat_names, list):
            print('ERROR: Bad input for cat_names, must be a list')
            return []
        result = []
        for cat_name in cat_names:
            query = self.sess.query(self.Category.ID).filter(
                self.Category.name == cat_name).first()
            if query is None:
                result.append(-1)
            else:
                result.append(query[0])
        return result

    def get_cat_names_by_IDs(self, cat_IDs):
        """
        cat_IDs - list of IDs of categories. Returns list of names (one-to-one correspondence).
        If category is not present, "" is returned on its position.
        In case of bad input empty list is returned.
        """
        if not isinstance(cat_IDs, list):
            print('ERROR: Bad input for cat_names, must be a list')
            return []
        result = []
        for cat_ID in cat_IDs:
            query = self.sess.query(self.Category.name).filter(
                self.Category.ID == cat_ID).first()
            if query is None:
                result.append("")
            else:
                result.append(query[0])
        return result

    def get_full_dataset_info(self, ds_id):
        """
        Input:
            ds_id - identifier of a dataset inside database (IDs can be aquired by, i.e., get_all_datasets method)
        Output:
            full information about dataset given its ID in database in a dictionary
            dictionary keys:
                'dataset_info' -> brief info on dataset
                'categories' -> explicit number of specific categories in this dataset
        """
        result = {}
        query = self.sess.query(self.Dataset).filter(self.Dataset.ID == ds_id)
        df = pd.read_sql(query.statement, query.session.bind)
        result = df.to_dict('list')
        for key in result:
            # to be consistent with ds_info default dict
            result[key] = result[key][0]
        cat_query = self.sess.query(self.Annotation.category_id).join(
            self.Image).filter(self.Image.dataset_id == ds_id)
        cat_counts = self.sess.query(self.Annotation.category_id, func.count(self.Annotation.category_id)).join(
            self.Image).filter(self.Image.dataset_id == ds_id).group_by(self.Annotation.category_id).all()
        cat_counts_dict = {}
        for record in cat_counts:
            cat_counts_dict[record[0]] = record[1]
        categories_in_ds = cat_query.group_by(
            self.Annotation.category_id).all()
        categories_in_ds = list(np.array(categories_in_ds).flatten())
        cats_df = self.get_all_categories()
        result['categories'] = {}
        for cat_id in categories_in_ds:
            cat_name = cats_df.loc[cats_df['ID'] == cat_id].values[0][2]
            cat_count = cat_counts_dict[cat_id]
            result['categories'][cat_name] = cat_count
        return result
