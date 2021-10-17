from sqlalchemy import *
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from sqlalchemy.orm import sessionmaker
import json
import ast
import cv2
from pathlib import Path
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import datetime
import os

Base = declarative_base()

class dbModule:

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
        def __init__(self, file_name, width, height, date_captured, dataset_id, coco_url = '' ,flickr_url = '', license_id = -1, _id = None, aux = ''):
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
            if _id != None:
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
        def __init__(self, description, url, version, year, contributor, date_created, _id = None, aux = ''):
            self.description = description
            self.url = url
            self.version = version
            self.year = year
            self.contributor = contributor
            self.date_created = date_created
            if _id != None:
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
        def __init__(self, supercategory, name, _id = None, aux = ''):
            self.supercategory = supercategory
            self.name = name
            if _id != None:
                self.ID = _id
            self.aux = aux
        
    class License(Base):
        __tablename__ = "license"
        ID = Column(Integer, primary_key=True)
        name = Column(String)
        url = Column(String)
        aux = Column(String)
        images = relationship("Image")
        def __init__(self, name, url, _id = None, aux = ''):
            self.url = url
            self.name = name
            if _id != None:
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
        def __init__(self, image_id, category_id, bbox, segmentation, isCrowd, area, _id = None, aux = ''):
            self.image_id = image_id
            self.category_id = category_id
            self.bbox = bbox
            self.segmentation = segmentation
            self.is_crowd = isCrowd
            self.area = area
            if _id != None:
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
        def __init__(self, metricName, metricValue, modelID, historyAddress = '', aux = '', _id = None):
            self.metric_name = metricName
            self.metric_value = metricValue
            self.history_address = historyAddress
            self.model_id = modelID
            if _id != None:
                self.ID = _id
            self.aux = aux
    
    class CategoryToModel(Base):
        __tablename__ = "categoryToModel"
        category_id = Column(Integer, ForeignKey("category.ID"), primary_key = True)
        model_id = Column(Integer, ForeignKey("model.ID"), primary_key = True) 
        def __init__(self, category_id, model_id):
            self.category_id = category_id
            self.model_id = model_id

    class Model(Base):
        __tablename__ = "model"
        ID = Column(Integer, primary_key=True)
        model_address = Column(String)
        task_type = Column(String) #TODO - this is very bad from DB perspective, but I'll leave for later
        aux = Column(String)
        train_results = relationship("TrainResult", backref=backref("model"))
        categories = relationship("CategoryToModel") 
        def __init__(self, modelAddress, taskType, aux = '', _id = None):
            self.model_address = modelAddress
            self.task_type = taskType
            if _id != None:
                self.ID = _id
            self.aux = aux
    ############################################################
    ##########        DB Module methods      ###################
    ############################################################

    def __init__(self, dbstring = 'sqlite:///datasets.sqlite', dbecho = False):
        self.engine = create_engine(dbstring, echo = dbecho)
        Session = sessionmaker(bind=self.engine)
        self.sess = Session()
    def create_sqlite_file(self):
        Base.metadata.create_all(self.engine)
    def fill_cats_dogs(self, annoFileName = 'dogs_vs_cats_coco_anno.json'):
        '''Method to fill Kaggle CatsVsDogs dataset into db. It is supposed to be called once.
        INPUT:
            annoFileName - file with json annotation in COCO format for cats and dogs
        OUTPUT:
            None
        '''
        with open(annoFileName) as json_file:
            data = json.load(json_file)
        dataset_info = data['info']
        dataset = self.Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'], dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit() #adding dataset
        ###################################
        im_objects = {}
        for im_data in data['images']:
            image = self.Image(im_data['file_name'], im_data['width'], im_data['height'], im_data['date_captured'], dataset.ID, im_data['coco_url'], im_data['flickr_url'], im_data['license'])
            im_objects[im_data['id']] = image
            self.sess.add(image)
        self.sess.commit() #adding images
        ###################################
        for an_data in data['annotations']:
            #TODO: +1 because of error in json - should be fixed later
            real_id = im_objects[an_data['image_id']].ID
            annotation = self.Annotation(real_id, an_data['category_id'] + 1, ';'.join(an_data['bbox']), ';'.join(an_data['segmentation']), an_data['iscrowd'], an_data['area'])
            self.sess.add(annotation)
        self.sess.commit() #adding annotations
    
    def fill_coco(self, annoFileName, firstTime = False, ds_info = None):
        '''Method to fill COCOdataset into db. It is supposed to be called once.
        INPUT:
            annoFileName - file with json annotation in COCO format for cats and dogs
            ds_info - dictionary with info about dataset (default - COCO2017). Necessary keys:
                description, url, version, year, contributor, date_created
        OUTPUT:
            None
        '''
        coco=COCO(annoFileName)
        cats = coco.loadCats(coco.getCatIds())
        ##CALL THE FOLLOWING TWO METHODS ONLY WHEN NEEDED - WE MAKE A CHECK - USER IS RESPONSIBLE
        if firstTime:
            self.add_categories(cats, True)
            self.add_default_licences()
        #######################################################################
        if ds_info == None:
            ds_info = {"description": "COCO 2017 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2017,"contributor": "COCO Consortium","date_created": "2017/09/01"}
        dsID = self.add_dataset_info(ds_info)
        imgIds = coco.getImgIds()
        imgs = coco.loadImgs(imgIds)
        annIds = coco.getAnnIds()
        anns = coco.loadAnns(annIds)
        print('Adding '+ str(len(anns)) + ' annotations in COCO format to DB')
        self.add_images_and_annotations(imgs, anns, dsID)
        return
    
    def load_specific_datasets_annotations(self, datasets_ids):
        '''Method to load annotations from specific datasets, given their IDs.
        INPUT:
            datasets_ids - list of datasets IDs to get annotations from
        OUTPUT:
            pandas dataframe with annotations for given datasets IDs
        '''
        query = self.sess.query(self.Image.file_name, self.Annotation.category_id, self.Annotation.bbox, self.Annotation.segmentation).join(self.Annotation).filter(self.Image.dataset_id.in_(datasets_ids))
        df = pd.read_sql(query.statement, query.session.bind)
        return df
    
    def load_specific_categories_annotations(self, cat_ids, **kwargs):
        '''Method to load annotations from specific categories, given their IDs.
        INPUT:
            cat_ids - list of categories IDs to get annotations from
        OUTPUT:
            pandas dataframe with full annotations for given cat_ids
            dictionary with train, test, val files
            average width, height of images
        '''
        query = self.sess.query(self.Image.file_name, self.Image.coco_url, self.Annotation.category_id, self.Annotation.bbox, self.Annotation.segmentation, self.Image.width, self.Image.height).join(self.Annotation).filter(self.Annotation.category_id.in_(cat_ids))
        df = pd.read_sql(query.statement, query.session.bind)
        av_width = df['width'].mean()
        av_height = df['height'].mean()
        if 'crop_bbox' in kwargs and kwargs['crop_bbox'] == True:
            column_names = ["file_name", "category_id"]
            buf_df = pd.DataFrame(columns = column_names)
            cropped_dir = 'buf_crops/'
            if 'cropped_dir' in kwargs and kwargs['cropped_dir'] != '':
                Path(kwargs['cropped_dir']).mkdir(parents=True, exist_ok=True)
                cropped_dir = kwargs['cropped_dir']
            files_dir = './'
            if 'files_dir' in kwargs and kwargs['files_dir'] != '':
                files_dir = kwargs['files_dir']
            for index, row in df.iterrows():
                bbox = ast.literal_eval(row['bbox'])
                if len(bbox) != 4:
                    buf_df = buf_df.append(row['file_name'], row['category_id']) #nothing to cut
                    continue
                image = cv2.imread(files_dir + row['file_name'])
                crop = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                buf_name = row["file_name"].split('.')
                cv2.imwrite(cropped_dir + buf_name[-2] + "-" + str(index) + "." + buf_name[-1] ,crop)
                newRow = pd.DataFrame([['.'.join(buf_name[:-2]) + "-" + str(index) + "." + buf_name[-1], row['category_id']]], columns = column_names)
                buf_df = buf_df.append(newRow)
            df = buf_df
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))]) #TODO Split in scpecific sizes
        print('Train shape:',train.shape,' test shape:', test.shape, 'validation shape', validate.shape)
        np.savetxt("train.csv", train, delimiter=",", fmt='%s')
        np.savetxt("test.csv", test, delimiter=",", fmt='%s')
        np.savetxt("validate.csv", validate, delimiter=",", fmt='%s')
        return df, {'train':'train.csv', 'test':'test.csv', 'validate':'validate.csv'}, av_width, av_height
    
    def load_specific_images_annotations(self, image_names):
        '''Method to load annotations from specific images, given their names.
        INPUT:
            image_names - list of image names to get annotations for
        OUTPUT:
            pandas dataframe with annotations for given image_names
        '''
        query = self.sess.query(self.Image.file_name, self.Annotation.category_id, self.Annotation.bbox, self.Annotation.segmentation).join(self.Annotation).filter(self.Image.file_name.in_(image_names))
        df = pd.read_sql(query.statement, query.session.bind)
        return df
    
    def add_categories(self, categories, respect_ids = True):
        '''Method to add given categories to database. Expected to be called rarely, since category list is almost permanent.
        INPUT:
            categories - disctionary with necessary fields: supercategory, name, id
            respect_ids - boolean, to specify if ids from dictionary are preserved in DB
        OUTPUT:
            None
        '''
        for category in categories:
            _id = None
            if respect_ids == True:
                _id = category['id']
            newCat = self.Category(category['supercategory'], category['name'], _id)
            self.sess.add(newCat)
        self.sess.commit() #adding categories in db
        
    def add_default_licences(self):
        '''Method to add default licenses to DB. Exptected to be called once'''
        licenses =  [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},{"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},{"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},{"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},{"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},{"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}]
        for license in licenses:
            lic = self.License(license['name'], license['url'], license['id'])
            self.sess.add(lic)
        self.sess.commit() #adding licenses from dogs_vs_cats.json
    
    def add_dataset_info(self, dataset_info):
        '''Method to add info about new dataset. Returns added dataset ID'''
        dataset = self.Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'], dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit() #adding dataset
        return dataset.ID
    
    def add_images_and_annotations(self, images, annotations, dataset_id, file_prefix = '', respect_ids = False):
        '''Method to add a chunk of images and their annotations to DB. 
        INPUT:
            images - array of dicts with attributes:
                license, file_name, coco_url, height, width, date_captured, flickr_url, id
            annotations - array of dicts with attributes:
                segmentation, area, iscrowd, image_id, bbox, category_id, id
            dataset_id - ID of a dataset images are from
            file_prefix - prefix to be added to filenames
            respect_ids - boolean to specify if input ids are preserved in DB
        OUTPUT:
            None
        '''
        print('Adding images')
        buf_images = {}
        for im_data in images:
            im_id = None
            if respect_ids == True:
                im_id = im_data['id']
            image = self.Image(file_prefix + im_data['file_name'], im_data['width'], im_data['height'], im_data['date_captured'], dataset_id, im_data['coco_url'], im_data['flickr_url'], im_data['license'], im_id)
            buf_images[im_data['id']] = image
            self.sess.add(image)
        self.sess.commit() #adding images
        print('Done adding images, adding annotations')
        counter = 0
        for an_data in annotations:
            #print(counter)
            counter+=1
            anno_id = None
            if respect_ids == True:
                anno_id = im_data['id']
            cur_image_id = buf_images[an_data['image_id']].ID
            seg_str = json.dumps(an_data['segmentation'])
            bbox_str = json.dumps(an_data['bbox'])
            annotation = self.Annotation(cur_image_id, an_data['category_id'], bbox_str, seg_str, an_data['iscrowd'], an_data['area'], anno_id)
            self.sess.add(annotation)
        self.sess.commit() #adding annotations
    
    def add_model_record(self, task_type, categories, model_address, metrics, history_address = ''):
        if not isinstance(task_type, str):
            print('ERROR: Bad input for global history record, expected string as task_type')
            return
        if not isinstance(categories, list):
            print('ERROR: Bad input for global history record, expected list of objects')
            return
        if not isinstance(model_address, str):
            print('ERROR: Bad input for global history record, expected string for model_address')
            return
        if not isinstance(metrics, dict):
            print('ERROR: Bad input for global history record, expected dictionary with merics')
            return

        abs_model_address = os.path.abspath(model_address)
        abs_history_address = os.path.abspath(history_address)
        modelFromDB= self.sess.query(self.Model).filter(self.Model.model_address == abs_model_address).filter(self.Model.task_type == task_type).first()
        if modelFromDB is None:
            #Model not in DB - add it (that's OK)
            new_model = self.Model(abs_model_address, task_type)
            self.sess.add(new_model)
            self.sess.commit()
            modelFromDB = new_model
        
        trainResultFromDB = {}
        #We do not check is metric is valid - module user should keep track on consistency of these records
        for key, value in metrics.items():
             #Metric not in DB - add it (that's OK)
            new_train_result = self.TrainResult(key, value, modelFromDB.ID, abs_history_address)
            self.sess.add(new_train_result)
            self.sess.commit()
            trainResultFromDB[key] = new_train_result

            for cat_name in categories:
                categoryFromDB = self.sess.query(self.Category).filter(self.Category.name == cat_name).first()
                if categoryFromDB is None:
                    #That's a very bad case - we cannot simply add new category, DB may become inconsistent
                    print("ERROR: No category " + cat_name + " in DB")
                    return
                new_cat_record = self.CategoryToModel(categoryFromDB.ID, trainResultFromDB[key].ID)
                self.sess.add(new_cat_record)
                self.sess.commit()
            #print(new_record.ID, new_record.metric_id, new_record.model_id)        
        return

    def get_models_by_filter(self, filter_dict, exact_category_match = False):
        '''
        filter_dict is a dictionary which contains params for model search. 
        Specification for this structure can be changed in time.
        Possible key-value pairs:
            'min_metrics':
                {
                    'metric_name': min_value
                }
            'categories': ['list','of','categories','names']
        
        returns pd with models info
        '''
        model_query = self.sess.query(self.Model, self.TrainResult)
        if('categories' in filter_dict):
            model_query = model_query.join(self.CategoryToModel).join(self.Category).filter(self.Category.name.in_(filter_dict['categories']))
            #TODO: exact category matches
        if('min_metrics' in filter_dict):   
            if not isinstance(filter_dict['min_metrics'], dict):
                print('ERROR: Bad input for min_metrics - should be dict')
                return
            model_query = model_query.join(self.TrainResult)
            for key, value in filter_dict['min_metrics'].items():
                model_query = model_query.filter(and_(self.TrainResult.metric_value >= value, self.TrainResult.metric_name == key))
        df = pd.read_sql(model_query.statement, model_query.session.bind)
        return df