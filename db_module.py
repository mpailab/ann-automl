from sqlalchemy import Column, Integer, String, ForeignKey, Table, Float, create_engine
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

Base = declarative_base()

class dbModule:

    ############################################################
    ##########        DB ORM description     ###################
    ############################################################
    
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
        images = relationship("Image", backref=backref("license"))
        def __init__(self, name, url, _id = None, aux = ''):
            self.url = url
            self.name = name
            if _id != None:
                self.ID = _id
            self.aux = aux
        
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

    class Annotation(Base):
        __tablename__ = "annotation"
        ID = Column(Integer, primary_key=True)
        image_id = Column(Integer, ForeignKey("image.ID"))
        category_id = Column(Integer, ForeignKey("category.ID"))
        bbox = Column(String)
        segmentation = Column(String)
        isCrowd = Column(Integer)
        area = Column(Float)
        aux = Column(String)
        def __init__(self, image_id, category_id, bbox, segmentation, isCrowd, area, _id = None, aux = ''):
            self.image_id = image_id
            self.category_id = category_id
            self.bbox = bbox
            self.segmentation = segmentation
            self.isCrowd = isCrowd
            self.area = area
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
    
    def fill_coco(self, annoFileName, ds_info = None):
        '''Method to fill COCOdataset into db. It is supposed to be called once.
        INPUT:
            annoFileName - file with json annotation in COCO format for cats and dogs
            ds_info - dictionary with info about dataset (default - COCO2017). Necessary keys:
                description, url, version, year, contributor, date_created
        OUTPUT:
            None
        '''
        coco=COCO(annFile)
        cats = coco.loadCats(coco.getCatIds())
        ##TODO: CALL THE FOLLOWING TWO METHODS ONLY WHEN NEEDED - MAKE A CHECK
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
                image = cv2.imread(files_dir + row['file_name'])
                crop = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                buf_name = row["file_name"].split('.')
                cv2.imwrite(cropped_dir + buf_name[-2] + "-" + str(index) + "." + buf_name[-1] ,crop)
                newRow = pd.DataFrame([[buf_name[-2] + "-" + str(index) + "." + buf_name[-1], row['category_id']]], columns = column_names)
                buf_df = buf_df.append(newRow)
            df = buf_df
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))]) #TODO Split in scpecific sizes
        np.savetxt("train.csv", train, delimiter=",")
        np.savetxt("test.csv", test, delimiter=",")
        np.savetxt("validate.csv", validate, delimiter=",")
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
            print(counter)
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