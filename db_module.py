from sqlalchemy import Column, Integer, String, ForeignKey, Table, Float
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from sqlalchemy.orm import sessionmaker
import json

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
        def __init__(self, description, url, version, year, contributor, date_created, aux = ''):
            self.description = description
            self.url = url
            self.version = version
            self.year = year
            self.contributor = contributor
            self.date_created = date_created
            self.aux = aux

    class Category(Base):
        __tablename__ = "category"
        ID = Column(Integer, primary_key=True)
        supercategory = Column(String)
        name = Column(String)
        aux = Column(String)
        images = relationship("Annotation", backref=backref("category"))
        def __init__(self, supercategory, name, aux = ''):
            self.supercategory = supercategory
            self.name = name
            self.aux = aux
        
    class License(Base):
        __tablename__ = "license"
        ID = Column(Integer, primary_key=True)
        name = Column(String)
        url = Column(String)
        aux = Column(String)
        images = relationship("Image", backref=backref("license"))
        def __init__(self, name, url, aux = ''):
            self.url = url
            self.name = name
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
        def __init__(self, file_name, width, height, date_captured, dataset_id, coco_url = '' ,flickr_url = '', license_id = -1, aux = ''):
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
        def __init__(self, image_id, category_id, bbox, segmentation, isCrowd, area, aux = ''):
            self.image_id = image_id
            self.category_id = category_id
            self.bbox = bbox
            self.segmentation = segmentation
            self.isCrowd = isCrowd
            self.area = area
            self.aux = aux
    
    ############################################################
    ##########        DB Module methods      ###################
    ############################################################

    def __init__(self, dbstring = 'sqlite:///datasets.sqlite', dbecho = False):
        self.engine = create_engine(dbstring, echo = dbecho)
        Session = sessionmaker(bind=engine)
        self.sess = Session()
    def create_sqlite_file(self):
        Base.metadata.create_all(engine)
    def fill_cats_dogs(self, categoriesFileName = '2017.csv', annoFileName = 'dogs_vs_cats_coco_anno.json'):
        ###################################
        df = pd.read_csv(categoriesFileName, sep = ';', header = None)
        buf_df = df.iloc[:, [1, 2]]
        for index, row in buf_df.iterrows():
            cat = Category(row[2], row[1])
            self.sess.add(cat)
        self.sess.commit() #adding categories in db
        ###################################
        with open(annoFileName) as json_file:
            data = json.load(json_file)
        for license in data['licenses']:
            lic = License(license['name'], license['url'])
            self.sess.add(lic)
        self.sess.commit() #adding licenses from dogs_vs_cats.json
        ###################################
        dataset = Dataset(dataset_info['description'], dataset_info['url'], dataset_info['version'], dataset_info['year'], dataset_info['contributor'], dataset_info['date_created'])
        self.sess.add(dataset)
        self.sess.commit() #adding dataset
        ###################################
        for im_data in data['images']:
            image = Image(im_data['file_name'], im_data['width'], im_data['height'], im_data['date_captured'], dataset_id, im_data['coco_url'], im_data['flickr_url'], im_data['license'])
            self.sess.add(image)
        self.sess.commit() #adding images
        ###################################
        for an_data in data['annotations']:
            #TODO: +1 because of error in json - should be fixed later
            annotation = Annotation(an_data['image_id'] + 1, an_data['category_id'], ';'.join(an_data['bbox']), ';'.join(an_data['segmentation']), an_data['iscrowd'], an_data['area'])
            self.sess.add(annotation)
        self.sess.commit() #adding annotations
    def load_specific_datasets_annotations(self, datasets_ids):
        #load specific list of dataset_ids information and return it in pandas table
        query = self.sess.query(Image.file_name, Annotation.category_id).join(Annotation).filter(Image.dataset_id.in_(datasets_ids))
        df = pd.read_sql(query.statement, query.session.bind)
        return df