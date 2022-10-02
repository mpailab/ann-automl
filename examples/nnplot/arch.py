import ann_automl.nnplot.vislib as vislib
from tensorflow import keras
model = keras.models.load_model('test.hdf5')
print(model.summary())
vislib.arch(model)
