import vislib
from tensorflow import keras
model = keras.models.load_model('test.hdf5')
print(model.summary())
vislib.plot_conv_weights(model,'conv2d_6') #conv2d_20