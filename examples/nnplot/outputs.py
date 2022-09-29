import vislib
from tensorflow import keras
from keras.datasets import cifar10
import cv2
mean=(125.30694,122.95031,122.95031)
std_dev=(62.993233,62.08874,66.70485)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cv2.imwrite("sample.png",x_test[1000])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test[:,:,:] -= mean
x_test[:,:,:] /= std_dev

model = keras.models.load_model('test.hdf5')
print(model.summary())
#vislib.plot_conv_outputs(model,x_test[1000],'input_1')
vislib.plot_outputs(model,x_test[1000],'activation_9') #'conv2d_6')