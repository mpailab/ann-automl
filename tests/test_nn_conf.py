from ann_automl.utils.nn_conf import add_layers_from_conf
import keras
import keras.layers

if __name__ == '__main__':
    x = keras.layers.Input(shape=64)
    y = add_layers_from_conf(x, 'ModelLastLayers', 'data/layer_hist.txt', verbose=True)
    model = keras.models.Model(inputs=x, outputs=y)
    model.summary()
