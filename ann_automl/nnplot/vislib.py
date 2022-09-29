import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from keras.models import Model

def arch(model,
        to_file='model.png'
    ):
    """
    Plots keras model architecture into image file
    Parameters
    ----------
    model: tf.keras.Model
        Model structure
    to_file: str, optional
        Image file name to save else it will be displayed
    Raises
    ------
    """    
    plot_model(model, to_file)

def plot_history(history,
        plotlist=['accuracy','val_accuracy'],
        label='accuracy',
        title='model accuracy', 
        to_file=None
    ):
    """
    Plots training model dynamic like accuracy/loss/smth else containing series of data for each epoch
    Parameters
    ----------
    history: dict
        Dictionary with data(key: array of values)
    plotlist: list of str
        List of keys to plot
    label: str
        Label of vertical axis (horisontal is for epochs)
    title: str
        Title of the plot
    to_file: str, optional
        Image file name to save else it will be displayed
    Raises
    ------
    """
    for pl in plotlist:
        plt.plot(history[pl])
    plt.title(title)
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.legend(plotlist, loc='upper left')
    if to_file is None:
        plt.show()
    else:
        plt.savefig(to_file)

def plot_conv_weights(model, 
        layer_name,
        title='Convolutional layer filters', 
        to_file=None
    ):
    """
    Plots weights of convolutional filters for certain convolutional layer
    Parameters
    ----------
    model: tf.keras.Model
        Model structure
    layer_name: str
        Name of convolutional layer to plot it's weights
    title: str
        Title of the plot
    to_file: str, optional
        Image file name to save else it will be displayed
    Raises
    ------
    """
    W = model.get_layer(name=layer_name).get_weights()[0]
    assert(len(W.shape) == 4)
    plt.title(title)
    W = np.squeeze(W)
    W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
    cnt=W.shape[2]
    if cnt>=25:
        fig, axs = plt.subplots(5,5, figsize=(8,8))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(W[:,:,i])
            axs[i].set_title(str(i))
    else:
        fig, axs = plt.subplots(cnt,1, figsize=(2,cnt*2))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(cnt):
            axs[i].imshow(W[:,:,i])
            axs[i].set_title(str(i))
    if to_file is None:
        plt.show()
    else:
        plt.savefig(to_file)

def plot_conv_outputs(model,
        inp, 
        layer_name,
        to_file=None
    ):
    """
    Plots convolutional layer outputs (maps) for certain layer on single input image
    Parameters
    ----------
    model: tf.keras.Model
        Model structure
    inp: numpy.array
        Preprocessed input image
    layer_name: str
        Name of convolutional layer to plot it's output
    title: str
        Title of the plot
    to_file: str, optional
        Image file name to save
    Raises
    ------
    """
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    O = intermediate_layer_model.predict(np.expand_dims(inp,axis=0))[0]
    assert(len(O.shape)==3)
    cnt=O.shape[2]
    if cnt>=25:
        fig, axs = plt.subplots(5,5, figsize=(8,8))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(O[:,:,i])
            axs[i].set_title(str(i))
    else:
        fig, axs = plt.subplots(cnt,1, figsize=(2,cnt*2))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(cnt):
            axs[i].imshow(O[:,:,i])
            axs[i].set_title(str(i))
    if to_file is None:
        plt.show()
    else:
        plt.savefig(to_file)
