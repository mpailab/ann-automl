import ann_automl.nnplot.vislib as vislib
import numpy as np
history=np.load('my_history.npy', allow_pickle=True).item()
vislib.plot_history(history)
