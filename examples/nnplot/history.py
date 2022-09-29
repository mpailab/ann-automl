import vislib
import numpy as np
history=np.load('my_history.npy', allow_pickle=True).item()
vislib.plot(history)