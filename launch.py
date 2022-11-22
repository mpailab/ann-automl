# run from terminal: panel serve launch.py --show --autoreload --allow-websocket-origin=localhost:<port>

from ann_automl.core.nnfuncs import set_emulation, multithreading_mode, set_multithreading_mode
from ann_automl.gui.nn_gui import interface

# set_emulation(True)
#with multithreading_mode():
set_multithreading_mode(True)
interface.servable()
