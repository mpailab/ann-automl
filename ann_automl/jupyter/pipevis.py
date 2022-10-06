import threading
from IPython.display import display
import ipywidgets as widgets
import time

from ..utils.process import VarWaiter


def show_progress(rr):

    rr._status_widget = widgets.Label('abcd', layout=widgets.Layout(width='100%'))
    rr._progress_widget = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    rr._cancel_button = widgets.Button(description='cancel')
    rr._cancel_button.on_click(lambda _: rr.cancel())
    ww = [rr._progress_widget]
    if rr.status == 'not started':
        rr._start_button = widgets.Button(description='start')
        ww.append(rr._start_button)
        def on_start(b):
            rr.start()
            rr._start_button.disabled = True
            rr._cancel_button.disabled = False
        rr._start_button.on_click(on_start)
        rr._cancel_button.disabled = True

    ww += [rr._cancel_button]
    rr._hbox = widgets.HBox(ww)
    rr._input = widgets.IntSlider(value=0, min=0, max=1000000000, step=100000000, description='Input:')
    rr._input_set = widgets.Button(description='set')
    rr._input_box = widgets.HBox([rr._input, rr._input_set])
    for w in [rr._input, rr._input_set]:
        w.disabled = True

    rr._input_box.disabled = True

    def get_input():
        val = VarWaiter()

        def set_val(_):
            val.value = rr._input.value
            for w in [rr._input, rr._input_set]:
                w.disabled = True
            rr._input_box.visible = False
            rr._input_set.on_click(None)

        rr._input_set.on_click(set_val)
        for w in [rr._input, rr._input_set]:
            w.disabled = False
        return val.wait_value()

    def work(rr):
        res = None

        while rr.status == 'not started':
            continue 

        while rr.status != 'canceled':
            if rr.empty():
                time.sleep(0.1)
                continue

            cmd = rr.receive()
            if cmd[0] == "result":
                res = str(cmd[1][0])
                break

            elif cmd[0] == "status":
                rr._status_widget.value = cmd[1][0]

            elif cmd[0] == "set":
                if "progress" in cmd[2]:
                    rr._progress_widget.value = cmd[2]["progress"]
                if "status" in cmd[2]:
                    rr._status_widget.value = cmd[2]["status"]

            elif cmd[0] == "get":
                if cmd[1] == "input_int":
                    rr.send("input_int", get_input())

        rr._hbox.close()
        del rr._hbox
        rr._input_box.close()
        del rr._input_box

        if res is not None:
            if len(res) > 200: res = res[0:200] + '...'
            rr._status_widget.value = "result is ready:\n" + res
        else:
            rr._status_widget.value = "process canceled"
            print("process canceled")

    rr._display_thread = threading.Thread(target=work, args=(rr,))
    display(rr._status_widget)
    display(rr._hbox)
    display(rr._input_box)
    rr._display_thread.start()
