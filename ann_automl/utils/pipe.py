from multiprocess import Process, Queue 
from functools import wraps
import multiprocessing_utils
import queue

class Pipe(object):

    def __init__(self, f, que_in, que_out):
        self._que_in = que_in
        self._que_out = que_out
        self._process = Process(target=f, args=(), kwargs={})
        self.status = "not started"

    def __del__(self):
        if self._process.is_alive():
            self._process.join()

    def start(self):
        if self._process.is_alive():
            raise Exception("already running")
        self._process.start()
        self.status = "started"

    def cancel(self):
        self.status = "canceled"

    @property
    def active(self):
        return self._process is not None

    def que_in(self):
        return self._que_in

    def que_out(self):
        return self._que_out

    def empty(self):
        return self._que_in.empty() and self._que_out.empty()

    def send(self, cmd, *args, **kwargs):
        self._que_in.put((cmd, args, kwargs))

    def receive(self):
        return self._que_out.get(True, 1)
        

# multiprocessing-local variable
process_vars = multiprocessing_utils.local()
process_vars.que_in = None
process_vars.que_out = None


def send(cmd, *args, **kwargs):
    que = process_vars.que_out
    if que is not None:
        que.put((cmd, args, kwargs))


def receive(cmd):
    if process_vars.que_in is not None and process_vars.que_out is not None:
        process_vars.que_out.put(("get", cmd))
        while True:
            try:
                request = process_vars.que_in.get(True, 1)
                if request[0] == cmd:
                    return request[1][0]
            except queue.Empty:
                continue
            break
    return None


def pipe(f):
    @wraps(f)
    def g(*args, start=True, **kwargs):
        que_in = Queue()
        que_out = Queue()

        def pp():
            process_vars.que_in = que_in
            process_vars.que_out = que_out
            try:
                res = f(*args, **kwargs)
            except Exception as e:
                res = e
            send("result", res)

        p = Pipe(pp, que_in, que_out)
        if start:
            p.start()
        return p

    return g
