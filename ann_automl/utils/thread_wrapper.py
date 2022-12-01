# wrapper around object such that all object's methods are executed
# in a unique separate thread where object was created.
# For each method of the object a special method is created
# that executes the original method in a thread where object was created
# For each property of the object a special property is created
# that executes the original property in a thread where object was created
import sys
import threading
import queue


def _request_process_loop(obj_class, obj_args, obj_kwargs, request_queue, response_queue):
    """ The main loop of the request process in a separate thread. """
    obj = obj_class(*obj_args, **obj_kwargs)
    response_queue.put(obj)
    with open(f'{obj_class.__name__}_log.txt', 'w') as f:
        while True:
            try:
                request = request_queue.get()
                if request is None:
                    print("Request process loop: received None request, exiting", file=f, flush=True)
                    break
                print("Request process loop: received request", file=f, flush=True)
                result = request()
                response_queue.put(result)
            except Exception as e:
                print(f"Exception in request process loop: {e}", file=sys.stderr)
                print(f"Exception in request process loop: {e}", file=f)
                raise e
    del obj


class ObjectWrapper:
    def __init__(self, cls, *args, **kwargs):
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = threading.Thread(target=_request_process_loop,
                                        args=(cls, args, kwargs, self._request_queue, self._response_queue))
        self._thread.start()
        self._obj = self._response_queue.get()

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                def request():
                    return attr(*args, **kwargs)
                self._request_queue.put(request)
                return self._response_queue.get()
            return wrapper
        return attr

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            def request():
                setattr(self._obj, name, value)
            self._request_queue.put(request)
            self._response_queue.get()

    def join_thread(self):
        self._obj = None
        self._request_queue.put(None)
        self._thread.join()
        self._thread = None

    def __del__(self):
        if self._thread is not None:
            self._thread.close()
