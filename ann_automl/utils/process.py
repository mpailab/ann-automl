import sys
import threading
import traceback
from functools import wraps


class PDelayed(object):
    """ Class to communicate with the asynchronous task """
    class State:
        """ State of the task """
        status = 'not started'
        progress = 0
        state = None
        result = None
        request = None

    def __init__(self):
        self._value = None      # result of the task
        self._canceled = False  # True if the task is canceled
        self._finished = False  # finished or cancelled

        self._stop = False    # flag to stop the task
        self._cancel = False  # flag to cancel the task

        self._th = None   # thread to run the task
        self._exn = None  # exception raised by the task

        self._state = PDelayed.State()  # state of the task
        self.lock = threading.RLock()   # lock to access the state from different threads
        self.handlers = {}  # map request names to functions
        self.on_finish = None  # function to call when the task is finished

    def finish(self):
        """ Finish the task """
        with self.lock:
            self._finished = True
            if self.on_finish is not None:
                self.on_finish(self)

    def set_handler(self, name, f):
        """ Set a handler for a request

        Parameters
        ----------
        name: str
            Name of the request
        f: Callable
            Function to handle the request
        """
        self.handlers[name] = f

    def prepare(self, f, args, kwargs):
        """ Prepare the task to run with the given function and arguments.
        After preparation, to run the task, call start().

        Parameters
        ----------
        f: Callable
            Function to run
        args: tuple
            Positional arguments
        kwargs: dict
            Keyword arguments
        """
        self._th = threading.Thread(target=f, args=args, kwargs=kwargs)

    def start(self):
        """ Start the task """
        if self._th is None:
            raise Exception("no function to run")
        elif self._th.is_alive():
            raise Exception("already running")
        self._th.start()

    def get(self, raise_exn=True):
        """ Get the result of the task.

        Parameters
        ----------
        raise_exn: bool
            If True, raise an exception if the task failed with an exception.

        Returns
        -------
        result: object
            Result of the task or None if the task is not finished or cancelled or failed.
        """
        with self.lock:
            if self._exn is not None:
                if raise_exn:
                    raise self._exn
                else:
                    return None
            return self._state.result

    value = property(get)

    def cancel(self):
        """ Send a cancel request to the task (task may not cancel immediately) """
        with self.lock:
            if not self._canceled and not self.ready:
                self._stop = True
                self._cancel = True
                self._canceled = True
        return self

    def stop(self):
        """ Send a stop request to the task (task may not stop immediately) """
        with self.lock:
            if not self._canceled and not self.ready:
                self._stop = True
                self._cancel = False
        return self

    def wait(self, timeout=None):
        """ Wait for the task to finish or be cancelled.

        Parameters
        ----------
        timeout: float
            Timeout in seconds. If None, wait indefinitely.

        Returns
        -------
        result: object
            Result of the task or None if the task is not finished.
        """
        if not self._canceled and not self.ready:
            self._th.join(timeout)
            # self._value = self.result
        return self.value

    def update(self):
        return self._state

    @property
    def canceled(self):
        return self._canceled

    @property
    def ready(self):
        """ True if the task is finished or cancelled """
        with self.lock:
            return self._state.result is not None or self._exn is not None

    @property
    def runnung(self):
        """ True if the task is running """
        return not self.canceled and not self.ready

    @property
    def state(self):
        with self.lock:
            return self._state.state

    @property
    def status(self):
        with self.lock:
            return self._state.status

    @property
    def prorgess(self):
        return self._state.progress

    def print_status(self):
        with self.lock:
            st = self._state
        print(f"Status   : {st.status}")
        if str(st.status) != 'finished' and st.progress > 0:
            print(f"Progress : {st.progress:.2%}")

    def __del__(self):
        if self._th.isAlive():
            self.cancel()
            if self._th is not threading.current_thread():
                self._th.join()

    def __enter__(self):
        self.lock.__enter__()
        return self._state

    def __exit__(self, t, v, tb):
        self.lock.release()


class SetState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def progress(pr):
    return SetState(progress=pr)


# def pstate(st):
#     return SetState(state=st, progress=0)


def preturn(r):
    return SetState(result=r)


def pstatus(r):
    return SetState(status=r)


#def pcall(func, *args, **kwargs):
#    return SetState(request=(func, args, kwargs))


# thread-local variable
thread_vars = threading.local()
thread_vars.pdelayed = None


def pstate():
    return thread_vars.pdelayed


def set_pstate(**kwargs):
    pst = pstate()
    if pst is not None:
        with pstate() as st:
            for k, v in kwargs.items():
                setattr(st, k, v)


class NoHandlerError(Exception):
    pass


def request(name, *args, raise_exn=True, **kwargs):
    """ Send a request from the task to some other thread.

    Parameters
    ----------
    name: str
        Name of the request
    *args:
        Positional arguments of handler function associated with the request
    raise_exn: bool
        If True, raise an exception if the request failed with an exception.
    **kwargs:
        Keyword arguments of handler function associated with the request

    Returns
    -------
    result: object
        Result of the request
    """
    pst = pstate()
    if pst is None:
        if raise_exn:
            raise NoHandlerError(f"No process descriptor for current task")
        else:
            return None
    handler = pstate().handlers.get(name, None)
    if handler is None:
        if raise_exn:
            raise NoHandlerError(f"Handler for request {name} not registered")
        else:
            return None
    return handler(*args, **kwargs)


def pcall(name, *args, **kwargs):
    """ Send a request from the task to some other thread.

    Parameters
    ----------
    name: str
        Name of the request
    *args:
        Positional arguments of handler function associated with the request
    **kwargs:
        Keyword arguments of handler function associated with the request

    Returns
    -------
    result: object
        Result of the request
    """
    return request(name, *args, **kwargs, raise_exn=False)


def get_pstate_value(name, default=None):
    pst = pstate()
    if pst is not None:
        with pstate() as st:
            return getattr(st, name, default)
    return default


class VarWaiter:
    """ Allows one or more threads to wait until a value is set by some thread. """

    def __init__(self, value=None):
        self._value = value
        self._was_set = False
        self._cond = threading.Condition()

    def wait_value(self, timeout=None):
        """ Wait until the value is set.

        Parameters
        ----------
        timeout : float
            The maximum time to wait for the value to be set. If None, wait indefinitely.

        Returns
        -------
        value : object
            The value that was set.
        """
        with self._cond:
            if not self._was_set:
                self._cond.wait(timeout)
            return self._value

    @property
    def value(self):
        """ The value that was set. Wait until the value is set if it has not been set yet. """
        return self.wait_value()

    @value.setter
    def value(self, v):
        with self._cond:
            if self._was_set:
                raise Exception("Value already set")
            self._value = v
            self._was_set = True
            self._cond.notify_all()


def process(f):
    """ Decorator for a function to be executed in a separate thread.
    Function with this decorator returns PDelayed object that can be used to communicate with the task.
    """

    @wraps(f)
    def g(*args, start=True, handlers=None, output_context=None, **kwargs):
        d = PDelayed()
        d.handlers = handlers or {}

        def pp():
            thread_vars.pdelayed = d
            try:
                d._state.result = f(*args, **kwargs)
            except Exception as e:
                d._exn = e
                if output_context is not None:
                    with output_context:
                        print("Error occured during thread execution:")
                        traceback.print_exc()
                else:
                    print("Error occured during thread execution:")
                    traceback.print_exc()
            d.finish()

        def qq():
            if output_context is not None:
                with output_context:
                    pp()
            else:
                pp()

        d.prepare(pp, args=(), kwargs={})
        if start:
            d.start()
        return d

    return g
