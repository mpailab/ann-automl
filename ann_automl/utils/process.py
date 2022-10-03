import threading
from functools import wraps


# def lockobj(f):
#    @wraps(f)
#    def g(self,*args,**kwargs):
#        with self.lock:
#            return f(self,*args,**kwargs)
#    return g

class PDelayed(object):
    class State:
        status = 'not started'
        progress = 0
        state = None
        result = None
        request = None

    def __init__(self):
        self._value = None
        self._canceled = False
        self._finished = False

        self._stop = False
        self._cancel = False

        self._th = None
        self._exn = None

        self._state = PDelayed.State()
        self.lock = threading.RLock()
        self.handlers = {}  # map request names to functions

    def set_handler(self, name, f):
        self.handlers[name] = f

    def prepare(self, f, args, kwargs):
        self._th = threading.Thread(target=f, args=args, kwargs=kwargs)

    def start(self):
        if self._th is None:
            raise Exception("no function to run")
        elif self._th.isAlive():
            raise Exception("already running")
        self._th.start()

    def get(self, raise_exn=True):
        with self.lock:
            if self._exn is not None:
                if raise_exn:
                    raise self._exn
                else:
                    return None
            return self._state.result

    value = property(get)

    def cancel(self):
        with self.lock:
            if not self._canceled and not self.ready:
                self._stop = True
                self._cancel = True
                self._canceled = True
        return self

    def stop(self):
        with self.lock:
            if not self._canceled and not self.ready:
                self._stop = True
                self._cancel = False
        return self

    def wait(self, timeout=None):
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
        with self.lock:
            return self._state.result is not None or self._exn is not None

    @property
    def runnung(self):
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


def pcall(func, *args, **kwargs):
    return SetState(request=(func, args, kwargs))


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


def request(name, *args, raise_exn=True, **kwargs):
    handler = pstate().handlers.get(name, None)
    if handler is None:
        if raise_exn:
            raise Exception(f"Handler for request {name} not registered")
        else:
            return None
    return handler(*args, **kwargs)


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
    @wraps(f)
    def g(*args, start=True, handlers=None, **kwargs):
        d = PDelayed()
        d.handlers = handlers or {}

        def pp():
            thread_vars.pdelayed = d
            try:
                d._state.result = f(*args, **kwargs)
            except Exception as e:
                d._exn = e
            # for a in f(*args, **kwargs):
            #     # print(d.lock.locked())
            #     with d.lock:
            #         for field in ['progress', 'state', 'result', 'status', 'request']:
            #             if hasattr(a, field):
            #                 setattr(d._state, field, getattr(a, field))
            #         if hasattr(a, 'result'):
            #             break
            #         if d._stop:
            #             if not d._cancel:
            #                 if not hasattr(d._state, 'result') and hasattr(d._state, 'state'):
            #                     d._state.result = d._state.state
            #                 else:
            #                     d._state.result = 'Stopped'
            #             break

        d.prepare(pp, args=(), kwargs={})
        if start:
            d.start()
        return d

    return g
