import numpy
import pickle
import codecs

def dump64(x):
    return codecs.encode(pickle.dumps(x), "base64")


def load64(dummy, s):
    return pickle.loads(codecs.decode(s, 'base64'))


def getobj(module, obj):
    if module == '__main__':
        return eval(obj)
    v = {}
    exec(f'from {v} import {obj}', v)
    return v[obj]


class ReprWrapper:
    def __init__(self, s):
        self._s = s

    def __repr__(self):
        return self._s

    def __str__(self):
        return self._s


def convert(x):
    if x is None:
        return x
    if type(x) in {int, complex, float, numpy.ndarray, bool, str, bytes, bytearray} or isinstance(x, numpy.generic):
        return x
    if type(x) in {list, tuple, set}:
        return type(x)([convert(y) for y in x])
    if type(x) is dict:
        return {convert(k):convert(v) for k,v in x.items()}
    if callable(x) and hasattr(x, '__module__') and hasattr(x, '__name__'):
        if x.__module__ == '__main__':
            return ReprWrapper(x.__name__)
        else:
            return ReprWrapper(f'{x.__module__}.{x.__name__}')
    if hasattr(x, '__unparse__'):
        return ReprWrapper(x.__unparse__())
    return ReprWrapper(f"load64('''{x}''',{dump64(x)})")


