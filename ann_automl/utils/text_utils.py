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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Helper to display progress bar, source from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    prev_percent = f"{100 * ((iteration-1) / float(total)):.{decimals}f}"
    prev_filled_length = int(length * (iteration-1) // total)
    if prev_filled_length == filled_length and prev_percent == percent and total > iteration > 1:
        return
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
