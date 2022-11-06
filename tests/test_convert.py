from ann_automl import utils
import ann_automl
import numpy


def test_convertor():
    obj = {'a': 3,
           'b': [1, 2.2, 3 + 5j, utils.convert],  # serialize list of different types
           'c': numpy.ndarray,  # serialize class type
           'd': numpy.array,    # serialize function
           2: {'4': 5, (0, 1): b"2364"}}

    print(repr(utils.convert(obj)))
    print(obj)
    print(eval(repr(utils.convert(obj))))
    assert eval(repr(utils.convert(obj))) == obj


if __name__ == '__main__':
    test_convertor()
