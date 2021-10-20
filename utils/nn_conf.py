import yaml
import keras.layers


def get_conf(filename):
    with open(filename, 'r') as f:
        res = yaml.safe_load(f)
    if type(res) is not dict:
        raise TypeError(f"loading {filename}: dict expected, got {type(res)}")
    return res


def add_layers(y, layers, verbose=False):
    for layer in layers:
        tp = layer['Type']
        if not hasattr(keras.layers, tp):
            raise TypeError(f"Invalid keras layer type {tp}")
        args = {**layer}
        del args['Type']
        y = getattr(keras.layers, tp)(**args)(y)
        if verbose:
            print(f'added layer: keras.layers.{tp}('+', '.join(f'{k}={repr(v)}' for k, v in args.items())+')')
    return y


def add_layers_from_conf(y, key, conf, verbose=False):
    if type(conf) is str:
        # if filename specified, read configuration from file
        conf = get_conf(conf)
    return add_layers(y, conf[key], verbose=verbose)
