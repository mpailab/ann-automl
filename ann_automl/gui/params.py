from copy import copy, deepcopy

import warnings

from ..core.nnfuncs import nn_hparams, tune_hparams


def update_params_deps(params):
    deps = {}
    for k, v in params.items():
        # capitalize first letter of title field
        if 'title' not in v:
            v['title'] = k
            warnings.warn(f'No title for parameter {k}')
        if 'default' not in v:
            raise ValueError(f'No default for parameter {k}')
        # check that for int and float types there is step and range fields
        if v['type'] in ['int', 'float']:
            if 'step' not in v:
                raise ValueError(f'No step for parameter {k}')
            if 'range' not in v:
                raise ValueError(f'No range for parameter {k}')

        v['title'] = v['title'][0].upper() + v['title'][1:]
        if 'values' in v and isinstance(v['values'], dict):
            for k2, v2 in v['values'].items():
                if isinstance(v2, dict) and 'params' in v2:
                    for p in v2['params']:
                        deps.setdefault(p, {}).setdefault(k, set()).add(k2)
    for p, d in deps.items():
        if p not in params:
            raise ValueError(f'Unknown dependent parameter {p}')
        if params[p].get('cond', False) is not True:
            if params[p].get('cond', False) is False:
                raise ValueError(f'Parameter {p} is not conditional')
            else:
                raise ValueError(f"'cond' field for parameter {p} is not boolean")
        params[p]['cond'] = list(d.items())
    return params


all_hparams = {
    'train': update_params_deps(deepcopy(nn_hparams)),
    'tune': update_params_deps(deepcopy(tune_hparams)),
}


param_groups = {
    'Learning': ['train.epochs', 'train.optimizer', 'train.learning_rate', 'train.decay', 'train.activation',
                 'train.loss', 'train.metrics', 'train.dropout', 'train.kernel_initializer',
                 'train.bias_initializer',
                 'train.kernel_regularizer', 'train.bias_regularizer', 'train.activity_regularizer',
                 'train.kernel_constraint', 'train.bias_constraint'],
    'Optimizer': ['train.nesterov', 'train.centered', 'train.amsgrad',
                  'train.momentum', 'train.rho', 'train.epsilon',
                  'train.beta_1', 'train.beta_2'],
    'Tune': ['tune.method', 'tune.radius', 'tune.grid_metric', 'tune.start_point']
}


def widget_type(param_description):
    if 'widget' in param_description.get('gui',{}):
        return param_description['gui']['widget']
    if 'values' in param_description:
        return 'Select'
    if 'range' in param_description:
        return 'Slider'
    if param_description['type'] == 'bool':
        return 'Checkbox'
    if param_description['type'] == 'str':
        return 'Text'
    raise ValueError(f'Unknown widget type for {param_description}')


def create_params_dict(params):
    result = {}
    for group, params_list in params.items():
        for param in params_list:
            pfrom, pname = param.split('.')
            pvalue = copy(all_hparams[pfrom][pname])
            pvalue['gui'] = {'group': group, 'widget': widget_type(pvalue)}
            pvalue['param_from'] = pfrom
            pvalue['param_key'] = pname
            result[param] = pvalue
            if 'cond' in pvalue:
                pvalue['cond'] = [(f"{pfrom}.{p}", v) for p, v in pvalue['cond']]
    return result


hyperparameters = create_params_dict(param_groups)
