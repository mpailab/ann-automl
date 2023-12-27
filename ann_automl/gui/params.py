from copy import copy, deepcopy

import warnings

from ..core.nnfuncs import nn_hparams, nns_hparams, nnd_hparams, tune_hparams


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
    'segmentation': update_params_deps(deepcopy(nns_hparams)),
    'detection': update_params_deps(deepcopy(nnd_hparams)),
    'tune': update_params_deps(deepcopy(tune_hparams)),
}


param_groups = {
    'train': ['train.model_arch', 'train.transfer_learning',
              'train.epochs', 'train.optimizer', 'train.learning_rate', 'train.batch_size', 
              'train.activation',
              'train.loss', 'train.metrics', 'train.dropout',
              'train.early_stopping', 'train.min_delta', 'train.patience',
             ],
    'segmentation': ['segmentation.model_arch', 'segmentation.transfer_learning',
                    'segmentation.epochs', 'segmentation.optimizer', 'segmentation.learning_rate', 'segmentation.batch_size', 
                    'segmentation.activation',
                    'segmentation.loss', 'segmentation.metrics', 'segmentation.dropout',
                    'segmentation.early_stopping', 'segmentation.min_delta', 'segmentation.patience',
             ],
    'detection': ['detection.model_arch', 'detection.transfer_learning',
                    'detection.epochs', 'detection.optimizer', 'detection.learning_rate', 'detection.batch_size', 
                    'detection.activation',
                    'detection.metrics', 'detection.dropout',
                    'detection.early_stopping', 'detection.min_delta', 'detection.patience',
             ],
    'optimizer': ['train.nesterov', 'train.centered', 'train.amsgrad',
                  'train.momentum', 'train.rho', 'train.epsilon',
                  'train.beta_1', 'train.beta_2'],
    'tune': ['tune.method',
             'tune.tuned_params',
             'tune.radius', 'tune.grid_metric', 'tune.start_point']  #, 'tune.exact_category_match']
}


def widget_type(param_description):
    if 'widget' in param_description.get('gui',{}):
        return param_description['gui']['widget']
    if param_description['type'] == 'list':
        return 'MultiChoice'
    if 'values' in param_description:
        return 'Select'
    if 'range' in param_description:
        return 'Slider'
    if param_description['type'] == 'bool':
        return 'Checkbox'
    if param_description['type'] == 'str':
        return 'Text'
    if param_description['type'] == 'date':
        return 'Date'
    raise ValueError(f'Unknown widget type for {param_description}')


def create_params_dict(params):
    result = {}
    for group, params_list in params.items():
        for param in params_list:
            pfrom, pname = param.split('.')
            pvalue = copy(all_hparams[pfrom][pname])
            pvalue['gui'] = {
                'group': group, 
                'widget': widget_type(pvalue),
                'info': True
            }
            pvalue['param_from'] = pfrom
            pvalue['param_key'] = pname
            result[f'{pfrom}_{pname}'] = pvalue
            if 'cond' in pvalue:
                pvalue['cond'] = [(f"{pfrom}_{p}", v) for p, v in pvalue['cond']]
    return result


hyperparameters = create_params_dict(param_groups)
