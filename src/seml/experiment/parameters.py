import itertools
import random
import uuid
from typing import DefaultDict

from seml.utils import unflatten
from seml.utils.errors import ConfigError


def sample_random_configs(random_config, samples=1, seed=None):
    """
    Sample random configurations from the specified search space.

    Parameters
    ----------
    random_config: dict
        dict where each key is a parameter and the value defines how the random sample is drawn. The samples will be
        drawn using the function sample_parameter.
    samples: int
        The number of samples to draw per parameter
    seed: int or None
        The seed to use when drawing the parameter value. Defaults to None.

    Returns
    -------
    random_configurations: list of dicts
        List of dicts, where each dict gives a value for all parameters defined in the input random_config dict.

    """

    if len(random_config) == 0:
        return [{}]

    rdm_keys = [k for k in random_config.keys() if k not in ['samples', 'seed']]
    random_config = {k: random_config[k] for k in rdm_keys}
    random_parameter_dicts = unflatten(random_config, levels=-1)
    random_samples = [
        sample_parameter(random_parameter_dicts[k], samples, seed, parent_key=k)
        for k in random_parameter_dicts.keys()
    ]
    random_samples = dict([sub for item in random_samples for sub in item])
    random_configurations = [
        {k: v[ix] for k, v in random_samples.items()} for ix in range(samples)
    ]

    return random_configurations


def sample_parameter(parameter, samples, seed=None, parent_key=''):
    """
    Generate random samples from the specified parameter.

    The parameter types are inspired from https://github.com/hyperopt/hyperopt/wiki/FMin. When implementing new types,
    please make them compatible with the hyperopt nomenclature so that we can switch to hyperopt at some point.

    Parameters
    ----------
    parameter: dict
        Defines the type of parameter. Dict must include the key "type" that defines how the parameter will be sampled.
        Supported types are
            - choice: Randomly samples <samples> entries (with replacement) from the list in parameter['options']
            - uniform: Uniformly samples between 'min' and 'max' as specified in the parameter dict.
            - loguniform:  Uniformly samples in log space between 'min' and 'max' as specified in the parameter dict.
            - randint: Randomly samples integers between 'min' (included) and 'max' (excluded).
    samples: int
        Number of samples to draw for the parameter.
    seed: int
        The seed to use when drawing the parameter value. Defaults to None.
    parent_key: str
        The key to prepend the parameter name with. Used for nested parameters, where we here create a flattened version
        where e.g. {'a': {'b': 11}, 'c': 3} becomes {'a.b': 11, 'c': 3}

    Returns
    -------
    return_items: tuple(str, np.array or list)
        tuple of the parameter name and a 1-D list/array of the samples drawn for the parameter.

    """
    import numpy as np

    if 'type' not in parameter:
        raise ConfigError(f'No type found in parameter {parameter}')
    return_items = []
    allowed_keys = ['seed', 'type']
    if 'seed' in parameter:
        np.random.seed(parameter['seed'])
    elif seed is not None:
        np.random.seed(seed)

    param_type = parameter['type']

    if param_type == 'choice':
        choices = parameter['options']
        allowed_keys.append('options')
        sampled_values = [random.choice(choices) for _ in range(samples)]
        return_items.append((parent_key, sampled_values))

    elif param_type == 'uniform':
        min_val = parameter['min']
        max_val = parameter['max']
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.random.uniform(min_val, max_val, samples)
        return_items.append((parent_key, sampled_values))

    elif param_type == 'loguniform':
        if parameter['min'] <= 0:
            raise ConfigError('Cannot take log of values <= 0')
        min_val = np.log(parameter['min'])
        max_val = np.log(parameter['max'])
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.exp(np.random.uniform(min_val, max_val, samples))
        return_items.append((parent_key, sampled_values))

    elif param_type == 'randint':
        min_val = int(parameter['min'])
        max_val = int(parameter['max'])
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.random.randint(min_val, max_val, samples)
        return_items.append((parent_key, sampled_values))

    elif param_type == 'randint_unique':
        min_val = int(parameter['min'])
        max_val = int(parameter['max'])
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.random.choice(
            np.arange(min_val, max_val), samples, replace=False
        )
        return_items.append((parent_key, sampled_values))

    elif param_type == 'parameter_collection':
        sub_items = [
            sample_parameter(
                v, parent_key=f'{parent_key}.{k}', seed=seed, samples=samples
            )
            for k, v in parameter['params'].items()
        ]
        return_items.extend([sub_item for item in sub_items for sub_item in item])

    else:
        raise ConfigError(f'Parameter type {param_type} not implemented.')

    if param_type != 'parameter_collection':
        extra_keys = set(parameter.keys()).difference(set(allowed_keys))
        if len(extra_keys) > 0:
            raise ConfigError(
                f"Unexpected keys in parameter definition. Allowed keys for type '{param_type}' are "
                f'{allowed_keys}. Unexpected keys: {extra_keys}'
            )
    return return_items


def generate_grid(parameter, parent_key=''):
    """
    Generate a grid of parameter values from the input configuration.

    Parameters
    ----------
    parameter: dict
        Defines the type of parameter. Options for parameter['type'] are
            - choice: Expects a list of options in paramter['options'], which will be returned.
            - range: Expects 'min', 'max', and 'step' keys with values in the dict that are used as
                     np.arange(min, max, step)
            - uniform: Generates the grid using np.linspace(min, max, num, endpoint=True)
            - loguniform: Uniformly samples 'num' points in log space (base 10) between 'min' and 'max'
            - parameter_collection: wrapper around a dictionary of parameters (of the types above); we call this
              function recursively on each of the sub-parameters.
    parent_key: str
        The key to prepend the parameter name with. Used for nested parameters, where we here create a flattened version
        where e.g. {'a': {'b': 11}, 'c': 3} becomes {'a.b': 11, 'c': 3}

    Returns
    -------
    return_items: tuple(str, tuple(list, str))
        Name of the parameter and tuple with list containing the grid values for this parameter and zip id.

    """
    import numpy as np

    if 'type' not in parameter:
        raise ConfigError(f'No type found in parameter {parameter}')

    param_type = parameter['type']
    allowed_keys = ['type', 'zip_id']

    return_items = []

    if param_type == 'choice':
        values = parameter['options']
        allowed_keys.append('options')
        return_items.append((parent_key, values))

    elif param_type == 'range':
        min_val = parameter['min']
        max_val = parameter['max']
        step = parameter['step']
        allowed_keys.extend(['min', 'max', 'step'])
        values = list(np.arange(min_val, max_val, step))
        return_items.append((parent_key, values))

    elif param_type == 'uniform':
        min_val = parameter['min']
        max_val = parameter['max']
        num = int(parameter['num'])
        allowed_keys.extend(['min', 'max', 'num'])
        values = list(np.linspace(min_val, max_val, num, endpoint=True))
        return_items.append((parent_key, values))

    elif param_type == 'loguniform':
        min_val = parameter['min']
        max_val = parameter['max']
        num = int(parameter['num'])
        allowed_keys.extend(['min', 'max', 'num'])
        values = np.logspace(np.log10(min_val), np.log10(max_val), num, endpoint=True)
        return_items.append((parent_key, values))

    elif param_type == 'parameter_collection':
        sub_items = [
            generate_grid(v, parent_key=f'{parent_key}.{k}')
            for k, v in parameter['params'].items()
        ]
        return_items.extend([sub_item for item in sub_items for sub_item in item])

    else:
        raise ConfigError(f'Parameter {param_type} not implemented.')

    if param_type != 'parameter_collection':
        extra_keys = set(parameter.keys()).difference(set(allowed_keys))
        if len(extra_keys) > 0:
            raise ConfigError(
                f"Unexpected keys in parameter definition. Allowed keys for type '{param_type}' are "
                f'{allowed_keys}. Unexpected keys: {extra_keys}'
            )

    zip_id = parameter['zip_id'] if 'zip_id' in parameter else uuid.uuid4()
    return_items = [(item[0], (item[1], zip_id)) for item in return_items]
    return return_items


def zipped_dict(input_dict):
    """Zips dictionaries of type:
    {
        'element1': (values, zip_id),
        ...
    }
    to
    {
        'zip_id1': {
            'element1': values
            'element2': values
        },
        ...
    }

    Args:
        input_dict (dict[str, tuple(list, str)]): unzipped dictionary

    Returns:
        dict[str, dict[str, list]]: zipped dictionary
    """
    # Zip by zip_id attribute
    zipped_dict = DefaultDict(dict)
    for k, (val, zip_id) in input_dict.items():
        zipped_dict[zip_id][k] = val

    # Check that parameters in within a bundle have the same number of configurations.
    for k, bundle in zipped_dict.items():
        if len({len(x) for x in bundle.values()}) != 1:
            raise ConfigError(
                f"Parameters with zip_id '{k}' have different number of configurations!"
            )
    return zipped_dict


def cartesian_product_zipped_dict(zipped_dict):
    """Compute the Cartesian product of the ziped input dictionary values.
    Parameters
    ----------
    zipped_dict: dict of dicts of lists

    Returns
    -------
    list of dicts
        Cartesian product of the lists in the input dictionary.

    """
    zip_lengths = {
        k: len(next(iter(bundle.values()))) for k, bundle in zipped_dict.items()
    }

    for idx in itertools.product(*[range(k) for k in zip_lengths.values()]):
        yield {
            key: values[i]
            for zip_id, i in zip(zipped_dict, idx)
            for key, values in zipped_dict[zip_id].items()
        }
