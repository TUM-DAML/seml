import itertools
import numpy as np


def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries.

    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.

    """

    return_dict = dict1.copy()
    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            else:
                return_dict[k] = dict2[k]

    return return_dict


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

    rdm_keys = [k for k in random_config.keys() if k not in ["samples", "seed"]]

    random_samples = {k: sample_parameter(random_config[k], samples, seed) for k in rdm_keys}
    random_configurations = [{k: v[ix] for k, v in random_samples.items()} for ix in range(samples)]

    return random_configurations


def sample_parameter(parameter, samples, seed=None):
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

    Returns
    -------
    samples: numpy array or list
        1-D list/array of the samples drawn for the parameter.

    """

    if "type" not in parameter:
        raise ValueError("No type found in parameter {}".format(parameter))

    if 'seed' in parameter:
        seed = parameter['seed']
    if seed is not None:
        np.random.seed(seed)
    param_type = parameter['type']

    if param_type == "choice":
        choices = parameter['options']
        sampled_values = np.random.choice(choices, replace=True, size=samples)

    elif param_type == "uniform":
        min_val = parameter['min']
        max_val = parameter['max']
        sampled_values = np.random.uniform(min_val, max_val, samples)

    elif param_type == "loguniform":
        if parameter['min'] <= 0:
            raise ValueError("Cannot take log of values <= 0")
        min_val = np.log(parameter['min'])
        max_val = np.log(parameter['max'])
        sampled_values = np.exp(np.random.uniform(min_val, max_val, samples))

    elif param_type == "randint":
        min_val = int(parameter['min'])
        max_val = int(parameter['max'])
        sampled_values = np.random.randint(min_val, max_val, samples)

    elif param_type == "randint_unique":
        min_val = int(parameter['min'])
        max_val = int(parameter['max'])
        sampled_values = np.random.choice(np.arange(min_val, max_val), samples, replace=False)

    else:
        raise NotImplementedError(f"Parameter type {param_type} not implemented.")

    return sampled_values


def generate_grid(parameter):
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

    Returns
    -------
    values: list
        List containing the grid values for this parameter.

    """
    if "type" not in parameter:
        raise ValueError("No type found in parameter {}".format(parameter))

    param_type = parameter['type']

    if param_type == "choice":
        values = parameter['options']

    elif param_type == "range":
        min_val = parameter['min']
        max_val = parameter['max']
        step = int(parameter['step'])
        values = list(np.arange(min_val, max_val, step))

    elif param_type == "uniform":
        min_val = parameter['min']
        max_val = parameter['max']
        num = int(parameter['num'])
        values = list(np.linspace(min_val, max_val, num, endpoint=True))

    elif param_type == "loguniform":
        min_val = parameter['min']
        max_val = parameter['max']
        num = int(parameter['num'])
        values = np.logspace(np.log10(min_val), np.log10(max_val), num, endpoint=True)
    else:
        raise NotImplementedError(f"Parameter {param_type} not implemented.")

    return values


def cartesian_product_dict(input_dict):
    """Compute the Cartesian product of the input dictionary values.
    Parameters
    ----------
    input_dict: dict of lists

    Returns
    -------
    list of dicts
        Cartesian product of the lists in the input dictionary.

    """

    keys = input_dict.keys()
    vals = input_dict.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
