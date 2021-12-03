import logging
import numpy as np
import yaml
import ast
import jsonpickle
import json
import os
from pathlib import Path
import copy
from itertools import combinations

from seml.sources import import_exe
from seml.parameters import sample_random_configs, generate_grid, cartesian_product_dict
from seml.utils import Hashabledict, merge_dicts, flatten, unflatten
from seml.errors import ConfigError, ExecutableError
from seml.settings import SETTINGS

RESERVED_KEYS = ['grid', 'fixed', 'random']


def unpack_config(config):
    config = convert_parameter_collections(config)
    children = {}
    reserved_dict = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        if key not in RESERVED_KEYS:
            children[key] = value
        else:
            if key == 'random':
                if 'samples' not in value:
                    raise ConfigError('Random parameters must specify "samples", i.e. the number of random samples.')
                reserved_dict[key] = value
            else:
                reserved_dict[key] = value
    return reserved_dict, children


def extract_parameter_set(input_config: dict, key: str):
    flattened_dict = flatten(input_config.get(key, {}))
    keys = flattened_dict.keys()
    if key != 'fixed':
        keys = [".".join(k.split(".")[:-1]) for k in keys
                if flattened_dict[k] != 'parameter_collection']
    return set(keys)


def convert_parameter_collections(input_config: dict):
    flattened_dict = flatten(input_config)
    parameter_collection_keys = [k for k in flattened_dict.keys()
                                 if flattened_dict[k] == "parameter_collection"]
    if len(parameter_collection_keys) > 0:
        logging.warning("Parameter collections are deprecated. Use dot-notation for nested parameters instead.")
    while len(parameter_collection_keys) > 0:
        k = parameter_collection_keys[0]
        del flattened_dict[k]
        # sub1.sub2.type ==> # sub1.sub2
        k = ".".join(k.split(".")[:-1])
        parameter_collections_params = [param_key for param_key in flattened_dict.keys() if param_key.startswith(k)]
        for p in parameter_collections_params:
            if f"{k}.params" in p:
                new_key = p.replace(f"{k}.params", k)
                if new_key in flattened_dict:
                    raise ConfigError(f"Could not convert parameter collections due to key collision: {new_key}.")
                flattened_dict[new_key] = flattened_dict[p]
                del flattened_dict[p]
        parameter_collection_keys = [k for k in flattened_dict.keys()
                                     if flattened_dict[k] == "parameter_collection"]
    return unflatten(flattened_dict)


def standardize_config(config: dict):
    config = unflatten(flatten(config), levels=[0])
    out_dict = {}
    for k in RESERVED_KEYS:
        if k == "fixed":
            out_dict[k] = config.get(k, {})
        else:
            out_dict[k] = unflatten(config.get(k, {}), levels=[-1])
    return out_dict


def invert_config(config: dict):
    reserved_sets = [(k, set(config.get(k, {}).keys())) for k in RESERVED_KEYS]
    inverted_config = {}
    for k, params in reserved_sets:
        for p in params:
            l = inverted_config.get(p, [])
            l.append(k)
            inverted_config[p] = l
    return inverted_config


def detect_duplicate_parameters(inverted_config: dict, sub_config_name: str = None, ignore_keys: dict = None):
    if ignore_keys is None:
        ignore_keys = {'random': ('seed', 'samples')}

    duplicate_keys = []
    for p, l in inverted_config.items():
        if len(l) > 1:
            if 'random' in l and p in ignore_keys['random']:
                continue
            duplicate_keys.append((p, l))

    if len(duplicate_keys) > 0:
        if sub_config_name:
            raise ConfigError(f"Found duplicate keys in sub-config {sub_config_name}: "
                              f"{duplicate_keys}")
        else:
            raise ConfigError(f"Found duplicate keys: {duplicate_keys}")

    start_characters = set([x[0] for x in inverted_config.keys()])
    buckets = {k: {x for x in inverted_config.keys() if x.startswith(k)} for k in start_characters}

    if sub_config_name:
        error_str = (f"Conflicting parameters in sub-config {sub_config_name}, most likely "
                     "due to ambiguous use of dot-notation in the config dict. Found "
                     "parameter '{p1}' in dot-notation starting with other parameter "
                     "'{p2}', which is ambiguous.")
    else:
        error_str = (f"Conflicting parameters, most likely "
                     "due to ambiguous use of dot-notation in the config dict. Found "
                     "parameter '{p1}' in dot-notation starting with other parameter "
                     "'{p2}', which is ambiguous.")

    for k in buckets.keys():
        for p1, p2 in combinations(buckets[k], r=2):
            if p1.startswith(f"{p2}."):   # with "." after p2 to catch cases like "test" and "test1", which are valid.
                raise ConfigError(error_str.format(p1=p1, p2=p2))
            elif p2.startswith(f"{p1}."):
                raise ConfigError(error_str.format(p1=p1, p2=p2))


def generate_configs(experiment_config, overwrite_params=None):
    """Generate parameter configurations based on an input configuration.

    Input is a nested configuration where on each level there can be 'fixed', 'grid', and 'random' parameters.

    In essence, we take the cartesian product of all the `grid` parameters and take random samples for the random
    parameters. The nested structure makes it possible to define different parameter spaces e.g. for different datasets.
    Parameter definitions lower in the hierarchy overwrite parameters defined closer to the root.

    For each leaf configuration we take the maximum of all num_samples values on the path since we need to have the same
    number of samples for each random parameter.

    For each configuration of the `grid` parameters we then create `num_samples` configurations of the random
    parameters, i.e. leading to `num_samples * len(grid_configurations)` configurations.

    See Also `examples/example_config.yaml` and the example below.

    Parameters
    ----------
    experiment_config: dict
        Dictionary that specifies the "search space" of parameters that will be enumerated. Should be
        parsed from a YAML file.
    overwrite_params: Optional[dict]
        Flat dictionary that overwrites configs. Resulting duplicates will be removed.

    Returns
    -------
    all_configs: list of dicts
        Contains the individual combinations of the parameters.


    """

    reserved, next_level = unpack_config(experiment_config)
    reserved = standardize_config(reserved)
    if not any([len(reserved.get(k, {})) > 0 for k in RESERVED_KEYS]):
        raise ConfigError("No parameters defined under grid, fixed, or random in the config file.")
    level_stack = [('', next_level)]
    config_levels = [reserved]
    final_configs = []

    detect_duplicate_parameters(invert_config(reserved), None)

    while len(level_stack) > 0:
        current_sub_name, sub_vals = level_stack.pop(0)
        sub_config, sub_levels = unpack_config(sub_vals)
        if current_sub_name != '' and not any([len(sub_config.get(k, {})) > 0 for k in RESERVED_KEYS]):
            raise ConfigError(f"No parameters defined under grid, fixed, or random in sub-config {current_sub_name}.")
        sub_config = standardize_config(sub_config)
        config_above = config_levels.pop(0)

        inverted_sub_config = invert_config(sub_config)
        detect_duplicate_parameters(inverted_sub_config, current_sub_name)

        inverted_config_above = invert_config(config_above)
        redefined_parameters = set(inverted_sub_config.keys()).intersection(set(inverted_config_above.keys()))

        if len(redefined_parameters) > 0:
            logging.info(f"Found redefined parameters in sub-config '{current_sub_name}': {redefined_parameters}. "
                         f"Definitions in sub-configs override more general ones.")
            config_above = copy.deepcopy(config_above)
            for p in redefined_parameters:
                sections = inverted_config_above[p]
                for s in sections:
                    del config_above[s][p]

        config = merge_dicts(config_above, sub_config)

        if len(sub_levels) == 0:
            final_configs.append((current_sub_name, config))

        for sub_name, sub_vals in sub_levels.items():
            new_sub_name = f'{current_sub_name}.{sub_name}' if current_sub_name != '' else sub_name
            level_stack.append((new_sub_name, sub_vals))
            config_levels.append(config)

    all_configs = []
    for subconfig_name, conf in final_configs:
        conf = standardize_config(conf)
        random_params = conf.get('random', {})
        fixed_params = flatten(conf.get('fixed', {}))
        grid_params = conf.get('grid', {})

        if len(random_params) > 0:
            num_samples = random_params['samples']
            root_seed = random_params.get('seed', None)
            random_sampled = sample_random_configs(flatten(random_params), seed=root_seed, samples=num_samples)

        grids = [generate_grid(v, parent_key=k) for k, v in grid_params.items()]
        grid_configs = dict([sub for item in grids for sub in item])
        grid_product = list(cartesian_product_dict(grid_configs))

        with_fixed = [{**d, **fixed_params} for d in grid_product]
        if len(random_params) > 0:
            with_random = [{**grid, **random} for grid in with_fixed for random in random_sampled]
        else:
            with_random = with_fixed
        all_configs.extend(with_random)

    # Cast NumPy integers to normal integers since PyMongo doesn't like them
    all_configs = [{k: int(v) if isinstance(v, np.integer) else v
                    for k, v in config.items()}
                   for config in all_configs]

    if overwrite_params is not None:
        all_configs = [merge_dicts(config, overwrite_params) for config in all_configs]
        base_length = len(all_configs)
        # We use a dictionary instead a set because dictionary keys are ordered as of Python 3
        all_configs = list({Hashabledict(**config): None for config in all_configs})
        new_length = len(all_configs)
        if base_length != new_length:
            diff = base_length - new_length
            logging.warn(f'Parameter overwrite caused {diff} identical configs. Duplicates were removed.')

    all_configs = [unflatten(conf) for conf in all_configs]
    return all_configs


def check_config(executable, conda_env, configs):
    """Check if the given configs are consistent with the Sacred experiment in the given executable.

    Parameters
    ----------
    executable: str
        The Python file containing the experiment.
    conda_env: str
        The experiment's Anaconda environment.
    configs: list of dicts
        Contains the parameter configurations.

    Returns
    -------
    None

    """
    import sacred

    exp_module = import_exe(executable, conda_env)

    # Extract experiment from module
    exps = [v for k, v in exp_module.__dict__.items() if type(v) == sacred.Experiment]
    if len(exps) == 0:
        raise ExecutableError(f"Found no Sacred experiment. Something is wrong in '{executable}'.")
    elif len(exps) > 1:
        raise ExecutableError(f"Found more than 1 Sacred experiment in '{executable}'. "
                              f"Can't check parameter configs. Disable via --no-sanity-check.")
    exp = exps[0]

    empty_run = sacred.initialize.create_run(exp, exp.default_command, config_updates=None, named_configs=())

    captured_args = {
            sacred.utils.join_paths(cf.prefix, n)
            for cf in exp.captured_functions
            for n in cf.signature.arguments
    }

    for config in configs:
        config_added = {k: v for k, v in config.items() if k not in empty_run.config.keys()}
        config_flattened = {k for k, _ in sacred.utils.iterate_flattened(config_added)}

        # Check for unused arguments
        for conf in sorted(config_flattened):
            if not (set(sacred.utils.iter_prefixes(conf)) & captured_args):
                raise sacred.utils.ConfigAddedError(conf, config=config_added)

        # Check for missing arguments
        options = empty_run.config.copy()
        options.update(config)
        options.update({k: None for k in sacred.utils.ConfigAddedError.SPECIAL_ARGS})
        empty_run.main_function.signature.construct_arguments((), {}, options, False)


def restore(flat):
    """
    Restore more complex data that Python's json can't handle (e.g. Numpy arrays).
    Copied from sacred.serializer for performance reasons.
    """
    return jsonpickle.decode(json.dumps(flat), keys=True)


def _convert_value(value):
    """
    Parse string as python literal if possible and fallback to string.
    Copied from sacred.arg_parser for performance reasons.
    """

    try:
        return restore(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value


def convert_values(val):
    if isinstance(val, dict):
        for key, inner_val in val.items():
            val[key] = convert_values(inner_val)
    elif isinstance(val, list):
        for i, inner_val in enumerate(val):
            val[i] = convert_values(inner_val)
    elif isinstance(val, str):
        return _convert_value(val)
    return val


class YamlUniqueLoader(yaml.FullLoader):
    """
    Custom YAML loader that disallows duplicate keys

    From https://github.com/encukou/naucse_render/commit/658197ed142fec2fe31574f1ff24d1ff6d268797
    Workaround for PyYAML issue: https://github.com/yaml/pyyaml/issues/165
    This disables some uses of YAML merge (`<<`)
    """


def construct_mapping(loader, node, deep=False):
    """Construct a YAML mapping node, avoiding duplicates"""
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"Found duplicate keys: '{key}'")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


YamlUniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_mapping,
)


def read_config(config_path):
    with open(config_path, 'r') as conf:
        config_dict = convert_values(yaml.load(conf, Loader=YamlUniqueLoader))

    if "seml" not in config_dict:
        raise ConfigError("Please specify a 'seml' dictionary.")

    seml_dict = config_dict['seml']
    del config_dict['seml']

    for k in seml_dict.keys():
        if k not in SETTINGS.VALID_SEML_CONFIG_VALUES:
            raise ConfigError(f"{k} is not a valid value in the `seml` config block.")

    set_executable_and_working_dir(config_path, seml_dict)

    if 'output_dir' in seml_dict:
        seml_dict['output_dir'] = str(Path(seml_dict['output_dir']).expanduser().resolve())

    if 'slurm' in config_dict:
        slurm_dict = config_dict['slurm']
        del config_dict['slurm']

        for k in slurm_dict.keys():
            if k not in SETTINGS.VALID_SLURM_CONFIG_VALUES:
                raise ConfigError(f"{k} is not a valid value in the `slurm` config block.")

        return seml_dict, slurm_dict, config_dict
    else:
        return seml_dict, None, config_dict


def set_executable_and_working_dir(config_path, seml_dict):
    """
    Determine the working directory of the project and chdir into the working directory.
    Parameters
    ----------
    config_path: Path to the config file
    seml_dict: SEML config dictionary

    Returns
    -------
    None
    """
    config_dir = str(Path(config_path).expanduser().resolve().parent)

    working_dir = config_dir
    os.chdir(working_dir)
    if "executable" not in seml_dict:
        raise ConfigError("Please specify an executable path for the experiment.")
    executable = seml_dict['executable']
    executable_relative_to_config = os.path.exists(executable)
    executable_relative_to_project_root = False
    if 'project_root_dir' in seml_dict:
        working_dir = str(Path(seml_dict['project_root_dir']).expanduser().resolve())
        seml_dict['use_uploaded_sources'] = True
        os.chdir(working_dir)  # use project root as base dir from now on
        executable_relative_to_project_root = os.path.exists(executable)
        del seml_dict['project_root_dir']  # from now on we use only the working dir
    else:
        seml_dict['use_uploaded_sources'] = False
        logging.warning("'project_root_dir' not defined in seml config. Source files will not be saved in MongoDB.")
    seml_dict['working_dir'] = working_dir
    if not (executable_relative_to_config or executable_relative_to_project_root):
        raise ExecutableError(f"Could not find the executable.")
    executable = str(Path(executable).expanduser().resolve())
    seml_dict['executable'] = (str(Path(executable).relative_to(working_dir)) if executable_relative_to_project_root
                               else str(Path(executable).relative_to(config_dir)))


def remove_prepended_dashes(param_dict):
    new_dict = {}
    for k, v in param_dict.items():
        if k.startswith('--'):
            new_dict[k[2:]] = v
        elif k.startswith('-'):
            new_dict[k[1:]] = v
        else:
            new_dict[k] = v
    return new_dict
