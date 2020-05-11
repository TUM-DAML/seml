import sys
import logging
import numpy as np
import yaml
import ast
import jsonpickle
import json
import os
from pathlib import Path

from seml.sources import import_exe
from seml.parameters import sample_random_configs, generate_grid, cartesian_product_dict
from seml.utils import merge_dicts, flatten, unflatten


def unpack_config(config):
    reserved_keys = ['grid', 'fixed', 'random']
    children = {}
    reserved_dict = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        if key not in reserved_keys:
            children[key] = value
        else:
            # value = munch.munchify(value)
            if key == 'random':
                if 'samples' not in value:
                    logging.error('Random parameters must specify "samples", i.e. the number of random samples.')
                    sys.exit(1)
                keys = [k for k in value.keys() if k not in ['seed', 'samples']]
                if 'seed' in value:
                    seed = value['seed']
                    rdm_dict = {
                        k: {'samples': value['samples'],
                            'seed': seed, **value[k]}
                        for k in keys
                    }
                else:
                    rdm_dict = {
                        k: {'samples': value['samples'], **value[k]}
                        for k in keys
                    }
                reserved_dict[key] = rdm_dict
            else:
                reserved_dict[key] = value
    return reserved_dict, children


def generate_configs(experiment_config):
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

    Returns
    -------
    all_configs: list of dicts
        Contains the individual combinations of the parameters.

    Examples
    -------
    ```yaml
    fixed:
      fixed_a: 10

    grid:
      grid_param_a:
        type: 'choice'
        options:
          - "grid_a"
          - "grid_b"

    random:
      samples: 3
      random_param_a:
        type: 'uniform'
        min: 0
        max: 1

    nested_1:
      fixed:
        fixed_b: 20

      grid:
        grid_param_b:
          type: 'choice'
          options:
            - "grid_c"
            - "grid_d"

      random:
        samples: 5
        random_param_b:
          type: 'uniform'
          min: 2
          max: 3

    nested_2:
      grid:
        grid_param_c:
          type: 'choice'
          options:
            - "grid_e"
            - "grid_f"
    ```
    returns 2*max{3,5}*2 + 2*3*2 = 32 configurations.

    """

    reserved, next_level = unpack_config(experiment_config)
    level_stack = [next_level]
    config_levels = [reserved]
    final_configs = []

    while len(level_stack) > 0:
        sub_config, sub_levels = unpack_config(level_stack.pop(0))
        config = merge_dicts(config_levels.pop(0), sub_config)

        if len(sub_levels) == 0:
            final_configs.append(config)

        for sub_name, sub_vals in sub_levels.items():
            level_stack.append(sub_vals)
            config_levels.append(config)

    all_configs = []
    for conf in final_configs:
        random_params = conf['random'] if 'random' in conf else {}
        fixed_params = flatten(conf['fixed']) if 'fixed' in conf else {}
        grid_params = conf['grid'] if 'grid' in conf else {}

        if len(random_params) > 0:
            all_num_samples = np.array([x['samples'] for x in random_params.values() if 'samples' in x])
            num_samples = np.max(all_num_samples)
            random_sampled = sample_random_configs(random_params, seed=None, samples=num_samples)

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
        logging.error(f"Found no Sacred experiment. Something is wrong in '{executable}'.")
        sys.exit(1)
    elif len(exps) > 1:
        logging.error(f"Found more than 1 Sacred experiment in '{executable}'. "
                      f"Can't check parameter configs. Disable via --no-config-check.")
        sys.exit(1)
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


def read_config(config_path):
    with open(config_path, 'r') as conf:
        config_dict = convert_values(yaml.load(conf, Loader=yaml.FullLoader))

    if "seml" not in config_dict:
        logging.error("Please specify a 'seml' dictionary in the experiment configuration.")
        sys.exit(1)

    seml_dict = config_dict['seml']
    del config_dict['seml']

    set_executable_and_working_dir(config_path, seml_dict)

    if "db_collection" in seml_dict:
        logging.warning("Specifying a the database collection in the config has been deprecated. "
                        "Please provide it via the command line instead.")
    if 'output_dir' in seml_dict:
        seml_dict['output_dir'] = os.path.abspath(os.path.realpath(seml_dict['output_dir']))

    if 'slurm' in config_dict:
        slurm_dict = config_dict['slurm']
        del config_dict['slurm']
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
    config_dir = os.path.abspath(os.path.dirname(config_path))

    working_dir = config_dir
    os.chdir(working_dir)
    if "executable" not in seml_dict:
        logging.error("Please specify an executable path for the experiment.")
        sys.exit(1)
    executable = seml_dict['executable']
    executable_relative_to_config = os.path.exists(executable)
    executable_relative_to_project_root = False
    if 'project_root_dir' in seml_dict:
        working_dir = os.path.abspath(os.path.realpath(seml_dict['project_root_dir']))
        seml_dict['use_uploaded_sources'] = True
        os.chdir(working_dir)  # use project root as base dir from now on
        executable_relative_to_project_root = os.path.exists(executable)
        del seml_dict['project_root_dir']  # from now on we use only the working dir
    else:
        seml_dict['use_uploaded_sources'] = False
        logging.warning("'project_root_dir' not defined in seml config. Source files will not be uploaded.")
    seml_dict['working_dir'] = working_dir
    if not (executable_relative_to_config or executable_relative_to_project_root):
        logging.error(f"Could not find the executable.")
        exit(1)
    executable = os.path.abspath(executable)
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
