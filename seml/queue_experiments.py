import os
import sys
import datetime
import numpy as np
import pymongo
import importlib
import sys
from pathlib import Path
import logging

from seml.database_utils import make_hash, upload_source_file
from seml import parameter_utils as utils
from seml import database_utils as db_utils
from seml.misc import get_default_slurm_config, s_if, unflatten, flatten, import_exe, is_local_source



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
                if not 'samples' in value:
                    raise ValueError('Random parameters must specify "samples", i.e. the number of random samples.')
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
        config = utils.merge_dicts(config_levels.pop(0), sub_config)

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
            random_sampled = utils.sample_random_configs(random_params, seed=None, samples=num_samples)

        grids = [utils.generate_grid(v, parent_key=k) for k,v in grid_params.items()]
        grid_configs = dict([sub for item in grids for sub in item])
        grid_product = list(utils.cartesian_product_dict(grid_configs))

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


def filter_experiments(collection, configurations, use_hash=True):
    """Check database collection for already present entries.

    Check the database collection for experiments that have the same configuration.
    Remove the corresponding entries from the input list of configurations to prevent
    re-running the experiments.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    configurations: list of dicts
        Contains the individual parameter configurations.
    use_hash: bool (default: True)
        Whether to use the hash of the config dictionary to perform a faster duplicate check.

    Returns
    -------
    filtered_configs: list of dicts
        No longer contains configurations that are already in the database collection.

    """

    filtered_configs = []
    for config in configurations:
        if use_hash:
            config_hash = make_hash(config)
            lookup_result = collection.find_one({'config_hash': config_hash})
        else:
            lookup_dict = {
                f'config.{key}': value for key, value in config.items()
            }

            lookup_result = collection.find_one(lookup_dict)

        if lookup_result is None:
            filtered_configs.append(config)

    return filtered_configs


def queue_configs(collection, seml_config, slurm_config, configs, source_files=None):
    """Put the input configurations into the database.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    seml_config: dict
        Configuration for the SEML library.
    slurm_config: dict
        Settings for the Slurm job. See `start_experiments.start_slurm_job` for details.
    configs: list of dicts
        Contains the parameter configurations.
    source_files: (optional) list of tuples
        Contains the uploaded source files corresponding to the batch. Entries are of the form
        (object_id, relative_path)

    Returns
    -------
    None

    """

    if len(configs) == 0:
        return

    start_id = db_utils.get_max_value(collection, "_id")
    if start_id is None:
        start_id = 1
    else:
        start_id = start_id + 1

    batch_id = db_utils.get_max_value(collection, "batch_id")
    if batch_id is None:
        batch_id = 1
    else:
        batch_id = batch_id + 1

    print(f"Queueing {len(configs)} configs into the database (batch-ID {batch_id}).")

    if source_files is not None:
        seml_config['source_files'] = source_files
    db_dicts = [{'_id': start_id + ix,
                 'batch_id': batch_id,
                 'status': 'QUEUED',
                 'seml': seml_config,
                 'slurm': slurm_config,
                 'config': c,
                 'config_hash': make_hash(c),
                 'queue_time': datetime.datetime.utcnow()}
                for ix, c in enumerate(configs)]

    collection.insert_many(db_dicts)


def get_imported_files(executable, root_dir):
    exe_path = os.path.abspath(executable)
    sys.path.insert(0, os.path.dirname(exe_path))
    exp_module = importlib.import_module(os.path.splitext(os.path.basename(executable))[0])
    del sys.path[0]
    root_path = os.path.abspath(root_dir)

    sources = set()
    for name, mod in sys.modules.items():
        if mod is None:
            continue
        if not getattr(mod, "__file__", False):
            continue
        filename = os.path.abspath(mod.__file__)
        if filename not in sources and is_local_source(filename, root_path):
            sources.add(filename)

    return sources


def check_sacred_config(executable, conda_env, configs):
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
        raise ValueError(f"Found no Sacred experiment. Something is wrong in '{executable}'.")
    elif len(exps) > 1:
        raise ValueError("Found more than 1 Sacred experiment. Can't check parameter configs. "
                         "Disable via --no-config-check.")
    exp = exps[0]

    empty_run = sacred.initialize.create_run(exp, exp.default_command, config_updates=None, named_configs=())

    captured_args = {
            sacred.utils.join_paths(cf.prefix, n)
            for cf in exp.captured_functions
            for n in cf.signature.arguments
    }

    for config in configs:
        config_added = {k: v for k, v in config.items() if k not in empty_run.config.keys()}
        config_flattened = {k for k, v in sacred.utils.iterate_flattened(config_added)}

        for conf in sorted(config_flattened):
            if not (set(sacred.utils.iter_prefixes(conf)) & captured_args):
                raise sacred.utils.ConfigAddedError(conf, config=config_flattened)

        options = empty_run.config.copy()
        options.update(config)
        empty_run.main_function.signature.construct_arguments((), {}, options, False)


def queue_experiments(config_file, force_duplicates, no_hash=False, no_config_check=False):
    seml_config, slurm_config, experiment_config = db_utils.read_config(config_file)

    # Use current Anaconda environment if not specified
    if 'conda_environment' not in seml_config and 'CONDA_DEFAULT_ENV' in os.environ:
        seml_config['conda_environment'] = os.environ['CONDA_DEFAULT_ENV']

    # Set Slurm config with default parameters as fall-back option
    default_slurm_config = get_default_slurm_config()
    for k, v in default_slurm_config['sbatch_options'].items():
        if k not in slurm_config['sbatch_options']:
            slurm_config['sbatch_options'][k] = v
    del default_slurm_config['sbatch_options']
    for k, v in default_slurm_config.items():
        if k not in slurm_config:
            slurm_config[k] = v

    slurm_config['sbatch_options'] = utils.remove_dashes(slurm_config['sbatch_options'])
    collection = db_utils.get_collection(seml_config['db_collection'])
    configs = generate_configs(experiment_config)

    batch_id = db_utils.get_max_value(collection, "batch_id")
    if batch_id is None:
        batch_id = 1
    else:
        batch_id = batch_id + 1

    if "project_root_dir" not in seml_config:
        logging.warning("'project_root_dir' not defined in seml config. Source files will not be uploaded.")
        uploaded_files = None
    else:
        uploaded_files = upload_sources(seml_config, collection, batch_id)

    if not no_config_check:
        check_sacred_config(seml_config['executable'], seml_config['conda_environment'], configs)

    if not force_duplicates:
        len_before = len(configs)
        use_hash = not no_hash
        configs = filter_experiments(collection, configs, use_hash=use_hash)
        len_after = len(configs)
        if len_after != len_before:
            print(f"{len_before - len_after} of {len_before} experiment{s_if(len_before)} were already found "
                  f"in the database. They were not added again.")

    # Create an index on the config hash. If the index is already present, this simply does nothing.
    collection.create_index("config_hash")
    # Add the configurations to the database with QUEUED status.
    if len(configs) > 0:
        queue_configs(collection, seml_config, slurm_config, configs, uploaded_files)


def upload_sources(seml_config, collection, batch_id):
    root_dir = os.path.abspath(os.path.expanduser(seml_config['project_root_dir']))
    sources = get_imported_files(seml_config['executable'], root_dir=root_dir)
    executable_abs = os.path.abspath(seml_config['executable'])
    executable_rel = Path(executable_abs).relative_to(root_dir)

    if not executable_abs in sources:
        raise ValueError(f"Executable {executable_abs} was not found in the sources to upload.")
    seml_config['executable_relative'] = str(executable_rel)

    uploaded_files = []
    for s in sources:
        file_id = upload_source_file(s, collection, batch_id)
        source_path = Path(s)
        uploaded_files.append((str(source_path.relative_to(root_dir)), file_id))
    return uploaded_files
