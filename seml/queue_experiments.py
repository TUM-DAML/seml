import datetime
import numpy as np
import pymongo

import parameter_utils as utils
import database_utils as db_utils
from misc import s_if


def unpack_config(config):
    reserved_keys = ['grid', 'fixed', 'random']
    children = {}
    reserved_dict = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        if key not in reserved_keys:
            children[key] = value
        elif key == 'random':
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
        fixed_params = conf['fixed'] if 'fixed' in conf else {}
        grid_params = conf['grid'] if 'grid' in conf else {}

        all_num_samples = np.array([x['samples'] for x in random_params.values() if 'samples' in x])
        num_samples = np.max(all_num_samples)

        random_sampled = utils.sample_random_configs(random_params, seed=None, samples=num_samples)

        grid_configs = {k: utils.generate_grid(v) for k,v in grid_params.items()}
        grid_product = list(utils.cartesian_product_dict(grid_configs))

        with_fixed = [{**d, **fixed_params} for d in grid_product]
        with_random = [{**grid, **random} for grid in with_fixed for random in random_sampled]

        all_configs.extend(with_random)

    # Cast NumPy integers to normal integers since PyMongo doesn't like them
    all_configs = [{k: int(v) if isinstance(v, np.integer) else v
                    for k, v in config.items()}
                   for config in all_configs]
    return all_configs


def filter_experiments(collection, configurations):
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

    Returns
    -------
    filtered_configs: list of dicts
        No longer contains configurations that are already in the database collection.

    """

    filtered_configs = []

    for config in configurations:
        lookup_dict = {
            f'config.{key}': value for key, value in config.items()
        }

        lookup_result = collection.find_one(lookup_dict)

        if lookup_result is None:
            filtered_configs.append(config)

    return filtered_configs


def queue_configs(collection, tracking_config, configs):
    """Put the input configurations into the database.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    tracking_config: dict
        Configuration for the tracking library.
    configs: list of dicts
        Contains the parameter configurations.

    Returns
    -------
    None

    """

    if len(configs) == 0:
        return

    ndocs = collection.count_documents({})
    c = collection.find({}, {'_id': 1, 'batch_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    start_id = c.next()['_id'] + 1 if ndocs != 0 else 1
    c.rewind()
    b = c.sort('batch_id', pymongo.DESCENDING).limit(1)

    if ndocs != 0:
        b_next = b.next()
        batch_id = b_next['batch_id'] + 1 if "batch_id" in b_next else 1
    else:
        batch_id = 1

    print(f"Queueing {len(configs)} configs into the database (batch-ID {batch_id}).")

    db_dicts = [{'_id': start_id + ix,
                 'batch_id': batch_id,
                 'status': 'QUEUED',
                 'tracking': tracking_config,
                 'config': c,
                 'queue_time': datetime.datetime.utcnow()}
                for ix, c in enumerate(configs)]
    collection.insert_many(db_dicts)


def queue_experiments(config_file, force_duplicates):
    tracking_config, _, experiment_config = db_utils.read_config(config_file)
    collection = db_utils.get_collection(tracking_config['db_collection'])

    configs = generate_configs(experiment_config)

    if not force_duplicates:
        len_before = len(configs)
        configs = filter_experiments(collection, configs)
        len_after = len(configs)
        if len_after != len_before:
            print(f"{len_before - len_after} of {len_before} experiment{s_if(len_before)} were already found "
                  f"in the database. They were not added again.")

    # Add the configurations to the database with QUEUED status.
    if len(configs) > 0:
        queue_configs(collection, tracking_config, configs)
