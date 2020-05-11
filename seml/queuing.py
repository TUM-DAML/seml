import os
import datetime
import logging

from seml.database import get_max_in_collection, get_collection
from seml.config import remove_prepended_dashes, read_config, generate_configs, check_config
from seml.sources import upload_sources, get_git_info
from seml.utils import s_if, make_hash
from seml.settings import SETTINGS


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


def queue_configs(collection, seml_config, slurm_config, configs, source_files=None,
                  git_info=None):
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
    git_info: (Optional) dict containing information about the git repo status.

    Returns
    -------
    None

    """

    if len(configs) == 0:
        return

    start_id = get_max_in_collection(collection, "_id")
    if start_id is None:
        start_id = 1
    else:
        start_id = start_id + 1

    batch_id = get_max_in_collection(collection, "batch_id")
    if batch_id is None:
        batch_id = 1
    else:
        batch_id = batch_id + 1

    logging.info(f"Queueing {len(configs)} configs into the database (batch-ID {batch_id}).")

    if source_files is not None:
        seml_config['source_files'] = source_files
    db_dicts = [{'_id': start_id + ix,
                 'batch_id': batch_id,
                 'status': 'QUEUED',
                 'seml': seml_config,
                 'slurm': slurm_config,
                 'config': c,
                 'config_hash': make_hash(c),
                 'git': git_info,
                 'queue_time': datetime.datetime.utcnow()}
                for ix, c in enumerate(configs)]

    collection.insert_many(db_dicts)


def queue_experiments(db_collection_name, config_file, force_duplicates, no_hash=False, no_config_check=False):
    seml_config, slurm_config, experiment_config = read_config(config_file)

    # Use current Anaconda environment if not specified
    if 'conda_environment' not in seml_config:
        if 'CONDA_DEFAULT_ENV' in os.environ:
            seml_config['conda_environment'] = os.environ['CONDA_DEFAULT_ENV']
        else:
            seml_config['conda_environment'] = None

    # Set Slurm config with default parameters as fall-back option

    for k, v in SETTINGS.SLURM_DEFAULT['sbatch_options'].items():
        if k not in slurm_config['sbatch_options']:
            slurm_config['sbatch_options'][k] = v
    del SETTINGS.SLURM_DEFAULT['sbatch_options']
    for k, v in SETTINGS.SLURM_DEFAULT.items():
        if k not in slurm_config:
            slurm_config[k] = v

    slurm_config['sbatch_options'] = remove_prepended_dashes(slurm_config['sbatch_options'])
    collection = get_collection(db_collection_name)
    configs = generate_configs(experiment_config)

    batch_id = get_max_in_collection(collection, "batch_id")
    if batch_id is None:
        batch_id = 1
    else:
        batch_id = batch_id + 1

    if seml_config['use_uploaded_sources']:
        uploaded_files = upload_sources(seml_config, collection, batch_id)
    else:
        uploaded_files = None

    if not no_config_check:
        check_config(seml_config['executable'], seml_config['conda_environment'], configs)

    path, commit, dirty = get_git_info(seml_config['executable'])
    git_info = None
    if path is not None:
        git_info = {'path': path, 'commit': commit, 'dirty': dirty}

    if not force_duplicates:
        len_before = len(configs)
        use_hash = not no_hash
        configs = filter_experiments(collection, configs, use_hash=use_hash)
        len_after = len(configs)
        if len_after != len_before:
            logging.info(f"{len_before - len_after} of {len_before} experiment{s_if(len_before)} were already found "
                         f"in the database. They were not added again.")

    # Create an index on the config hash. If the index is already present, this simply does nothing.
    collection.create_index("config_hash")
    # Add the configurations to the database with QUEUED status.
    if len(configs) > 0:
        queue_configs(collection, seml_config, slurm_config, configs, uploaded_files, git_info)
