import os
import datetime
import logging

from seml.database import get_max_in_collection, get_collection
from seml.config import remove_prepended_dashes, read_config, generate_configs, check_config
from seml.sources import upload_sources, get_git_info
from seml.utils import merge_dicts, s_if, make_hash
from seml.settings import SETTINGS
from seml.errors import ConfigError

States = SETTINGS.STATES


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
        if 'config_hash' in config:
            config_hash = config['config_hash']
            del config['config_hash']
            lookup_result = collection.find_one({'config_hash': config_hash})
        else:
            lookup_dict = {
                f'config.{key}': value for key, value in config.items()
            }

            lookup_result = collection.find_one(lookup_dict)

        if lookup_result is None:
            filtered_configs.append(config)

    return filtered_configs


def add_configs(collection, seml_config, slurm_config, configs, source_files=None,
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

    logging.info(f"Adding {len(configs)} configs to the database (batch-ID {batch_id}).")

    if source_files is not None:
        seml_config['source_files'] = source_files
    db_dicts = [{'_id': start_id + ix,
                 'batch_id': batch_id,
                 'status': States.STAGED[0],
                 'seml': seml_config,
                 'slurm': slurm_config,
                 'config': c,
                 'config_hash': make_hash(c),
                 'git': git_info,
                 'add_time': datetime.datetime.utcnow()}
                for ix, c in enumerate(configs)]

    collection.insert_many(db_dicts)


def add_experiments(db_collection_name, config_file, force_duplicates, overwrite_params=None, no_hash=False, no_sanity_check=False,
                    no_code_checkpoint=False):
    """
    Add configurations from a config file into the database.

    Parameters
    ----------
    db_collection_name: the MongoDB collection name.
    config_file: path to the YAML configuration.
    force_duplicates: if True, disable duplicate detection.
    overwrite_params: optional flat dictionary to overwrite parameters in all configs.
    no_hash: if True, disable hashing of the configurations for duplicate detection. This is much slower, so use only
        if you have a good reason to.
    no_sanity_check: if True, do not check the config for missing/unused arguments.
    no_code_checkpoint: if True, do not upload the experiment source code files to the MongoDB.

    Returns
    -------
    None
    """

    seml_config, slurm_config, experiment_config = read_config(config_file)

    # Use current Anaconda environment if not specified
    if 'conda_environment' not in seml_config:
        seml_config['conda_environment'] = os.environ.get('CONDA_DEFAULT_ENV')

    # Set Slurm config with default parameters as fall-back option
    slurm_config = merge_dicts(SETTINGS.SLURM_DEFAULT, slurm_config)

    # Check for and use sbatch options template
    sbatch_options_template = slurm_config.get('sbatch_options_template', None)
    if sbatch_options_template is not None:
        if sbatch_options_template not in SETTINGS.SBATCH_OPTIONS_TEMPLATES:
            raise ConfigError(f"sbatch options template '{sbatch_options_template}' not found in settings.py.")
        for k, v in SETTINGS.SBATCH_OPTIONS_TEMPLATES[sbatch_options_template].items():
            if k not in slurm_config['sbatch_options']:
                slurm_config['sbatch_options'][k] = v

    slurm_config['sbatch_options'] = remove_prepended_dashes(slurm_config['sbatch_options'])
    configs = generate_configs(experiment_config, overwrite_params=overwrite_params)
    collection = get_collection(db_collection_name)

    batch_id = get_max_in_collection(collection, "batch_id")
    if batch_id is None:
        batch_id = 1
    else:
        batch_id = batch_id + 1

    if seml_config['use_uploaded_sources'] and not no_code_checkpoint:
        uploaded_files = upload_sources(seml_config, collection, batch_id)
    else:
        uploaded_files = None
    del seml_config['use_uploaded_sources']

    if not no_sanity_check:
        check_config(seml_config['executable'], seml_config['conda_environment'], configs)

    path, commit, dirty = get_git_info(seml_config['executable'])
    git_info = None
    if path is not None:
        git_info = {'path': path, 'commit': commit, 'dirty': dirty}

    use_hash = not no_hash
    if use_hash:
        configs = [{**c, **{'config_hash': make_hash(c)}} for c in configs]

    if not force_duplicates:
        len_before = len(configs)

        # First, check for duplicates withing the experiment configurations from the file.
        if not use_hash:
            # slow duplicate detection without hashes
            unique_configs = []
            for c in configs:
                if c not in unique_configs:
                    unique_configs.append(c)
            configs = unique_configs
        else:
            # fast duplicate detection using hashing.
            configs_dict = {c['config_hash']: c for c in configs}
            configs = [v for k, v in configs_dict.items()]

        len_after_deduplication = len(configs)
        # Now, check for duplicate configurations in the database.
        configs = filter_experiments(collection, configs)
        len_after = len(configs)
        if len_after_deduplication != len_before:
            logging.info(f"{len_before - len_after_deduplication} of {len_before} experiment{s_if(len_before)} were "
                         f"duplicates. Adding only the {len_after_deduplication} unique configurations.")
        if len_after != len_after_deduplication:
            logging.info(f"{len_after_deduplication - len_after} of {len_after_deduplication} "
                         f"experiment{s_if(len_before)} were already found in the database. They were not added again.")

    # Create an index on the config hash. If the index is already present, this simply does nothing.
    collection.create_index("config_hash")
    # Add the configurations to the database with STAGED status.
    if len(configs) > 0:
        add_configs(collection, seml_config, slurm_config, configs, uploaded_files, git_info)
