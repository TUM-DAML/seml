import copy
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple

from seml.config import (check_config, generate_configs, generate_named_configs, read_config,
                         remove_prepended_dashes, resolve_configs, config_get_exclude_keys)
from seml.database import get_collection, get_max_in_collection
from seml.description import resolve_description
from seml.errors import ConfigError
from seml.settings import SETTINGS
from seml.sources import get_git_info, upload_sources
from seml.utils import Hashabledict, flatten, make_hash, merge_dicts, remove_keys_from_nested, s_if, unflatten

States = SETTINGS.STATES


def filter_experiments(collection: 'pymongo.collection.Collection', 
                       configurations: List[Dict],
                       configurations_unresolved: List[Dict],
                       use_hash: bool = True):
    """Check database collection for already present entries.

    Check the database collection for experiments that have the same configuration.
    Remove the corresponding entries from the input list of configurations to prevent
    re-running the experiments.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    configurations: List[Dict]
        Contains the individual parameter configurations.
    configurations_unresolved: List[Dict]
        Contains the individual parameter configurations before resolution via sacred.
    use_hash : bool
        Whether to use hashes (faster)

    Returns
    -------
    filtered_configs: list of dicts
        No longer contains configurations that are already in the database collection.

    """
        
    filtered_configs = []
    for config, config_unresolved in zip(configurations, configurations_unresolved):
        exclude_keys = config_get_exclude_keys(config, config_unresolved)
        if use_hash:
            lookup_result = collection.find_one({'config_hash': make_hash(config, exclude_keys)})
        else:
            lookup_dict = flatten({'config': remove_keys_from_nested(config, exclude_keys)})
            lookup_result = collection.find_one(unflatten(lookup_dict))
        if lookup_result is None:
            filtered_configs.append(config)

    return filtered_configs


def add_configs(
    collection: 'pymongo.collection.Collection', 
    seml_config: Dict, 
    slurm_config: Dict, 
    configs: List[Dict], 
    configs_unresolved: List[Dict], 
    source_files: Optional[List[Tuple]] = None,
    git_info: Optional[Dict] = None,
    resolve_descriptions: bool = True):
    """Put the input configurations into the database.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    seml_config: Dict
        Configuration for the SEML library.
    slurm_config: Dict
        Settings for the Slurm job. See `start_experiments.start_slurm_job` for details.
    configs: List[Dict]
        Contains the parameter configurations.
    configs_unresolved: List[Dict]
        Contains the parameter configurations before resolution.
    source_files: Optional[List[Tuple]]
        Contains the uploaded source files corresponding to the batch. Entries are of the form
        (object_id, relative_path)
    git_info: Optional[Dict]
        containing information about the git repo status.
    resolve_descriptions : bool, optional
        Whether to use omegaconf to resolve descriptions.
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
    db_dicts = [{'_id': start_id + idx,
                 'batch_id': batch_id,
                 'status': States.STAGED[0],
                 'seml': seml_config,
                 'slurm': slurm_config,
                 'config': c,
                 'config_unresolved' : c_unresolved,
                 'config_hash': make_hash(c, config_get_exclude_keys(c, c_unresolved)),
                 'git': git_info,
                 'add_time': datetime.datetime.utcnow()}
                for idx, (c, c_unresolved) in enumerate(zip(configs, configs_unresolved))]
    if resolve_descriptions:
        for db_dict in db_dicts:
            if 'description' in db_dict['seml']:
                db_dict['seml']['description'] = resolve_description(db_dict['seml']['description'], db_dict['config'])

    collection.insert_many(db_dicts)


def add_config_files(db_collection_name: str, 
                     config_files: List[str], 
                     force_duplicates: bool = False, 
                     overwrite_params: Optional[Dict] = None, 
                     no_hash: bool = False, 
                     no_sanity_check: bool = False,
                     no_code_checkpoint: bool = False,
                     resolve_descriptions: bool = True,):
    """Adds configuration files to the MongoDB

    Parameters
    ----------
    db_collection_name : str
        The collection to which to add
    config_files : List[str]
        A list of paths to configuration files (YAML) to add
    force_duplicates : bool, optional
        Whether to force adding configurations even if they are already in the MongoDB, by default False
    overwrite_params : Optional[Dict], optional
        Optional flat dict to override configuration values, by default None
    no_hash : bool, optional
        Whether to skip duplicate detection by computing hashes, which may result in slowdowns, by default False
    no_sanity_check : bool, optional
        Whether to skip feeding configuration values into the sacred experiment to detect unsupported or missing keys, by default False
    no_code_checkpoint : bool, optional
        Whether to not base the experiments on a copy of the current codebase, by default False
    resolve_descriptions : bool, optional
        Whether to use omegaconf to resolve experiment descriptions.
    """
    config_files = [os.path.abspath(file) for file in config_files]
    for config_file in config_files:
        add_config_file(db_collection_name, config_file, force_duplicates,
                        overwrite_params, no_hash, no_sanity_check,no_code_checkpoint,
                        resolve_descriptions=resolve_descriptions)


def assemble_slurm_config_dict(experiment_slurm_config: dict):
    """
    Realize inheritance for the slurm configuration, with the following relationship:
    Default -> Template -> Experiment

    Parameters
    ----------
    experiment_slurm_config: The slurm experiment configuration as returned by the function read_config

    Returns
    -------
    slurm_config

    """
    # Rename
    slurm_config = experiment_slurm_config
    # Assemble the Slurm config:
    # Basis config is the default config. This can be overridden by the sbatch_options_template.
    # And this in turn can be overridden by the sbatch config defined in the experiment .yaml file.
    slurm_config_base = copy.deepcopy(SETTINGS.SLURM_DEFAULT)

    # Check for and use sbatch options template
    sbatch_options_template = slurm_config.get('sbatch_options_template', None)
    if sbatch_options_template is not None:
        if sbatch_options_template not in SETTINGS.SBATCH_OPTIONS_TEMPLATES:
            raise ConfigError(f"sbatch options template '{sbatch_options_template}' not found in settings.py.")
        slurm_config_base['sbatch_options'] = merge_dicts(slurm_config_base['sbatch_options'],
                                                          SETTINGS.SBATCH_OPTIONS_TEMPLATES[sbatch_options_template])

    # Integrate experiment specific config
    slurm_config = merge_dicts(slurm_config_base, slurm_config)

    slurm_config['sbatch_options'] = remove_prepended_dashes(slurm_config['sbatch_options'])
    return slurm_config


def add_config_file(db_collection_name: str, 
                     config_file: str, 
                     force_duplicates: bool = False, 
                     overwrite_params: Optional[Dict] = None, 
                     no_hash: bool = False, 
                     no_sanity_check: bool = False,
                     no_code_checkpoint: bool = False,
                     resolve_descriptions: bool = True,):
    """Adds configuration files to the MongoDB

    Parameters
    ----------
    db_collection_name : str
        The collection to which to add
    config_files : str
        A path to configuration file (YAML) to add
    force_duplicates : bool, optional
        Whether to force adding configurations even if they are already in the MongoDB, by default False
    overwrite_params : Optional[Dict], optional
        Optional flat dict to override configuration values, by default None
    no_hash : bool, optional
        Whether to skip duplicate detection by computing hashes, which may result in slowdowns, by default False
    no_sanity_check : bool, optional
        Whether to skip feeding configuration values into the sacred experiment to detect unsupported or missing keys, by default False
    no_code_checkpoint : bool, optional
        Whether to not base the experiments on a copy of the current codebase, by default False
    resolve_descriptions : bool, optional
        Whether to use omegaconf to resolve descriptions
    """
    seml_config, slurm_config, experiment_config = read_config(config_file)

    # Use current Anaconda environment if not specified
    if 'conda_environment' not in seml_config:
        seml_config['conda_environment'] = os.environ.get('CONDA_DEFAULT_ENV')

    # Assemble the Slurm config:
    slurm_config = assemble_slurm_config_dict(slurm_config)

    configs_unresolved = generate_configs(experiment_config, overwrite_params=overwrite_params)
    configs, named_configs = generate_named_configs(configs_unresolved)
    configs = resolve_configs(seml_config['executable'], seml_config['conda_environment'], configs, named_configs, seml_config['working_dir'])
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
        # Sanity checking uses the resolved values (after considering named configs)
        check_config(seml_config['executable'], seml_config['conda_environment'], configs, seml_config['working_dir'])
        
    path, commit, dirty = get_git_info(seml_config['executable'], seml_config['working_dir'])

    git_info = None
    if path is not None:
        git_info = {'path': path, 'commit': commit, 'dirty': dirty}

    use_hash = not no_hash
    if use_hash:
        config_hashes = [make_hash(c, config_get_exclude_keys(c, c_unresolved)) for c, c_unresolved in zip(configs, configs_unresolved)]

    if not force_duplicates:
        len_before = len(configs)

        # First, check for duplicates withing the experiment configurations from the file.
        if not use_hash:
            # slow duplicate detection without hashes
            unique_configs, unique_keys = [], set()
            for c, c_unresolved in zip(configs, configs_unresolved):
                key = Hashabledict(**remove_keys_from_nested(c, config_get_exclude_keys(c, c_unresolved)))
                if key not in unique_keys:
                    unique_configs.append(c)
                    unique_keys.add(key)
            configs = unique_configs
        else:
            # fast duplicate detection using hashing.
            configs_dict = {config_hash: c for c, config_hash in zip(configs, config_hashes)}
            configs = [v for k, v in configs_dict.items()]

        len_after_deduplication = len(configs)
        # Now, check for duplicate configurations in the database.
        configs = filter_experiments(collection, configs, configs_unresolved, use_hash=use_hash)
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
        add_configs(collection, seml_config, slurm_config, configs, configs_unresolved, uploaded_files, git_info, resolve_descriptions=resolve_descriptions)
        