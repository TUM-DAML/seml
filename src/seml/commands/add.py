from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

from seml.database import get_collection, get_max_in_collection
from seml.document import ExperimentDoc, SemlDoc
from seml.experiment.config import (
    assemble_slurm_config_dict,
    check_config,
    config_get_exclude_keys,
    generate_configs,
    generate_named_configs,
    read_config,
    remove_duplicates,
    requires_interpolation,
    resolve_configs,
    resolve_interpolations,
)
from seml.experiment.description import resolve_description
from seml.experiment.sources import get_git_info, upload_sources
from seml.settings import SETTINGS
from seml.utils import (
    flatten,
    make_hash,
    remove_keys_from_nested,
    to_typeddict,
    unflatten,
    utcnow,
)
from seml.utils.errors import ConfigError

if TYPE_CHECKING:
    from pymongo.collection import Collection

States = SETTINGS.STATES


def remove_existing_experiments(
    collection: Collection[ExperimentDoc],
    documents: list[ExperimentDoc],
    use_hash: bool = True,
):
    """Check database collection for already present entries.

    Check the database collection for experiments that have the same configuration.
    Remove the corresponding entries from the input list of configurations to prevent
    re-running the experiments.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    documents: List[Dict]
        The documents to filter.
    use_hash : bool
        Whether to use hashes (faster)

    Returns
    -------
    filtered_configs: list of dicts
        No longer contains configurations that are already in the database collection.

    """
    filtered_documents: list[ExperimentDoc] = []
    for document in documents:
        if use_hash:
            lookup_result = collection.find_one(
                {'config_hash': document['config_hash']}
            )
        else:
            lookup_dict = flatten(
                {
                    'config': remove_keys_from_nested(
                        document['config'], document['config_unresolved'].keys()
                    )
                }
            )
            lookup_result = collection.find_one(unflatten(lookup_dict))
        if lookup_result is None:
            filtered_documents.append(document)
    return filtered_documents


def add_configs(
    collection: Collection[ExperimentDoc],
    documents: list[ExperimentDoc],
    description: str | None = None,
    resolve_descriptions: bool = True,
):
    """Put the input configurations into the database.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    documents : List[Dict]
        The documents to add.
    description : Optional[str], optional
        Optional description for the experiments, by default None
    resolve_descriptions : bool, optional
        Whether to use omegaconf to resolve descriptions.
    """

    if len(documents) == 0:
        return

    start_id = get_max_in_collection(collection, '_id')
    if start_id is None:
        start_id = 1
    else:
        start_id = start_id + 1

    logging.info(
        f"Adding {len(documents)} configs to the database (batch-ID {documents[0]['batch_id']})."
    )

    documents = [
        cast(
            ExperimentDoc,
            {
                **document,
                **{
                    '_id': start_id + idx,
                    'status': States.STAGED[0],
                    'add_time': utcnow(),
                },
            },
        )
        for idx, document in enumerate(documents)
    ]
    if description is not None:
        for db_dict in documents:
            db_dict['seml']['description'] = description
        # If description is not supplied via CLI, it will already be resolved.
        if resolve_descriptions and requires_interpolation(
            {'description': description}, ['description']
        ):
            for db_dict in documents:
                db_dict['seml']['description'] = resolve_description(
                    db_dict['seml'].get('description', ''), db_dict
                )

    collection.insert_many(documents)


def add_config_files(
    db_collection_name: str,
    config_files: list[str],
    force_duplicates: bool = False,
    overwrite_params: dict | None = None,
    no_hash: bool = False,
    no_sanity_check: bool = False,
    no_code_checkpoint: bool = False,
    description: str | None = None,
    resolve_descriptions: bool = True,
):
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
    description : Optional[str], optional
        Optional description for the experiments, by default None
    resolve_descriptions : bool, optional
        Whether to use omegaconf to resolve experiment descriptions.
    """
    config_files = [os.path.abspath(file) for file in config_files]
    for config_file in config_files:
        add_config_file(
            db_collection_name,
            config_file,
            force_duplicates,
            overwrite_params,
            no_hash,
            no_sanity_check,
            no_code_checkpoint,
            description=description,
            resolve_descriptions=resolve_descriptions,
        )


def add_config_file(
    db_collection_name: str,
    config_file: str,
    force_duplicates: bool = False,
    overwrite_params: dict[str, Any] | None = None,
    no_hash: bool = False,
    no_sanity_check: bool = False,
    no_code_checkpoint: bool = False,
    description: str | None = None,
    resolve_descriptions: bool = True,
):
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
    description : Optional[str], optional
        Optional description for the experiments, by default None
    resolve_descriptions : bool, optional
        Whether to use omegaconf to resolve descriptions
    """

    collection = get_collection(db_collection_name)
    seml_config, slurm_configs, experiment_config = read_config(config_file)

    # Use current Anaconda environment if not specified
    if 'conda_environment' not in seml_config:
        seml_config['conda_environment'] = os.environ.get('CONDA_DEFAULT_ENV')

    # Get git info
    git_info = get_git_info(seml_config['executable'], seml_config['working_dir'])

    # Compute batch id
    batch_id = get_max_in_collection(collection, 'batch_id', int)
    batch_id = 1 if batch_id is None else batch_id + 1

    # Assemble the Slurm config:
    slurm_configs = list(map(assemble_slurm_config_dict, slurm_configs))
    if len(slurm_configs) == 0:
        raise ConfigError('No slurm configuration found.')

    configs_unresolved = generate_configs(
        experiment_config, overwrite_params=overwrite_params
    )
    configs, named_configs = generate_named_configs(configs_unresolved)
    configs = resolve_configs(
        seml_config['executable'],
        seml_config['conda_environment'],
        configs,
        named_configs,
        seml_config['working_dir'],
    )

    # Upload source files: This also determines the batch_id
    use_uploaded_sources = seml_config['use_uploaded_sources']
    seml_config = to_typeddict(seml_config, SemlDoc)
    if use_uploaded_sources and not no_code_checkpoint:
        seml_config['source_files'] = upload_sources(seml_config, collection, batch_id)

    # Create documents that can be interpolated
    documents = [
        cast(
            ExperimentDoc,
            {
                **resolve_interpolations(
                    {
                        'seml': seml_config,
                        'slurm': slurm_configs,
                        'git': git_info,
                        'batch_id': batch_id,  # needs to be determined now for source file uploading
                        'config': config,
                        'config_unresolved': config_unresolved,
                    }
                ),
                'config_unresolved': config_unresolved,
            },
        )
        for config, config_unresolved in zip(configs, configs_unresolved)
    ]

    if not no_sanity_check:
        # Sanity checking uses the resolved values (after considering named configs)
        check_config(
            seml_config['executable'],
            seml_config['conda_environment'],
            [document['config'] for document in documents],
            seml_config['working_dir'],
        )

    for document in documents:
        document['config_hash'] = make_hash(
            document['config'],
            config_get_exclude_keys(document['config_unresolved']),
        )

    if not force_duplicates:
        documents = remove_duplicates(collection, documents, use_hash=not no_hash)

    # Create an index on the config hash. If the index is already present, this simply does nothing.
    collection.create_index('config_hash')
    # Add the configurations to the database with STAGED status.
    if len(configs) > 0:
        add_configs(
            collection,
            documents,
            description=description,
            resolve_descriptions=resolve_descriptions,
        )
