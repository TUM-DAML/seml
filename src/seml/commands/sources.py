import logging
import os
from typing import Dict, List, Optional

from seml.database import build_filter_dict, get_collection
from seml.experiment.sources import load_sources_from_db
from seml.settings import SETTINGS

States = SETTINGS.STATES


def download_sources(
    target_directory: str,
    collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
):
    """
    Restore source files from the database to the provided path. This is a helper function for the CLI.

    Parameters
    ----------
    target_directory: str
        The directory where the source files should be restored.
    collection_name: str
        The name of the MongoDB collection.
    sacred_id: int
        The ID of the Sacred experiment.
    filter_states: List[str]
        The states of the experiments to filter.
    batch_id: int
        The ID of the batch.
    filter_dict: Dict
        Additional filter dictionary.
    """
    from seml.console import prompt

    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    collection = get_collection(collection_name)
    experiments = list(collection.find(filter_dict))
    batch_ids = {exp['batch_id'] for exp in experiments}

    if len(batch_ids) > 1:
        logging.error(
            f'Multiple source code versions found for batch IDs: {batch_ids}.'
        )
        logging.error('Please specify the target experiment more concretely.')
        exit(1)

    exp = experiments[0]
    target_directory = os.path.expandvars(os.path.expanduser(target_directory))
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)
    if os.listdir(target_directory):
        logging.warning(
            f'Target directory "{target_directory}" is not empty. '
            f'Files may be overwritten.'
        )
        if not prompt('Are you sure you want to continue? (y/n)', type=bool):
            exit(0)

    load_sources_from_db(exp, collection, target_directory)
    logging.info(f'Source files restored to "{target_directory}".')
