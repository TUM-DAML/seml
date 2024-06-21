import logging
from typing import Dict, List, Optional

from seml.commands.manage import detect_killed
from seml.database import build_filter_dict, get_collection
from seml.experiment.description import resolve_description
from seml.settings import SETTINGS
from seml.utils import slice_to_str, to_slices
from seml.utils.errors import MongoDBError

States = SETTINGS.STATES


def collection_set_description(
    db_collection_name: str,
    description: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    filter_dict: Optional[Dict] = None,
    batch_id: Optional[int] = None,
    yes: bool = False,
    resolve: bool = True,
):
    """Sets (or updates) the description of experiment(s).

    Parameters
    ----------
    db_collection_name : str
        Name of the collection to delete descriptions from
    description : str
        The description to set.
    sacred_id : Optional[int], optional
        If given, the id of the experiment to delete the description of. Overrides other filters, by default None
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    filter_dict : Optional[Dict], optional
        Additional filters, by default None
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    yes : bool, optional
        Whether to override confirmation prompts, by default False
    resolve : bool, optional
        Whether to use omegaconf to resolve descriptions
    """
    from pymongo import UpdateOne

    from seml.console import prompt

    collection = get_collection(db_collection_name)

    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    exps = list(collection.find(filter_dict, {}))
    if len(exps) == 0 and sacred_id is not None:
        raise MongoDBError(f'No experiment found with ID {sacred_id}.')
    descriptions_resolved = {
        exp['_id']: resolve_description(description, exp) if resolve else description
        for exp in exps
    }
    num_to_overwrite = len(
        list(
            filter(
                lambda exp: exp.get('seml', {}).get(
                    'description', descriptions_resolved[exp['_id']]
                )
                != descriptions_resolved[exp['_id']],
                exps,
            )
        )
    )

    if (
        not yes
        and num_to_overwrite >= SETTINGS.CONFIRM_THRESHOLD.DESCRIPTION_UPDATE
        and not prompt(
            f'{num_to_overwrite} experiment(s) have a different description. Proceed?',
            type=bool,
        )
    ):
        exit(1)
    if len(list(filter(lambda exp: exp['status'] in States.RUNNING, exps))):
        logging.warn(
            f'Updating the description of {States.RUNNING[0]} experiments: This may not have an'
            ' effect, as sacred overwrites experiments with each tick.'
        )
    result = collection.bulk_write(
        [
            UpdateOne({'_id': _id}, {'$set': {'seml.description': description}})
            for _id, description in descriptions_resolved.items()
        ]
    )
    logging.info(f'Updated the descriptions of {result.modified_count} experiments.')


def collection_delete_description(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    filter_dict: Optional[Dict] = None,
    batch_id: Optional[int] = None,
    yes: bool = False,
):
    """Deletes the description of experiments

    Parameters
    ----------
    db_collection_name : str
        Name of the collection to delete descriptions from
    sacred_id : Optional[int], optional
        If given, the id of the experiment to delete the descriptino of. Overrides other filters, by default None
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    filter_dict : Optional[Dict], optional
        Additional filters, by default None
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    yes : bool, optional
        Whether to override confirmation prompts, by default False
    """
    from seml.console import prompt

    collection = get_collection(db_collection_name)
    update = {'$unset': {'seml.description': ''}}
    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    exps = [
        exp
        for exp in collection.find(filter_dict, {'seml.description': 1})
        if exp.get('seml', {}).get('description', None) is not None
    ]
    if (
        not yes
        and len(exps) >= SETTINGS.CONFIRM_THRESHOLD.DESCRIPTION_DELETE
        and not prompt(
            f'Deleting descriptions of {len(exps)} experiment(s). Proceed?', type=bool
        )
    ):
        exit(1)
    result = collection.update_many(filter_dict, update)
    logging.info(f'Deleted the descriptions of {result.modified_count} experiments.')


def collection_list_descriptions(db_collection_name: str, update_status: bool = False):
    """Lists the descriptions of experiments

    Parameters
    ----------
    db_collection_name : str
        Name of the collection to list descriptions from
    update_status : bool
        Whether to detect killed experiments
    """
    from rich.align import Align

    from seml.console import Table, console

    collection = get_collection(db_collection_name)

    # Handle status updates
    if update_status:
        detect_killed(db_collection_name, print_detected=False)
    else:
        logging.warning(
            f'Status of {States.RUNNING[0]} experiments may not reflect if they have died or been canceled. Use the `--update-status` flag instead.'
        )

    description_slices = {
        (obj['_id'] if obj['_id'] else ''): {
            'ids': to_slices(obj['ids']),
            'batch_ids': to_slices(obj['batch_ids']),
            'states': set(obj['states']),
        }
        for obj in collection.aggregate(
            [
                {
                    '$group': {
                        '_id': '$seml.description',
                        'ids': {'$addToSet': '$_id'},
                        'batch_ids': {'$addToSet': '$batch_id'},
                        'states': {'$addToSet': '$status'},
                    }
                }
            ]
        )
    }

    table = Table(show_header=True)
    table.add_column('Description', justify='left')
    table.add_column('Experiment IDs', justify='left')
    table.add_column('Batch IDs', justify='left')
    table.add_column('Status', justify='left')
    for description in sorted(description_slices):
        slices = description_slices[description]
        table.add_row(
            description,
            ', '.join(map(slice_to_str, slices['ids'])),
            ', '.join(map(slice_to_str, slices['batch_ids'])),
            ', '.join(slices['states']),
        )
    console.print(Align(table, align='center'))
