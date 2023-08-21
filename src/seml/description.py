import logging
import string
from typing import Dict, List, Optional

from pymongo import UpdateOne

from seml.database import build_filter_dict, get_collection
from seml.errors import MongoDBError
from seml.manage import detect_killed
from seml.settings import SETTINGS
from seml.typer import prompt
from seml.utils import slice_to_str, to_slices

States = SETTINGS.STATES

def collection_set_description(
    db_collection_name: str,
    description: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    filter_dict: Optional[Dict] = None,
    batch_id: Optional[int] = None,
    yes: bool = False):
    """ Sets (or updates) the description of experiment(s). 
    
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
    """
    collection = get_collection(db_collection_name)
    interpolation_vars = [t[1] for t in string.Formatter().parse(description) if t[1] is not None]
    mongo_db_vars = {var.replace('[', '.').replace(']', '') for var in interpolation_vars}
    
    filter_dict = build_filter_dict(filter_states, batch_id, filter_dict, sacred_id=sacred_id)
    exps = list(collection.find(filter_dict, {'seml.description': 1, **{k: 1 for k in mongo_db_vars}}))
    if len(exps) == 0 and sacred_id is not None:
        raise MongoDBError(f"No experiment found with ID {sacred_id}.")
    final_desc = {
        exp['_id']: description.format(**exp)
        for exp in exps
    }
    num_to_overwrite = sum(
        1 for exp in exps
        if exp.get('seml', {}).get('description', final_desc[exp['_id']]) != final_desc[exp['_id']]
    )
    if not yes and num_to_overwrite >= SETTINGS.CONFIRM_DESCRIPTION_UPDATE_THRESHOLD and \
        not prompt(f"{num_to_overwrite} experiment(s) have a different description. Proceed?", type=bool):
        exit(1)
    result = collection.bulk_write([
        UpdateOne({'_id': _id}, {'$set': {'seml.description': desc}})
        for _id, desc in final_desc.items()
    ])
    logging.info(f'Updated the descriptions of {result.modified_count} experiments.')
    
def collection_delete_description(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    filter_dict: Optional[Dict] = None,
    batch_id: Optional[int] = None,
    yes: bool = False):
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
    collection = get_collection(db_collection_name)
    update = {'$unset' : {'seml.description' : ''}}
    filter_dict = build_filter_dict(filter_states, batch_id, filter_dict, sacred_id=sacred_id)
    exps = [exp for exp in collection.find(filter_dict, {'seml.description' : 1})
            if exp.get('seml', {}).get('description', None) is not None]
    if not yes and len(exps) >= SETTINGS.CONFIRM_DESCRIPTION_DELETE_THRESHOLD and \
        not prompt(f"Deleting descriptions of {len(exps)} experiment(s). Proceed?", type=bool):
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

    from seml.console import console, Table
    collection = get_collection(db_collection_name)
    
    # Handle status updates
    if update_status:
        detect_killed(db_collection_name, print_detected=False)
    else:
        logging.warning(f"Status of {States.RUNNING[0]} experiments may not reflect if they have died or been canceled. Use the `--update-status` flag instead.")
    
    description_slices = {
        (obj['_id'] if obj['_id'] else ''): {
            'ids' : to_slices(obj['ids']),
            'batch_ids' : to_slices(obj['batch_ids']),
            'states' : set(obj['states']),
        }
        for obj in collection.aggregate([{
            '$group': {
                '_id': '$seml.description',
                'ids': {'$addToSet': '$_id'},
                'batch_ids' : {'$addToSet' : '$batch_id'},
                'states' : {'$addToSet' : '$status'},
            }
        }])
    }

    table = Table(show_header=True)
    table.add_column("Description", justify="left")
    table.add_column("Experiment IDs", justify="left")
    table.add_column("Batch IDs", justify="left")
    table.add_column("Status", justify="left")
    for description in sorted(description_slices):
        slices = description_slices[description]
        table.add_row(description, 
                      ", ".join(map(slice_to_str, slices['ids'])),
                      ", ".join(map(slice_to_str, slices['batch_ids'])),
                      ", ".join(slices['states'])
                    )
    console.print(Align(table, align="center"))
