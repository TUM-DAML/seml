from typing import List, Optional, Dict, Union
import logging

from seml.database import (build_filter_dict, get_collection)
from seml.errors import MongoDBError
from seml.settings import SETTINGS
from seml.typer import prompt

States = SETTINGS.STATES

def collection_set_description(
    db_collection_name: str,
    description: str,
    sacred_id: Optional[int] = None,
    filter_states: Union[Optional[List[str]], str] = None,
    filter_dict: Optional[Dict] = None,
    batch_id: Optional[int] = None,
    yes: bool = False) -> List[Dict]:
    """ Sets (or updates) the description of experiment(s). """
    collection = get_collection(db_collection_name)
    update = {'$set' : {'seml.description' : description}}
    if sacred_id is None:
        if isinstance(filter_states, str):
            filter_states = [filter_states]
        filter_dict = build_filter_dict(filter_states, batch_id, filter_dict)
        exps = [exp for exp in collection.find(filter_dict, {'seml.description' : 1})
                if exp.get('seml', {}).get('description', None) != description
                    and exp.get('seml', {}).get('description', None) is not None]
        if not yes and len(exps) >= SETTINGS.CONFIRM_UPDATE_DESCRIPTION_THRESHOLD and \
            not prompt(f"{len(exps)} experiment(s) have a different description. Proceed?", type=bool):
            filter_dict = {**filter_dict, **{'_id' : {'$nin' : [exp['_id'] for exp in exps]}}}
        result = collection.update_many(filter_dict, update)
    else:
        exp = collection.find_one({'_id': sacred_id}, {'seml.description' : 1})
        if exp is None:
            raise MongoDBError(f"No experiment found with ID {sacred_id}.")
        if not yes and exp.get('seml', {}).get('description', None) != description and \
            exp.get('seml', {}).get('description', None) is not None and SETTINGS.CONFIRM_UPDATE_DESCRIPTION_THRESHOLD <= 1 and \
            not prompt(f'Experiment with ID {sacred_id} has a different description ("{exp.get("seml", {}).get("description", None)}")'\
                '. Do you want to overwrite it?', type=bool):
            exit(1)
        result = collection.update_one({'_id': sacred_id}, update)
    logging.info(f'Updated the descriptions of {result.modified_count} experiments.')
    
def collection_delete_description(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Union[Optional[List[str]], str] = None,
    filter_dict: Optional[Dict] = None,
    batch_id: Optional[int] = None,
    yes: bool = False) -> List[Dict]:
    """ Deletes the description of experiment(s). """
    collection = get_collection(db_collection_name)
    update = {'$unset' : {'seml.description' : ''}}
    if sacred_id is None:
        if isinstance(filter_states, str):
            filter_states = [filter_states]
        filter_dict = build_filter_dict(filter_states, batch_id, filter_dict)
        exps = [exp for exp in collection.find(filter_dict, {'seml.description' : 1})
                if exp.get('seml', {}).get('description', None) is not None]
        if not yes and len(exps) >= SETTINGS.CONFIRM_DELETE_DESCRIPTION_THRESHOLD and \
            not prompt(f"Deleting descriptions of {len(exps)} experiment(s). Proceed?", type=bool):
            exit(1)
        result = collection.update_many(filter_dict, update)
    else:
        exp = collection.find_one({'_id': sacred_id}, {'seml.description' : 1})
        if exp is None:
            raise MongoDBError(f"No experiment found with ID {sacred_id}.")
        if not yes and exp.get('seml', {}).get('description', None) is not None and \
            SETTINGS.CONFIRM_DELETE_DESCRIPTION_THRESHOLD <= 1 and \
            not prompt(f'Deleting the description of experiment with ID {sacred_id} ("{exp.get("seml", {}).get("description", None)}")?', type=bool):
            exit(1)
        result = collection.update_one({'_id': sacred_id}, update)
    logging.info(f'Deleted the descriptions of {result.modified_count} experiments.')