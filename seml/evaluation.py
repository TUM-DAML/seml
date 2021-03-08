import logging
import json
import jsonpickle
from tqdm.autonotebook import tqdm

from seml.database import get_collection
from seml.settings import SETTINGS

States = SETTINGS.STATES

__all__ = ['get_results']


def parse_jsonpickle(db_entry):
    import jsonpickle.ext.numpy as jsonpickle_numpy

    jsonpickle_numpy.register_handlers()
    try:
        p = jsonpickle.pickler.Pickler(keys=False)
        parsed = jsonpickle.loads(json.dumps(db_entry, default=p.flatten), keys=False)
    except IndexError:
        parsed = db_entry
    return parsed


def get_results(db_collection_name, fields=None,
                to_data_frame=False, mongodb_config=None,
                states=None, filter_dict=None, parallel=False):
    """
    Get experiment results from the MongoDB.
    Parameters
    ----------
    db_collection_name: str
        Name of the MongoDB collection.
    fields: list (optional).
        Database attributes to extract. Default: ['config', 'result'].
    to_data_frame: bool, default: False
        Whether to convert the results into a Pandas DataFrame.
    mongodb_config: dict (optional)
        MongoDB credential dictionary. If None, uses the credentials specified by `seml configure`.
    states: list of strings (optional)
        Extract only experiments with certain states. Default: ['COMPLETED'].
    filter_dict: dict (optional)
        Custom dictionary for filtering results from the MongoDB.
    parallel: bool, default: False
        If True, unserialize entries in parallel. Use for very large experiment collections.

    Returns
    -------

    """
    import pandas as pd
    if fields is None:
        fields = ['config', 'result']

    if states is None:
        states = States.COMPLETED

    if filter_dict is None:
        filter_dict = {}

    collection = get_collection(db_collection_name, mongodb_config=mongodb_config,)

    if len(states) > 0:
        if 'status' in filter_dict:
            logging.warning("'states' argument is not empty and will overwrite 'filter_dict['status']'.")
        filter_dict['status'] = {'$in': states}

    cursor = collection.find(filter_dict, fields)
    results = [x for x in tqdm(cursor, total=collection.count_documents(filter_dict))]

    if parallel:
        from multiprocessing import Pool
        with Pool() as p:
            parsed = list(tqdm(p.imap(parse_jsonpickle, results),
                               total=len(results)))
    else:
        parsed = [parse_jsonpickle(entry) for entry in tqdm(results)]
    if to_data_frame:
        parsed = pd.io.json.json_normalize(parsed, sep='.')
    return parsed
