import logging
import json
import jsonpickle
from tqdm.autonotebook import tqdm

from seml.database import get_collection

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


def get_results(db_collection_name, fields=['config', 'result'],
                to_data_frame=False, mongodb_config=None, suffix=None,
                states=['COMPLETED'], db_filter={}, parallel=False):
    import pandas as pd

    collection = get_collection(db_collection_name, mongodb_config=mongodb_config, suffix=suffix)

    if len(states) > 0:
        if 'status' in db_filter:
            logging.warning("'states' argument is not empty and will overwrite 'db_filter['status']'.")
        db_filter['status'] = {'$in': states}

    cursor = collection.find(db_filter, fields)
    results = [x for x in tqdm(cursor, total=collection.count_documents(db_filter))]

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
