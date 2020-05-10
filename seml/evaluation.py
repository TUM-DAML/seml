import json
import jsonpickle
from bson import json_util
try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable

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
                to_data_frame=False, suffix=None,
                states=['COMPLETED'], parallel=False):
    import pandas as pd

    collection = get_collection(db_collection_name, suffix=suffix)
    if len(states) > 0:
        filter = {'status': {'$in': states}}
    else:
        filter = {}
    cursor = collection.find(filter, fields)
    results = [x for x in tqdm(cursor, total=collection.count_documents(filter))]

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
