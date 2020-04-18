import os

import gridfs
from pymongo import MongoClient
import numpy as np
import yaml
import json
import jsonpickle
from bson import json_util
import warnings
import ast
import logging

from pymongo.collection import Collection

from seml.settings import SETTINGS
import urllib.parse
import pymongo

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


def get_results_flattened(collection_name):
    warnings.warn("This method is deprecated. Use database_utils.get_results instead.",
                  DeprecationWarning)

    collection = get_collection(collection_name)
    cursor = collection.find({'status': 'COMPLETED'}, {'config': 1, 'result': 1})
    results = list(cursor)

    def flatten_dict(prefix, d):
        dict_out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                dict_out.update(flatten_dict(prefix + key + '.', val))
            else:
                dict_out[prefix + key] = val
        return dict_out

    results_flattened = []
    for res in results:
        res_flattened = {}
        res_flattened['id'] = res['_id']
        res_flattened.update(flatten_dict("", res['config']))
        if isinstance(res['result'], dict):
            res_flattened.update(flatten_dict("", res['result']))
        else:
            res_flattened['result'] = res['result']
        results_flattened.append(res_flattened)
    return results_flattened


def get_results(collection_name, fields=['config', 'result'],
                to_data_frame=False, suffix=None,
                states=None, parallel=False):
    import pandas as pd
    if states is None:
        states = ['COMPLETED']
    collection = get_collection(collection_name, suffix=suffix)
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


def parse_jsonpickle(db_entry):
    import jsonpickle.ext.numpy as jsonpickle_numpy
    jsonpickle_numpy.register_handlers()
    try:
        parsed = jsonpickle.loads(json.dumps(db_entry, default=json_util.default), keys=False)
    except IndexError:
        parsed = db_entry
    return parsed


def restore(flat):
    """
    Restore more complex data that Python's json can't handle (e.g. Numpy arrays).
    Copied from sacred.serializer for performance reasons.
    """
    return jsonpickle.decode(json.dumps(flat), keys=True)


def _convert_value(value):
    """
    Parse string as python literal if possible and fallback to string.
    Copied from sacred.arg_parser for performance reasons.
    """

    try:
        return restore(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value


def convert_values(val):
    if isinstance(val, dict):
        for key, inner_val in val.items():
            val[key] = convert_values(inner_val)
    elif isinstance(val, list):
        for i, inner_val in enumerate(val):
            val[i] = convert_values(inner_val)
    elif isinstance(val, str):
        return _convert_value(val)
    return val


def read_config(config_path):
    with open(config_path, 'r') as conf:
        config_dict = convert_values(yaml.load(conf, Loader=yaml.FullLoader))

    if "seml" not in config_dict:
        raise ValueError("Please specify a 'seml' dictionary in the experiment configuration.")
    seml_dict = config_dict['seml']
    del config_dict['seml']
    if "executable" not in seml_dict:
        raise ValueError("Please specify an executable path for the experiment.")
    if "db_collection" not in seml_dict:
        raise ValueError("Please specify a database collection to store the experimental results.")

    if 'slurm' in config_dict:
        slurm_dict = config_dict['slurm']
        del config_dict['slurm']
        return seml_dict, slurm_dict, config_dict
    else:
        return seml_dict, None, config_dict


def get_collection(collection_name, mongodb_config=None, suffix=None):
    if mongodb_config is None:
        mongodb_config = get_mongodb_config()
    db = get_database(**mongodb_config)
    if suffix is not None and not collection_name.endswith(suffix):
        collection_name = f'{collection_name}{suffix}'

    return db[collection_name]


def get_database(db_name, host, port, username, password):
    db = MongoClient(host, int(port))[db_name]
    db.authenticate(name=username, password=password)
    return db


def get_collection_from_config(config):
    seml_config, _, _ = read_config(config)
    db_collection_name = seml_config['db_collection']
    return get_collection(db_collection_name)


def get_mongodb_config(path=SETTINGS.DATABASE.MONGODB_CONFIG_PATH):
    """Read the MongoDB connection configuration.

    Reads the file at the provided path or otherwise {SETTINGS.DATABASE.MONGODB_CONFIG_PATH} to get
        - database host
        - database port
        - database name
        - username
        - password

    Default path is $HOME/.config/seml/mongodb.config.

    Config file should be in the format:
    username: <your_username>
    password: <your_password>
    port: <port>
    database: <database_name>
    host: <host>

    Returns
    -------
    dict
        Contains the MongoDB config as detailed above.

    """

    access_dict = {}

    error_str = f"Please supply your MongoDB credentials at {path} in the format:\n"\
                "username: <your_username>\npassword: <your_password>\nport: <port>\n"\
                "database:<database_name>\n host: <hostname>"

    if not os.path.exists(path):
        raise ValueError(error_str)

    with open(path, 'r') as f:
        for line in f.readlines():
            # ignore lines that are empty
            if len(line.strip()) > 0:
                split = line.split(':')
                key = split[0].strip()
                value = split[1].strip()
                access_dict[key] = value

    assert 'username' in access_dict, f'No username found in {path}. {error_str}'
    assert 'password' in access_dict, f'No password found in {path}. {error_str}'
    assert 'port' in access_dict, f'No database port found in {path}. {error_str}'
    assert 'host' in access_dict, f'No host found in {path}. {error_str}'
    assert 'database' in access_dict, f'No database name found in {path}. {error_str}'

    db_port = access_dict['port']
    db_name = access_dict['database']
    db_host = access_dict['host']
    db_username = access_dict['username']
    db_password = access_dict['password']

    return {'password': db_password, 'username': db_username, 'host': db_host, 'db_name': db_name, 'port': db_port}


def create_mongodb_observer(collection,
                            mongodb_config=None,
                            overwrite=None):
    """ Create a MongoDB observer for a Sacred experiment

    Parameters
    ----------
    collection: str
        Name of the collection in the database to write the results to.
    mongodb_config: dict
        Dictionary containing the connection details to the MongoDB. See get_mongodb_config().
    overwrite: int
        ID of an experiment to overwrite, e.g. a queued or failed experiment.

    Returns
    -------
    observer: MongoObserver
    """
    from sacred.observers import MongoObserver

    if mongodb_config is None:
        mongodb_config = get_mongodb_config()

    db_name = urllib.parse.quote(mongodb_config['db_name'])
    db_username = urllib.parse.quote(mongodb_config['username'])
    db_password = urllib.parse.quote(mongodb_config['password'])
    db_port = urllib.parse.quote(mongodb_config['port'])
    db_host = urllib.parse.quote(mongodb_config['host'])
    observer = MongoObserver.create(
        url=f'mongodb://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}?authMechanism=SCRAM-SHA-1',
        db_name=db_name,
        collection=collection,
        overwrite=overwrite)

    return observer


def build_filter_dict(filter_states, batch_id, filter_dict):
    """
    Construct a dictionary to be used for filtering a MongoDB collection.

    Parameters
    ----------
    filter_states: list
        The states to filter for, e.g. ["RUNNING", "PENDING"].
    batch_id: int
        The batch ID to filter.
    filter_dict: dict
        The filter dict passed via the command line. Values in here take precedence over values passed via
        batch_id or filter_states.

    Returns
    -------
    filter_dict: dict
    """

    if filter_dict is None:
        filter_dict = {}

    if filter_states is not None and len(filter_states) > 0:
        if 'status' not in filter_dict:
            filter_dict['status'] = {'$in': filter_states}
        else:
            warnings.warn("'status' was defined in the filter dictionary passed via"
                          " the command line (-f): {} AND --status was set to {}. I'm using the value passed"
                          " via -f.".format(filter_dict['status'], filter_states))

    if batch_id is not None:
        if 'batch_id' not in filter_dict:
            filter_dict['batch_id'] = batch_id
        else:
            warnings.warn("'batch_id' was defined in the filter dictionary passed via"
                          " the command line (-f): {} AND --batch-id was set to {}. I'm using the value passed"
                          " via -f.".format(filter_dict['status'], filter_states))
    return filter_dict


def chunk_list(exps):
    """
    Divide experiments into chunks of `experiments_per_job` that will be run in parallel in one job.
    This assumes constant Slurm settings per batch (which should be the case if MongoDB wasn't edited manually).

    Parameters
    ----------
    exps: list[dict]
        List of dictionaries containing the experiment settings as saved in the MongoDB

    Returns
    -------
    exp_chunks: list
    """
    batch_idx = [exp['batch_id'] for exp in exps]
    unique_batch_idx = np.unique(batch_idx)
    exp_chunks = []
    for batch in unique_batch_idx:
        idx = [i for i, batch_id in enumerate(batch_idx)
               if batch_id == batch]
        size = exps[idx[0]]['slurm']['experiments_per_job']
        exp_chunks.extend(([exps[i] for i in idx[pos:pos + size]] for pos in range(0, len(idx), size)))
    return exp_chunks


def batch_chunks(exp_chunks):
    """
    Divide chunks of experiments into Slurm job arrays with one experiment batch per array. Each array is started together.
    This assumes constant Slurm settings per batch (which should be the case if MongoDB wasn't edited manually).

    Parameters
    ----------
    exp_chunks: list[list[dict]]
        List of list of dictionaries containing the experiment settings as saved in the MongoDB

    Returns
    -------
    exp_arrays: list[list[list[dict]]]
    """
    batch_idx = np.array([chunk[0]['batch_id'] for chunk in exp_chunks])
    unique_batch_idx = np.unique(batch_idx)
    ids_per_array = [np.where(batch_idx == array_bidx)[0] for array_bidx in unique_batch_idx]
    exp_arrays = [[exp_chunks[idx] for idx in chunk_ids] for chunk_ids in ids_per_array]
    return exp_arrays


def make_hash(d: dict):
    """
    Generate a hash for the input dictionary.
    From: https://stackoverflow.com/a/22003440
    Parameters
    ----------
    d: input dictionary

    Returns
    -------
    hash (hex encoded) of the input dictionary.
    """
    import hashlib
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def get_max_value(collection: Collection, field: str):
    """
    Find the maximum value in the input collection for the input field.
    Parameters
    ----------
    collection
    field

    Returns
    -------
    max_val: the maximum value in the field.
    """

    ndocs = collection.count_documents({})
    if field == "_id":
        c = collection.find({}, {'_id': 1})
    else:
        c = collection.find({}, {'_id': 1, field: 1})
    b = c.sort(field, pymongo.DESCENDING).limit(1)
    if ndocs != 0:
        b_next = b.next()
        max_val = b_next[field] if field in b_next else None
    else:
        max_val = None

    return max_val


def upload_source_file(filename, db_collection: Collection, batch_id):
    """
    Upload a source file to the MongoDB.
    Parameters
    ----------
    filename: str
    db_collection: Collection
    batch_id: int

    Returns
    -------
    file_id: ID of the inserted file, or None if there was an error.
    """
    db = db_collection.database
    fs = gridfs.GridFS(db)
    try:
        with open(filename, "rb") as f:
            db_filename = f"file://{db_collection.name}/{batch_id}/{filename}"
            file_id = fs.put(
                f, filename=db_filename, metadata={"collection_name": db_collection.name,
                                                   "type": "source_file",
                                                   "batch_id": batch_id}
            )
            return file_id
    except IOError:
        logging.error(f"IOError: could not read {filename}")
    return None


def load_sources_from_db(exp, to_directory):
    collection = get_collection(exp['seml']['db_collection'])
    db = collection.database
    fs = gridfs.GridFS(db)
    if not 'source_files' in exp['seml']:
        raise ValueError(f'No source files found for experiment with ID {exp["_id"]}')
    source_files = exp['seml']['source_files']
    for path, _id in source_files:
        _dir = f"{to_directory}/{os.path.dirname(path)}"
        if not os.path.exists(_dir):
            os.makedirs(_dir, mode=0o700)  # only current user can read, write, or execute
        with open(f'{to_directory}/{path}', 'wb') as f:
            file = fs.find_one(_id)
            if file is None:
                raise ValueError(f"Source file was not found on the MongoDB.")
            f.write(file.read())
