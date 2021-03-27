import gridfs
import pymongo
from pymongo.collection import Collection
import logging
from tqdm.autonotebook import tqdm

from seml.utils import s_if
from seml.settings import SETTINGS
from seml.errors import MongoDBError


def get_collection(collection_name, mongodb_config=None, suffix=None):
    if mongodb_config is None:
        mongodb_config = get_mongodb_config()
    db = get_database(**mongodb_config)
    if suffix is not None and not collection_name.endswith(suffix):
        collection_name = f'{collection_name}{suffix}'

    return db[collection_name]


def get_database(db_name, host, port, username, password):
    db = pymongo.MongoClient(host, int(port))[db_name]
    db.authenticate(name=username, password=password)
    return db


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
    config_str = "\nPlease run `seml configure` to provide your credentials."

    if not path.exists():
        raise MongoDBError(f"MongoDB credentials could not be read at '{path}'.{config_str}")

    with open(path, 'r') as f:
        for line in f.readlines():
            # ignore lines that are empty
            if len(line.strip()) > 0:
                split = line.split(':')
                key = split[0].strip()
                value = split[1].strip()
                access_dict[key] = value

    required_entries = ['username', 'password', 'port', 'host', 'database']
    for entry in required_entries:
        if entry not in access_dict:
            raise MongoDBError(f"No {entry} found in '{path}'.{config_str}")

    db_port = access_dict['port']
    db_name = access_dict['database']
    db_host = access_dict['host']
    db_username = access_dict['username']
    db_password = access_dict['password']

    return {'password': db_password, 'username': db_username, 'host': db_host, 'db_name': db_name, 'port': db_port}


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
            logging.warning(f"'status' was defined in the filter dictionary passed via the command line (-f): "
                            f"{filter_dict['status']} AND --status was set to {filter_states}. "
                            f"I'm using the value passed via -f.")

    if batch_id is not None:
        if 'batch_id' not in filter_dict:
            filter_dict['batch_id'] = batch_id
        else:
            logging.warning(f"'batch_id' was defined in the filter dictionary passed via the command line (-f): "
                            f"{filter_dict['status']} AND --batch-id was set to {filter_states}. "
                            f"I'm using the value passed via -f.")
    return filter_dict


def get_max_in_collection(collection: Collection, field: str):
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


def upload_file(filename, db_collection: Collection, batch_id, filetype):
    """
    Upload a source file to the MongoDB.
    Parameters
    ----------
    filename: str
    db_collection: Collection
    batch_id: int
    filetype: str

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
                                                   "batch_id": batch_id,
                                                   "type": filetype}
            )
            return file_id
    except IOError:
        logging.error(f"IOError: could not read {filename}")
    return None


def clean_unreferenced_artifacts(db_collection_name, all_collections=False):
    """
    Delete orphaned artifacts from the database. That is, artifacts that were generated by experiments, but whose
    experiment's database entry has been removed. This leads to storage accumulation, and this function cleans this
    excess storage.
    Parameters
    ----------
    db_collection_name: str
        Database collection to be scanned.
    all_collections: bool
        If yes, will scan ALL collections (not just the one provided in the config file) for orphaned artifacts.

    Returns
    -------
    None
    """
    import gridfs
    if all_collections:
        config = get_mongodb_config()
        db = get_database(**config)
        collection_names = db.list_collection_names()
    else:
        collection = get_collection(db_collection_name)
        db = collection.database
        collection_names = [collection.name]
    collection_names = set(collection_names)
    collection_blacklist = {'fs.chunks', 'fs.files'}
    collection_names = collection_names - collection_blacklist

    fs = gridfs.GridFS(db)
    referenced_files = set()
    for collection_name in tqdm(collection_names):
        collection = db[collection_name]
        experiments = list(collection.find({}, {'artifacts': 1, 'experiment.sources': 1, 'source_files': 1}))
        for exp in experiments:
            if 'artifacts' in exp:
                referenced_files.update({x[1] for x in exp['artifacts']})
            if 'experiment' in exp and 'sources' in exp['experiment']:
                referenced_files.update({x[1] for x in exp['experiment']['sources']})
            if 'source_files' in exp:
                referenced_files.update({x[1] for x in exp['source_files']})

    all_files_in_db = list(db['fs.files'].find({}, {'_id': 1, 'filename': 1, 'metadata': 1}))
    filtered_file_ids = set()
    for file in all_files_in_db:
        if 'filename' in file:
            filename = file['filename']
            file_collection = None
            if filename.startswith("file://") and 'metadata' in file and file['metadata']:
                # seml-uploaded source
                metadata = file['metadata']
                file_collection = metadata['collection_name'] if 'collection_name' in metadata else None
            elif filename.startswith("artifact://"):
                # artifact uploaded by Sacred
                filename = filename[11:]
                file_collection = filename.split("/")[0]
            if file_collection is not None and file_collection in collection_names:
                # only delete files corresponding to collections we want to clean
                filtered_file_ids.add(file['_id'])

    not_referenced_artifacts = filtered_file_ids - referenced_files
    n_delete = len(not_referenced_artifacts)
    if n_delete == 0:
        logging.info("No unreferenced artifacts found.")
        return

    if input(f"Deleting {n_delete} not referenced artifact{s_if(n_delete)} from database {db.name}. "
             f"Are you sure? (y/n) ").lower() != "y":
        exit()
    logging.info('Deleting not referenced artifacts...')
    for to_delete in tqdm(not_referenced_artifacts):
        fs.delete(to_delete)
    logging.info(f'Successfully deleted {n_delete} not referenced artifact{s_if(n_delete)}.')
