import logging
from pathlib import Path
from typing import List, Optional

from seml.settings import SETTINGS
from seml.utils import s_if
from seml.utils.errors import MongoDBError
from seml.utils.ssh_forward import get_forwarded_mongo_client

States = SETTINGS.STATES


def get_collection(collection_name, mongodb_config=None, suffix=None):
    if mongodb_config is None:
        mongodb_config = get_mongodb_config()
    db = get_database(**mongodb_config)
    if suffix is not None and not collection_name.endswith(suffix):
        collection_name = f'{collection_name}{suffix}'

    return db[collection_name]


def get_mongo_client(
    db_name, host, port, username, password, ssh_config=None, **kwargs
):
    import pymongo

    if ssh_config is not None:
        client = get_forwarded_mongo_client(
            db_name, username, password, ssh_config, **kwargs
        )
    else:
        client = pymongo.MongoClient(
            host,
            int(port),
            username=username,
            password=password,
            authSource=db_name,
            **kwargs,
        )
    return client


def get_database(db_name, host, port, username, password, **kwargs):
    db = get_mongo_client(db_name, host, port, username, password, **kwargs)[db_name]
    return db


def get_collections_from_mongo_shell_or_pymongo(
    db_name: str, host: str, port: int, username: str, password: str, **kwargs
) -> List[str]:
    """Gets all collections in the database by first using the mongo shell and if that fails uses pymongo.

    Args:
        db_name (str): the name of the database
        host (str): the name of the host
        port (int): the port at which to access the mongodb
        username (str): the username
        password (str): the password

    Returns:
        List[str]: all collections in the database
    """
    import subprocess

    cmd = (
        f"mongo -u '{username}' --authenticationDatabase '{db_name}' {host}:{port}/{db_name} -p {password} "
        "--eval 'db.getCollectionNames().forEach(function(f){print(f)})' --quiet"
    )
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        collection_names = output.decode('utf-8').split('\n')
    except (subprocess.CalledProcessError, SyntaxError):
        db = get_database(db_name, host, port, username, password, **kwargs)
        collection_names = db.list_collection_names()
    return [name for name in collection_names if name not in ('fs.chunks', 'fs.files')]


def get_mongodb_config(path=SETTINGS.DATABASE.MONGODB_CONFIG_PATH):
    """Read the MongoDB connection configuration.

    Reads the file at the provided path or otherwise {SETTINGS.DATABASE.MONGODB_CONFIG_PATH} to get
        - database host
        - database port
        - database name
        - username
        - password
        - directConnection (Optional)
        - ssh_config (Optional)

    Default path is $HOME/.config/seml/mongodb.config.

    Config file should be in the format:
    username: <your_username>
    password: <your_password>
    port: <port>
    database: <database_name>
    host: <host>
    directConnection: <bool> (Optional)
    ssh_config: <dict> (Optional)
      ssh_address_or_host: <the url of the jump host>
      ssh_pkey: <the ssh host key>
      ssh_username: <username for jump host>
      retries_max: <number of retries to establish shh tunnel, default 6> (Optional)
      retries_delay: <initial wait time for exponential retry, default 1> (Optional)
      lock_file: <lockfile to avoid establishing ssh tunnel parallely, default `~/seml_ssh.lock`> (Optional)
      lock_timeout: <timeout for aquiring lock, default 30> (Optional)
      ** further arguments passed to `SSHTunnelForwarder` (see https://github.com/pahaz/sshtunnel)

    Returns
    -------
    dict
        Contains the MongoDB config as detailed above.

    """
    import yaml

    access_dict = {}
    config_str = '\nPlease run `seml configure` to provide your credentials.'

    if not path.exists():
        raise MongoDBError(
            f"MongoDB credentials could not be read at '{path}'.{config_str}"
        )

    with open(path, 'r') as conf:
        access_dict = yaml.safe_load(conf)

    required_entries = ['username', 'password', 'port', 'host', 'database']
    for entry in required_entries:
        if entry not in access_dict:
            raise MongoDBError(f"No {entry} found in '{path}'.{config_str}")

    db_port = access_dict['port']
    db_name = access_dict['database']
    db_host = access_dict['host']
    db_username = access_dict['username']
    db_password = access_dict['password']
    # False is the default value for PyMongo > 4.0
    db_direct = (
        access_dict['directConnection'] == 'True'
        if 'directConnection' in access_dict
        else False
    )

    cfg = {
        'password': db_password,
        'username': db_username,
        'host': db_host,
        'db_name': db_name,
        'port': db_port,
        'directConnection': db_direct,
    }

    if 'ssh_config' not in access_dict:
        return cfg

    cfg['ssh_config'] = access_dict['ssh_config']
    cfg['ssh_config']['remote_bind_address'] = (db_host, db_port)
    cfg['directConnection'] = True

    return cfg


def build_filter_dict(filter_states, batch_id, filter_dict, sacred_id=None):
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
    sacred_id: int
        Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.

    Returns
    -------
    filter_dict: dict
    """

    if sacred_id is not None:
        return {'_id': sacred_id}

    if filter_dict is None:
        filter_dict = {}

    if filter_states is not None and len(filter_states) > 0:
        if 'status' not in filter_dict:
            filter_dict['status'] = {'$in': filter_states}
        else:
            logging.warning(
                f"'status' was defined in the filter dictionary passed via the command line (-f): "
                f"{filter_dict['status']} AND --status was set to {filter_states}. "
                f"I'm using the value passed via -f."
            )

    if batch_id is not None:
        if 'batch_id' not in filter_dict:
            filter_dict['batch_id'] = batch_id
        else:
            logging.warning(
                f"'batch_id' was defined in the filter dictionary passed via the command line (-f): "
                f"{filter_dict['status']} AND --batch-id was set to {filter_states}. "
                f"I'm using the value passed via -f."
            )
    return filter_dict


def get_max_in_collection(collection, field: str):
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
    import pymongo

    ndocs = collection.count_documents({})
    if field == '_id':
        c = collection.find({}, {'_id': 1})
    else:
        c = collection.find({}, {'_id': 1, field: 1})
    b = c.sort(field, pymongo.DESCENDING).limit(1)
    if ndocs != 0:
        b_next = b.next()
        max_val = b_next.get(field)
    else:
        max_val = None

    return max_val


def upload_file(filename, db_collection, batch_id, filetype):
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
    import gridfs

    db = db_collection.database
    fs = gridfs.GridFS(db)
    try:
        with open(filename, 'rb') as f:
            db_filename = f'file://{db_collection.name}/{batch_id}/{filename}'
            file_id = fs.put(
                f,
                filename=db_filename,
                metadata={
                    'collection_name': db_collection.name,
                    'batch_id': batch_id,
                    'type': filetype,
                },
            )
            return file_id
    except IOError:
        logging.error(f'IOError: could not read {filename}')
    return None


def delete_files(database, file_ids, progress=False):
    import gridfs

    from seml.console import track

    fs = gridfs.GridFS(database)
    it = track(file_ids, disable=not progress)
    for to_delete in it:
        fs.delete(to_delete)


def clean_unreferenced_artifacts(db_collection_name=None, yes=False):
    """
    Delete orphaned artifacts from the database. That is, artifacts that were generated by experiments, but whose
    experiment's database entry has been removed. This leads to storage accumulation, and this function cleans this
    excess storage.
    Parameters
    ----------
    db_collection_name: str, optional
        Database collection to be scanned. If None, all collections will be scanned.
    yes: bool
        Whether to automatically confirm the deletion dialog.
    Returns
    -------
    None
    """
    from seml.console import prompt, track

    all_collections = not bool(db_collection_name)
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

    referenced_files = set()
    tq = track(collection_names)
    logging.info('Scanning collections for orphaned artifacts...')
    for collection_name in tq:
        collection = db[collection_name]
        experiments = list(
            collection.find(
                {}, {'artifacts': 1, 'experiment.sources': 1, 'source_files': 1}
            )
        )
        for exp in experiments:
            if 'artifacts' in exp:
                try:
                    referenced_files.update({x[1] for x in exp['artifacts']})
                except KeyError:
                    referenced_files.update({x['file_id'] for x in exp['artifacts']})
            if 'experiment' in exp and 'sources' in exp['experiment']:
                referenced_files.update({x[1] for x in exp['experiment']['sources']})
            if 'source_files' in exp:
                referenced_files.update({x[1] for x in exp['source_files']})

    all_files_in_db = list(
        db['fs.files'].find({}, {'_id': 1, 'filename': 1, 'metadata': 1})
    )
    filtered_file_ids = set()
    for file in all_files_in_db:
        if 'filename' in file:
            filename = file['filename']
            file_collection = None
            if (
                filename.startswith('file://')
                and 'metadata' in file
                and file['metadata']
            ):
                # seml-uploaded source
                metadata = file['metadata']
                file_collection = metadata.get('collection_name')
            elif filename.startswith('artifact://'):
                # artifact uploaded by Sacred
                filename = filename[11:]
                file_collection = filename.split('/')[0]
            if (
                file_collection is not None and file_collection in collection_names
            ) or all_collections:
                # only delete files corresponding to collections we want to clean
                filtered_file_ids.add(file['_id'])

    not_referenced_artifacts = filtered_file_ids - referenced_files
    n_delete = len(not_referenced_artifacts)
    if n_delete == 0:
        logging.info('No unreferenced artifacts found.')
        return

    logging.info(
        f'Deleting {n_delete} not referenced artifact{s_if(n_delete)} from database {db.name}. WARNING: This cannot be undone! Artifacts/ files might have been inserted to MongoDB manually or by tools other than seml/ sacred. They will be deleted.'
    )
    if not yes and not prompt('Are you sure? (y/n)', type=bool):
        exit(1)
    logging.info('Deleting not referenced artifacts...')
    delete_files(db, not_referenced_artifacts, progress=True)
    logging.info(
        f'Successfully deleted {n_delete} not referenced artifact{s_if(n_delete)}.'
    )


def update_working_dir(
    db_collection_name: str,
    working_directory: str,
    batch_ids: Optional[List[int]] = None,
):
    """Changes the working directory of experiments in the database.

    Parameters
    ----------
    db_collection_name : str
        The collection to change the working directory in.
    working_directory: str
        The new working directory.
    batch_ids : Optional[List[int]], optional
        Filter on the batch ids, by default None
    """

    collection = get_collection(db_collection_name)
    if batch_ids is not None and len(batch_ids) > 0:
        filter_dict = {'batch_id': {'$in': list(batch_ids)}}
    else:
        filter_dict = {}

    working_directory = str(Path(working_directory).expanduser().resolve())

    update_result = collection.update_many(
        filter_dict,
        {'$set': {'seml.working_dir': working_directory}},
    )
    logging.info(
        f'Updated the working directory of {update_result.modified_count} experiments.'
    )
