import urllib.parse
import logging
import os

from seml.database import get_mongodb_config
from seml.settings import SETTINGS

__all__ = ['create_mongodb_observer', 'create_slack_observer', 'create_neptune_observer',
           'create_file_storage_observer', 'add_to_file_storage_observer']


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
        ID of an experiment to overwrite, e.g. a staged or failed experiment.

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


def create_file_storage_observer(runs_folder_name, basedir=None, **kwargs):
    from sacred.observers import FileStorageObserver
    if basedir is None:
        basedir = SETTINGS.OBSERVERS.FILE.DEFAULT_BASE_DIR
        logging.info(f"Starting file observer in location {basedir}/{runs_folder_name}. To change the default base "
                     f"directory, modify entry SETTINGS.OBSERVERS.FILE.DEFAULT_BASE_DIR in seml/settings.py.")
    else:
        logging.info(f"Starting file observer in location {basedir}/{runs_folder_name}.")
    observer = FileStorageObserver(f"{basedir}/{runs_folder_name}", kwargs)
    return observer


def add_to_file_storage_observer(file, experiment, delete_local_file=False):
    """

    Parameters
    ----------
    file: str
        Path to file to add to the file storage observer.
    experiment: sacred.experiment.Experiment
        The Sacred Experiment containing the FileStorageObserver.
    delete_local_file: bool, default: False
        If True, delete the local file after copying it to the FileStorageObserver.

    Returns
    -------
    None
    """
    has_file_observer = False
    for obs in experiment.current_run.observers:
        if "FileStorageObserver" in str(type(obs)):
            obs.artifact_event(name=None, filename=file, )
            has_file_observer = True
    if not has_file_observer:
        logging.warning(
            "'add_to_file_storage_observer' was called but found no FileStorageObserver for the experiment."
                        )
    if delete_local_file:
        os.remove(file)

def create_slack_observer(webhook=None):
    from sacred.observers import SlackObserver

    slack_obs = None
    if webhook is None:
        if "OBSERVERS" in SETTINGS and "SLACK" in SETTINGS.OBSERVERS:
            if "WEBHOOK" in SETTINGS.OBSERVERS.SLACK:
                webhook = SETTINGS.OBSERVERS.SLACK.WEBHOOK
                slack_obs = SlackObserver(webhook)
    else:
        slack_obs = SlackObserver(webhook)

    if slack_obs is None:
        logging.warning('Failed to create Slack observer.')
    return slack_obs


def create_neptune_observer(project_name, api_token=None,
                            source_extensions=['**/*.py', '**/*.yaml', '**/*.yml']):
    try:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
    except ImportError:
        logging.error("Could not import neptunecontrib. Install via `pip install neptune-contrib`.")

    if api_token is None:
        if "OBSERVERS" in SETTINGS and "NEPTUNE" in SETTINGS.OBSERVERS:
            if "AUTH_TOKEN" in SETTINGS.OBSERVERS.NEPTUNE:
                api_token = SETTINGS.OBSERVERS.NEPTUNE.AUTH_TOKEN
    else:
        api_token = SETTINGS.OBSERVERS.NEPTUNE.AUTH_TOKEN
    if api_token is None:
        logging.info('No API token for Nepune provided. Trying to use environment variable NEPTUNE_API_TOKEN.')
    neptune_obs = NeptuneObserver(api_token=api_token, project_name=project_name, source_extensions=source_extensions)
    return neptune_obs
