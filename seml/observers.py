import urllib.parse
import logging

from seml.database import get_mongodb_config
from seml.settings import SETTINGS

__all__ = ['create_mongodb_observer', 'create_slack_observer', 'create_neptune_observer']


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
    from neptunecontrib.monitoring.sacred import NeptuneObserver

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
