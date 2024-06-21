import logging
import os

from seml.database import get_mongo_client, get_mongodb_config
from seml.settings import SETTINGS
from seml.utils import warn_multiple_calls

__all__ = [
    'create_mongodb_observer',
    'create_slack_observer',
    'create_neptune_observer',
    'create_file_storage_observer',
    'add_to_file_storage_observer',
    'create_mattermost_observer',
]


@warn_multiple_calls(
    'Created {num_calls} MongoDB observers.\n'
    'This might not be intended.\n'
    'seml.experiment.Experiment creates one by default.\n'
    'Either disable the default observer (seml.experiment.Experiment(add_mongodb_observer=False)) \n'
    'or remove the explicit call to create_mongodb_observer.'
)
def create_mongodb_observer(collection, mongodb_config=None, overwrite=None):
    """Create a MongoDB observer for a Sacred experiment

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

    observer = MongoObserver(
        client=get_mongo_client(**mongodb_config),
        collection=collection,
        db_name=mongodb_config['db_name'],
        overwrite=overwrite,
    )
    return observer


def create_file_storage_observer(runs_folder_name, basedir=None, **kwargs):
    from sacred.observers import FileStorageObserver

    if basedir is None:
        basedir = SETTINGS.OBSERVERS.FILE.DEFAULT_BASE_DIR
        logging.info(
            f'Starting file observer in location {basedir}/{runs_folder_name}. To change the default base '
            f'directory, modify entry SETTINGS.OBSERVERS.FILE.DEFAULT_BASE_DIR in seml/settings.py.'
        )
    else:
        logging.info(
            f'Starting file observer in location {basedir}/{runs_folder_name}.'
        )
    observer = FileStorageObserver(f'{basedir}/{runs_folder_name}', kwargs)
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
        if 'FileStorageObserver' in str(type(obs)):
            obs.artifact_event(
                name=None,
                filename=file,
            )
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
        if 'OBSERVERS' in SETTINGS and 'SLACK' in SETTINGS.OBSERVERS:
            if 'WEBHOOK' in SETTINGS.OBSERVERS.SLACK:
                webhook = SETTINGS.OBSERVERS.SLACK.WEBHOOK
                slack_obs = SlackObserver(webhook)
    else:
        slack_obs = SlackObserver(webhook)

    if slack_obs is None:
        logging.warning('Failed to create Slack observer.')
    return slack_obs


def create_mattermost_observer(webhook=None, channel=None, **kwargs):
    """
    Create a Mattermost observer, which sends notifications via Mattermost.

    Parameters
    ----------
    webhook: str
        The webhook of the Mattermost instance. If you don't know this, ask your Mattermost administrator.
    channel: str
        The channel to send notifications to. This should usually be  @your.username (starting with '@').
    kwargs: dict
        Keyword arguments that are passed to the MattermostObserver. See MattermostObserver.__init__ for details.

    Returns
    -------
    The observer.

    """
    from seml.experiment.mattermost_observer import MattermostObserver

    if channel is None:
        if 'OBSERVERS' in SETTINGS and 'MATTERMOST' in SETTINGS.OBSERVERS:
            if channel is None and 'DEFAULT_CHANNEL' in SETTINGS.OBSERVERS.MATTERMOST:
                channel = SETTINGS.OBSERVERS.MATTERMOST.DEFAULT_CHANNEL
    if webhook is None:
        if 'OBSERVERS' in SETTINGS and 'MATTERMOST' in SETTINGS.OBSERVERS:
            if channel is None and 'DEFAULT_CHANNEL' in SETTINGS.OBSERVERS.MATTERMOST:
                channel = SETTINGS.OBSERVERS.MATTERMOST.DEFAULT_CHANNEL
            if 'WEBHOOK' in SETTINGS.OBSERVERS.MATTERMOST:
                webhook = SETTINGS.OBSERVERS.MATTERMOST.WEBHOOK
        else:
            raise ValueError('No webhook provided and none found in settings.py.')

    mattermost_observer = MattermostObserver(webhook, channel=channel, **kwargs)
    return mattermost_observer


def create_neptune_observer(
    project_name, api_token=None, source_extensions=['**/*.py', '**/*.yaml', '**/*.yml']
):
    try:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
    except ImportError:
        logging.error(
            'Could not import neptunecontrib. Install via `pip install neptune-contrib`.'
        )

    if api_token is None:
        if 'OBSERVERS' in SETTINGS and 'NEPTUNE' in SETTINGS.OBSERVERS:
            if 'AUTH_TOKEN' in SETTINGS.OBSERVERS.NEPTUNE:
                api_token = SETTINGS.OBSERVERS.NEPTUNE.AUTH_TOKEN
                # Ignore example token setting
                if api_token == 'YOUR_AUTH_TOKEN':
                    api_token = None

    if api_token is None:
        logging.info(
            'No API token for Neptune provided. Trying to use environment variable NEPTUNE_API_TOKEN.'
        )
    neptune_obs = NeptuneObserver(
        api_token=api_token,
        project_name=project_name,
        source_extensions=source_extensions,
    )
    return neptune_obs
