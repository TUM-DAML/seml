import urllib.parse
import logging
from sacred.observers.base import RunObserver, td_format
from sacred.config.config_files import load_config_file
import json

from seml.database import get_mongodb_config
from seml.settings import SETTINGS

__all__ = ['create_mongodb_observer', 'create_slack_observer', 'create_neptune_observer', 'create_mattermost_observer',
           'MattermostObserver']


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


def create_mattermost_observer(webhook=None, channel=None):

    if channel is None:
        if "OBSERVERS" in SETTINGS and "MATTERMOST" in SETTINGS.OBSERVERS:
            if channel is None and "DEFAULT_CHANNEL" in SETTINGS.OBSERVERS.MATTERMOST:
                channel = SETTINGS.OBSERVERS.MATTERMOST.DEFAULT_CHANNEL
    if webhook is None:
        if "OBSERVERS" in SETTINGS and "MATTERMOST" in SETTINGS.OBSERVERS:
            if channel is None and "DEFAULT_CHANNEL" in SETTINGS.OBSERVERS.MATTERMOST:
                channel = SETTINGS.OBSERVERS.MATTERMOST.DEFAULT_CHANNEL
            if "WEBHOOK" in SETTINGS.OBSERVERS.MATTERMOST:
                webhook = SETTINGS.OBSERVERS.MATTERMOST.WEBHOOK

    mattermost_observer = MattermostObserver(webhook, channel=channel)
    return mattermost_observer


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


class MattermostObserver(RunObserver):
    """
    Based on Sacred's Slack observer: https://github.com/IDSIA/sacred/blob/master/sacred/observers/slack.py
    Sends a message to Mattermost upon completion/failing of an experiment.
    """

    @classmethod
    def from_config(cls, filename):
        """
        Create a MattermostObserver from a given configuration file.

        The file can be in any format supported by Sacred
        (.json, .pickle, [.yaml]).
        It has to specify a ``webhook_url`` and can optionally set
        ``bot_name``, ``icon``, ``completed_text``, ``interrupted_text``, and
        ``failed_text``.
        """
        return cls(**load_config_file(filename))

    def __init__(
        self,
        webhook_url,
        channel=None,
        bot_name="sacredbot",
        icon=":angel:",
        completed_text=None,
        interrupted_text=None,
        failed_text=None,
        notify_on_completed=True,
        notify_on_interrupted=True,
        notify_on_failed=True,
    ):
        """
        Create a Sacred observer that will send notifications to Mattermost.
        Parameters
        ----------
        webhook_url: str
            The webhook for the bot account.
        channel: str
            The channel to which to send notifications. To send direct messages, set to @username.
        bot_name: str
            The name of the bot.
        icon: str
            The icon of the bot.
        completed_text: str
            Text to be sent upon completion.
        interrupted_text: str
            Text to be sent upon interruption.
        failed_text: str
            Text to be sent upon failure.
        notify_on_completed: bool
            Whether to send a notification upon completion.
        notify_on_interrupted: bool
            Whether to send a notification when the experiment is interrupted.
        notify_on_failed: bool
            Whether to send a notification when the experiment fails.
        """
        self.webhook_url = webhook_url
        self.bot_name = bot_name
        self.icon = icon
        self.completed_text = completed_text or (
            ":white_check_mark: *{experiment[name]}* "
            "completed after _{elapsed_time}_ with result=`{result}`"
        )
        self.interrupted_text = interrupted_text or (
            ":warning: *{experiment[name]}* " "interrupted after _{elapsed_time}_"
        )
        self.failed_text = failed_text or (
            ":x: *{experiment[name]}* failed after " "_{elapsed_time}_ with `{error}`"
        )
        self.run = None
        self.channel = channel

        self.notify_on_completed = notify_on_completed
        self.notify_on_failed = notify_on_failed
        self.notify_on_interrupted = notify_on_interrupted

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        self.run = {
            "_id": _id,
            "config": config,
            "start_time": start_time,
            "experiment": ex_info,
            "command": command,
            "host_info": host_info,
        }

    def get_completed_text(self):
        return self.completed_text.format(**self.run)

    def get_interrupted_text(self):
        return self.interrupted_text.format(**self.run)

    def get_failed_text(self):
        return self.failed_text.format(**self.run)

    def completed_event(self, stop_time, result):
        import requests

        if self.completed_text is None or not self.notify_on_completed:
            return

        self.run["result"] = result
        self.run["stop_time"] = stop_time
        self.run["elapsed_time"] = td_format(stop_time - self.run["start_time"])

        data = {
            "username": self.bot_name,
            "icon_emoji": self.icon,
            "text": self.get_completed_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def interrupted_event(self, interrupt_time, status):
        import requests

        if self.interrupted_text is None or not self.notify_on_interrupted:
            return

        self.run["status"] = status
        self.run["interrupt_time"] = interrupt_time
        self.run["elapsed_time"] = td_format(interrupt_time - self.run["start_time"])

        data = {
            "username": self.bot_name,
            "icon_emoji": self.icon,
            "text": self.get_interrupted_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def failed_event(self, fail_time, fail_trace):
        import requests

        if self.failed_text is None or not self.notify_on_failed:
            return

        self.run["fail_trace"] = fail_trace
        self.run["error"] = fail_trace[-1].strip()
        self.run["fail_time"] = fail_time
        self.run["elapsed_time"] = td_format(fail_time - self.run["start_time"])

        data = {
            "username": self.bot_name,
            "icon_emoji": self.icon,
            "text": self.get_failed_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)