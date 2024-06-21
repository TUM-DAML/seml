import json
import re
from datetime import datetime, timedelta, timezone

from bson import json_util
from sacred.config.config_files import load_config_file
from sacred.observers.base import RunObserver, td_format

from seml.utils.json import NumpyEncoder


def to_local_timezone(dtime):
    return dtime.replace(tzinfo=timezone.utc).astimezone(tz=None)


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
        bot_name='sacredbot',
        icon=':angel:',
        notify_on_started=False,
        notify_on_completed=True,
        notify_on_failed=True,
        notify_on_interrupted=False,
        heartbeat_interval=None,
        started_text=None,
        completed_text=None,
        failed_text=None,
        interrupted_text=None,
        heartbeat_text=None,
        convert_utc_to_local_timezone=True,
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
        notify_on_started: bool
            Whether to send a notification when the experiment starts.
        notify_on_completed: bool
            Whether to send a notification upon completion.
        notify_on_failed: bool
            Whether to send a notification when the experiment fails.
        notify_on_interrupted: bool
            Whether to send a notification when the experiment is interrupted.
        heartbeat_interval: str
            String in the format D-hh:mm indicating how often to send heartbeat notifications. If None, send no
            notifications.
        started_text: str
            Text to be sent when the experiment starts. If None, this default will be used:
            ":hourglass_flowing_sand: *{experiment[name]}* "
            "started on host `{host_info[hostname]}` at _{start_time}_."
        completed_text: str
            Text to be sent upon completion. If None, this default will be used:
            ":white_check_mark: *{experiment[name]}* "
            "completed after _{elapsed_time}_ with result: \n```json\n{result}\n````\n"
        failed_text: str
            Text to be sent upon failure. If None, this default will be used:
            ":x: *{experiment[name]}* failed after " "_{elapsed_time}_ with `{error}`"
        interrupted_text: str
            Text to be sent upon interruption. If None, this default will be used:
            ":warning: *{experiment[name]}* " "interrupted after _{elapsed_time}_"
        heartbeat_text: str
            Text to be sent to notify that the experiment is still running. If None, this default will be used:
            ":heartpulse: *{experiment[name]}* has been up and running for " "_{elapsed_time}_. "
            "Current info dict: \n```json\n{info}\n```\n"
            "Next heartbeat will be sent in about _{heartbeat_interval}_, i.e., on _{next_heartbeat_date}_."
        convert_utc_to_local_timezone: bool
            Whether to convert UTC times to local timezone in the notifications.
        """

        self.webhook_url = webhook_url
        self.bot_name = bot_name
        self.icon = icon
        self.completed_text = completed_text or (
            ':white_check_mark: *{experiment[name]}* '
            'completed after _{elapsed_time}_ with result: \n```json\n{result}\n````\n'
        )
        self.started_text = started_text or (
            ':hourglass_flowing_sand: *{experiment[name]}* '
            'started on host `{host_info[hostname]}` at _{start_time}_.'
        )
        self.interrupted_text = interrupted_text or (
            ':warning: *{experiment[name]}* ' 'interrupted after _{elapsed_time}_'
        )
        self.failed_text = failed_text or (
            ':x: *{experiment[name]}* failed after ' '_{elapsed_time}_ with `{error}`'
        )
        self.heartbeat_text = heartbeat_text or (
            ':heartpulse: *{experiment[name]}* has been up and running for '
            '_{elapsed_time}_. '
            'Current info dict: \n```json\n{info}\n```\n'
            'Next heartbeat will be sent in about _{heartbeat_interval}_, i.e., on _{next_heartbeat_date}_.'
        )

        self.run = None
        self.channel = channel

        self.notify_on_completed = notify_on_completed
        self.notify_on_failed = notify_on_failed
        self.notify_on_interrupted = notify_on_interrupted
        self.notify_on_started = notify_on_started
        self.notify_on_heartbeat = False
        self.last_heartbeat_notification = None
        self.convert_utc_to_local_timezone = convert_utc_to_local_timezone

        self.heartbeat_interval = None
        if heartbeat_interval is not None:
            # unfortunately datetime.strptime() doesn't work with timedeltas, so we parse the date ourselves:
            pattern = re.compile('([0-9]+)-([0-9]+):([0-9]+)')
            days, hours, minutes = pattern.match(heartbeat_interval).groups()
            self.heartbeat_interval = timedelta(
                days=int(days), hours=int(hours), minutes=int(minutes)
            )
            self.notify_on_heartbeat = True

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        import requests

        if self.convert_utc_to_local_timezone:
            start_time = to_local_timezone(start_time)

        self.run = {
            '_id': _id,
            'config': config,
            'start_time': start_time,
            'experiment': ex_info,
            'command': command,
            'host_info': host_info,
        }
        if self.heartbeat_interval is not None:
            self.run['heartbeat_interval'] = td_format(self.heartbeat_interval)
            self.last_heartbeat_notification = start_time

        if not self.notify_on_started:
            return

        data = {
            'username': self.bot_name,
            'icon_emoji': self.icon,
            'text': self.get_started_text(),
        }

        if self.channel is not None:
            data['channel'] = self.channel
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def get_completed_text(self):
        return self.completed_text.format(**self.run)

    def get_started_text(self):
        return self.started_text.format(**self.run)

    def get_interrupted_text(self):
        return self.interrupted_text.format(**self.run)

    def get_failed_text(self):
        return self.failed_text.format(**self.run)

    def get_heartbeat_text(self):
        return self.heartbeat_text.format(**self.run)

    def completed_event(self, stop_time, result):
        import requests

        if self.completed_text is None or not self.notify_on_completed:
            return

        if self.convert_utc_to_local_timezone:
            stop_time = to_local_timezone(stop_time)

        self.run['result'] = json.dumps(result, indent=4, cls=NumpyEncoder)
        self.run['stop_time'] = stop_time
        self.run['elapsed_time'] = td_format(stop_time - self.run['start_time'])

        data = {
            'username': self.bot_name,
            'icon_emoji': self.icon,
            'text': self.get_completed_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def interrupted_event(self, interrupt_time, status):
        import requests

        if self.interrupted_text is None or not self.notify_on_interrupted:
            return

        if self.convert_utc_to_local_timezone:
            interrupt_time = to_local_timezone(interrupt_time)

        self.run['status'] = status
        self.run['interrupt_time'] = interrupt_time
        self.run['elapsed_time'] = td_format(interrupt_time - self.run['start_time'])

        data = {
            'username': self.bot_name,
            'icon_emoji': self.icon,
            'text': self.get_interrupted_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def failed_event(self, fail_time, fail_trace):
        import requests

        if self.failed_text is None or not self.notify_on_failed:
            return
        if self.convert_utc_to_local_timezone:
            fail_time = to_local_timezone(fail_time)

        self.run['fail_trace'] = '\n'.join(fail_trace)
        self.run['error'] = fail_trace[-1].strip()
        self.run['fail_time'] = fail_time
        self.run['elapsed_time'] = td_format(fail_time - self.run['start_time'])

        data = {
            'username': self.bot_name,
            'icon_emoji': self.icon,
            'text': self.get_failed_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def heartbeat_event(self, info, captured_out, beat_time, result):
        import requests

        if self.heartbeat_text is None or not self.notify_on_heartbeat:
            return

        if self.convert_utc_to_local_timezone:
            beat_time = to_local_timezone(beat_time)

        if beat_time < self.last_heartbeat_notification + self.heartbeat_interval:
            return

        next_heartbeat_notification = beat_time + self.heartbeat_interval
        self.run['next_heartbeat_date'] = datetime.strftime(
            next_heartbeat_notification, '%Y-%m-%d %H:%M'
        )
        self.run['elapsed_time'] = td_format(beat_time - self.run['start_time'])
        self.run['info'] = json.dumps(info, indent=4, default=json_util.default)

        data = {
            'username': self.bot_name,
            'icon_emoji': self.icon,
            'text': self.get_heartbeat_text(),
        }
        if self.channel is not None:
            data['channel'] = self.channel
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        requests.post(self.webhook_url, data=json.dumps(data), headers=headers)
        self.last_heartbeat_notification = beat_time
