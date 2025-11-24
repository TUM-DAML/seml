from __future__ import annotations

from pathlib import Path
from runpy import run_path
from typing import Any, Dict, List, Mapping, cast

import typer
from typing_extensions import TypeVar

from seml.cli_utils.module_hider import ModuleHider
from seml.document import SBatchOptions, SlurmConfig
from seml.utils import merge_dicts

# The YAML, json import is rather slow
with ModuleHider(
    'yaml',
    'json',
    'simplejson',
    'importlib_metadata',
    'importlib.metadata',
):
    from munch import munchify

T = TypeVar('T', default=Any)


class SettingsDict(Mapping[str, T]):
    def __getattr__(self, name: str) -> T: ...


class DatabaseSettings(SettingsDict):
    MONGODB_CONFIG_PATH: Path


class States(SettingsDict[List[str]]):
    STAGED: list[str]
    PENDING: list[str]
    RUNNING: list[str]
    RESCHEDULED: list[str]
    FAILED: list[str]
    KILLED: list[str]
    INTERRUPTED: list[str]
    COMPLETED: list[str]


class SlurmStates(SettingsDict[List[str]]):
    PENDING: list[str]
    RUNNING: list[str]
    PAUSED: list[str]
    INTERRUPTED: list[str]
    FAILED: list[str]
    COMPLETED: list[str]
    ACTIVE: list[str]


class FileObserverSettings(SettingsDict[Any]):
    DEFAULT_BASE_DIR: str


class SlackObserverSettings(SettingsDict[str]):
    WEBHOOK: str


class MattermostObserverSettings(SettingsDict[str]):
    WEBHOOK: str
    DEFAULT_CHANNEL: str


class NetupeObserverSettings(SettingsDict[str]):
    AUTH_TOKEN: str


class ObserverSettings(SettingsDict[Dict[str, str]]):
    FILE: FileObserverSettings
    SLACK: SlackObserverSettings
    NEPTUNE: NetupeObserverSettings
    MATTERMOST: MattermostObserverSettings


class NamedConfigSettings(SettingsDict[str]):
    PREFIX: str
    KEY_NAME: str
    KEY_PRIORITY: str


class ConfirmThresholdSettings(SettingsDict[int]):
    DELETE: int
    RESET: int
    CANCEL: int
    DESCRIPTION_DELETE: int
    DESCRIPTION_UPDATE: int


class ExperimentSettings(SettingsDict):
    CAPTURE_OUTPUT: bool
    TERMINAL_WIDTH: int
    ENVIRONMENT: dict[str, str]


class MigrationSettings(SettingsDict):
    SKIP: bool
    YES: bool
    BACKUP: bool
    BACKUP_TMP: str


class SSHForwardSettings(SettingsDict):
    LOCK_FILE: str
    RETRIES_MAX: int
    RETRIES_DELAY: int
    LOCK_TIMEOUT: int
    HEALTH_CHECK_INTERVAL: int


class Settings(SettingsDict):
    USER_SETTINGS_PATH: Path
    TMP_DIRECTORY: str
    TEMPLATE_REMOTE: str
    DATABASE: DatabaseSettings
    SLURM_DEFAULT: SlurmConfig
    SBATCH_OPTIONS_TEMPLATES: SettingsDict[SBatchOptions]
    STATES: States
    SLURM_STATES: SlurmStates
    VALID_SEML_CONFIG_VALUES: list[str]
    SEML_CONFIG_VALUE_VERSION: str
    VALID_SLURM_CONFIG_VALUES: list[str]
    LOGIN_NODE_NAMES: list[str]
    OBSERVERS: ObserverSettings
    CONFIG_EXCLUDE_KEYS: list[str]
    CONFIG_KEY_SEED: str
    ALLOW_INTERPOLATION_IN: list[str]
    NAMED_CONFIG: NamedConfigSettings
    CONFIRM_THRESHOLD: ConfirmThresholdSettings
    EXPERIMENT: ExperimentSettings
    MIGRATION: MigrationSettings
    CANCEL_TIMEOUT: int
    CONFIG_RESOLUTION_PROGRESS_BAR_THRESHOLD: int
    AUTOCOMPLETE_CACHE_ALIVE_TIME: int
    SETUP_COMMAND: str
    END_COMMAND: str
    SSH_FORWARD: SSHForwardSettings


SETTINGS: Settings

__all__ = ('SETTINGS',)

APP_DIR = Path(typer.get_app_dir('seml'))

SETTINGS = cast(
    Settings,
    munchify(
        {
            # Location of user-specific settings.py file containing a SETTINGS dict.
            # With this dict you can change anything that is set here, conveniently from your home directory.
            # Default: $HOME/.config/seml/settings.py
            'USER_SETTINGS_PATH': APP_DIR / 'settings.py',
            # Directory which is used on the compute nodes to dump scripts and Python code.
            # Only change this if you know what you're doing.
            'TMP_DIRECTORY': '/tmp',
            'TEMPLATE_REMOTE': 'https://github.com/TUM-DAML/seml-templates.git',
            'DATABASE': {
                # location of the MongoDB config. Default: $HOME/.config/seml/monogdb.config
                'MONGODB_CONFIG_PATH': APP_DIR / 'mongodb.config'
            },
            'SLURM_DEFAULT': {
                'experiments_per_job': 1,
                'sbatch_options': {
                    'time': '0-08:00',
                    'nodes': 1,
                    'cpus-per-task': 1,
                    'mem': '8G',
                },
            },
            'SBATCH_OPTIONS_TEMPLATES': {
                # This is a special template used for `seml jupyter`
                'JUPYTER': {
                    'cpus-per-task': 2,
                    'mem': '16G',
                    'gres': 'gpu:1',
                    'qos': 'interactive',
                    'job-name': 'jupyter',
                    'output': 'jupyter-%j.out',
                    'partition': 'gpu_gtx1080',
                },
                # Extend this with your custom templates.
                'GPU': {
                    'cpus-per-task': 2,
                    'mem': '16G',
                    'gres': 'gpu:1',
                },
            },
            'STATES': {
                'STAGED': ['STAGED', 'QUEUED'],  # QUEUED for backward compatibility
                'PENDING': ['PENDING'],
                'RUNNING': ['RUNNING'],
                'FAILED': ['FAILED'],
                'KILLED': ['KILLED'],
                'INTERRUPTED': ['INTERRUPTED'],
                'COMPLETED': ['COMPLETED'],
                'RESCHEDULED': ['RESCHEDULED'],
            },
            'SLURM_STATES': {
                'PENDING': [
                    'PENDING',
                    'CONFIGURING',
                    'REQUEUE_FED',
                    'REQUEUE_HOLD',
                    'REQUEUED',
                    'RESIZING',
                ],
                'RUNNING': [
                    'RUNNING',
                    'SIGNALING',
                ],  # Python code can still be executed while in SIGNALING
                'PAUSED': ['STOPPED', 'SUSPENDED', 'SPECIAL_EXIT'],
                'INTERRUPTED': ['CANCELLED'],  # Caused by user command
                'FAILED': [
                    'FAILED',
                    'BOOT_FAIL',
                    'DEADLINE',
                    'NODE_FAIL',
                    'OUT_OF_MEMORY',
                    'PREEMPTED',
                    'REVOKED',
                    'TIMEOUT',
                ],
                # REVOKED is not failed, but would need code that handles multi-cluster operation
                'COMPLETED': ['COMPLETED', 'COMPLETING', 'STAGE_OUT'],
            },
            'VALID_SEML_CONFIG_VALUES': [
                'executable',
                'name',
                'output_dir',
                'conda_environment',
                'project_root_dir',
                'description',
                'stash_all_py_files',
                'reschedule_timeout',
            ],
            'SEML_CONFIG_VALUE_VERSION': 'version',
            'VALID_SLURM_CONFIG_VALUES': [
                'experiments_per_job',
                'max_simultaneous_jobs',
                'sbatch_options_template',
                'sbatch_options',
            ],
            'LOGIN_NODE_NAMES': ['fs'],
            'OBSERVERS': {
                'NEPTUNE': {
                    'AUTH_TOKEN': 'YOUR_AUTH_TOKEN',
                },
                'SLACK': {
                    'WEBHOOK': 'YOUR_WEBHOOK',
                },
                'MATTERMOST': {
                    'WEBHOOK': 'YOUR_WEBHOOK',
                    'DEFAULT_CHANNEL': 'YOUR_DEFAULT_CHANNEL',
                },
            },
            'CONFIG_EXCLUDE_KEYS': [
                '__doc__',
                'db_collection',
                'overwrite',
            ],  # keys that will be excluded from resolved configurations, sacred for some reason captures the docstring attribute
            # Which key is treated as the experiment seed
            'CONFIG_KEY_SEED': 'seed',
            'ALLOW_INTERPOLATION_IN': [
                'seml.description',
                'config',
            ],  # in which fields to allow variable interpolation
            'NAMED_CONFIG': {
                'PREFIX': '+',  # prefix for all named configuration parameters
                'KEY_NAME': 'name',  # key that identifies the name of a named config
                'KEY_PRIORITY': 'priority',  # key that identifies the priority of a named config
            },
            'CONFIRM_THRESHOLD': {
                'DELETE': 10,
                'RESET': 10,
                'CANCEL': 10,
                'DESCRIPTION_DELETE': 10,
                'DESCRIPTION_UPDATE': 10,
            },
            'EXPERIMENT': {
                'CAPTURE_OUTPUT': False,  # whether to capture the output of the experiment in the database
                'TERMINAL_WIDTH': 80,  # width of the terminal for rich output
                'ENVIRONMENT': {},  # Additional environment variables to set for the experiment - these override existing ones
            },
            'MIGRATION': {
                'SKIP': False,  # always ignore migrations, changing this most likely breaks compatibility!
                'YES': False,  # always confirm migrations
                'BACKUP': False,  # always backup the database before running migrations
                'BACKUP_TMP': '{collection}_backup_{time}',  # format for backup collections
            },
            'CANCEL_TIMEOUT': 60,  # wait up to 60s for canceling an experiment
            'CONFIG_RESOLUTION_PROGRESS_BAR_THRESHOLD': 25,
            'AUTOCOMPLETE_CACHE_ALIVE_TIME': 60 * 60 * 24,  # one day
            'SETUP_COMMAND': '',
            'END_COMMAND': '',
            'SSH_FORWARD': {
                'LOCK_FILE': '/tmp/seml_ssh_forward.lock',
                'RETRIES_MAX': 6,
                'RETRIES_DELAY': 1,
                'LOCK_TIMEOUT': 30,
                'HEALTH_CHECK_INTERVAL': 10,
            },
        },
    ),
)

# Load user settings
if SETTINGS.USER_SETTINGS_PATH.exists():
    user_settings_source = run_path(str(SETTINGS.USER_SETTINGS_PATH))
    SETTINGS = cast(
        Settings,
        munchify(merge_dicts(SETTINGS, user_settings_source['SETTINGS'])),  # type: ignore
    )

SETTINGS.SLURM_STATES.ACTIVE = (
    SETTINGS.SLURM_STATES.PENDING
    + SETTINGS.SLURM_STATES.RUNNING
    + SETTINGS.SLURM_STATES.PAUSED
)
