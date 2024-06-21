from pathlib import Path
from runpy import run_path

import seml.utils.typer as typer
from seml.utils import merge_dicts
from seml.utils.module_hider import ModuleHider

# The YAML, json import is rather slow
with ModuleHider(
    'yaml',
    'json',
    'simplejson',
    'importlib_metadata',
    'importlib.metadata',
):
    from munch import munchify

__all__ = ('SETTINGS',)

APP_DIR = Path(typer.get_app_dir('seml'))

SETTINGS = munchify(
    {
        # Location of user-specific settings.py file containing a SETTINGS dict.
        # With this dict you can change anything that is set here, conveniently from your home directory.
        # Default: $HOME/.config/seml/settings.py
        'USER_SETTINGS_PATH': APP_DIR / 'settings.py',
        # Directory which is used on the compute nodes to dump scripts and Python code.
        # Only change this if you know what you're doing.
        'TMP_DIRECTORY': '/tmp',
        'TEMPLATE_REMOTE': 'https://github.com/TUM-DAML/seml-templates.git',
        'CODE_CHECKPOINT_REMOVE_SRC_DIRECTORY': True,
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
        ],
        'SEML_CONFIG_VALUE_VERSION': 'version',
        'VALID_SLURM_CONFIG_VALUES': [
            'experiments_per_job',
            'max_simultaneous_jobs',
            'sbatch_options_template',
            'sbatch_options',
            'overrides',
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
            '__doc__'
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
        },
        'MIGRATION': {
            'SKIP': False,  # alwys ignore migrations, changing this most likely breaks compatibility!
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
)

# Load user settings
if SETTINGS.USER_SETTINGS_PATH.exists():
    user_settings_source = run_path(str(SETTINGS.USER_SETTINGS_PATH))
    SETTINGS = munchify(merge_dicts(SETTINGS, user_settings_source['SETTINGS']))

SETTINGS.SLURM_STATES.ACTIVE = (
    SETTINGS.SLURM_STATES.PENDING
    + SETTINGS.SLURM_STATES.RUNNING
    + SETTINGS.SLURM_STATES.PAUSED
)
