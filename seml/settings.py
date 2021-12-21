import imp
from munch import munchify
from pathlib import Path

from seml.utils import merge_dicts

__all__ = ("SETTINGS",)

SETTINGS = munchify(
    {
        # Location of user-specific settings.py file containing a SETTINGS dict.
        # With this dict you can change anything that is set here, conveniently from your home directory.
        # Default: $HOME/.config/seml/settings.py
        "USER_SETTINGS_PATH": Path.home() / ".config/seml/settings.py",

        "DATABASE": {
            # location of the MongoDB config. Default: $HOME/.config/seml/monogdb.config
            "MONGODB_CONFIG_PATH": Path.home() / ".config/seml/mongodb.config"
        },
        "SLURM_DEFAULT": {
            'experiments_per_job': 1,
            'sbatch_options': {
                'time': '0-08:00',
                'nodes': 1,
                'cpus-per-task': 1,
                'mem': '8G',
                },
        },
        "SBATCH_OPTIONS_TEMPLATES": {
            # This is a special template used for `seml jupyter`
            "JUPYTER": {
                'cpus-per-task': 2,
                'mem': '16G',
                'gres': "gpu:1",
                'qos': 'interactive',
                'job-name': 'jupyter',
                'output': 'jupyter-%j.out',
            },
            # Extend this with your custom templates.
            "GPU": {
                'cpus-per-task': 2,
                'mem': '16G',
                'gres': "gpu:1",
            },
        },
        "STATES": {
            "STAGED": ["STAGED", "QUEUED"],  # QUEUED for backward compatibility
            "PENDING": ["PENDING"],
            "RUNNING": ["RUNNING"],
            "FAILED": ["FAILED"],
            "KILLED": ["KILLED"],
            "INTERRUPTED": ["INTERRUPTED"],
            "COMPLETED": ["COMPLETED"],
        },
        "SLURM_STATES": {
            "PENDING": ["PENDING", "CONFIGURING", "REQUEUE_FED", "REQUEUE_HOLD", "REQUEUED", "RESIZING"],
            "RUNNING": ["RUNNING", "SIGNALING"],  # Python code can still be executed while in SIGNALING
            "PAUSED": ["STOPPED", "SUSPENDED", "SPECIAL_EXIT"],
            "INTERRUPTED": ["CANCELLED"],  # Caused by user command
            "FAILED": ["FAILED", "BOOT_FAIL", "DEADLINE", "NODE_FAIL",
                       "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"],
            # REVOKED is not failed, but would need code that handles multi-cluster operation
            "COMPLETED": ["COMPLETED", "COMPLETING", "STAGE_OUT"],
        },
        "VALID_SEML_CONFIG_VALUES": ['executable', 'name', 'output_dir',
                                     'conda_environment', 'project_root_dir'],
        "VALID_SLURM_CONFIG_VALUES": ['experiments_per_job', 'max_simultaneous_jobs',
                                      'sbatch_options_template', 'sbatch_options'],
        "LOGIN_NODE_NAMES": ["fs"],

        "OBSERVERS": {
            "NEPTUNE": {
                "AUTH_TOKEN": "YOUR_AUTH_TOKEN",
            },
            "SLACK": {
                "WEBHOOK": "YOUR_WEBHOOK",
            },
            "MATTERMOST": {
                "WEBHOOK": "YOUR_WEBHOOK",
                "DEFAULT_CHANNEL": "YOUR_DEFAULT_CHANNEL",
            }
        },

        "CONFIRM_CANCEL_THRESHOLD": 10,
        "CONFIRM_DELETE_THRESHOLD": 10,
        "CONFIRM_RESET_THRESHOLD": 10,

        "SETUP_COMMAND": "",
        "END_COMMAND": "",
    },
)

# Load user settings
if SETTINGS.USER_SETTINGS_PATH.exists():
    user_settings_source = imp.load_source('SETTINGS', str(SETTINGS.USER_SETTINGS_PATH))
    SETTINGS = munchify(merge_dicts(SETTINGS, user_settings_source.SETTINGS))

SETTINGS.SLURM_STATES.ACTIVE = (
        SETTINGS.SLURM_STATES.PENDING + SETTINGS.SLURM_STATES.RUNNING + SETTINGS.SLURM_STATES.PAUSED)
