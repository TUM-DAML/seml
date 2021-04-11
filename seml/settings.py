from munch import munchify
import os
from pathlib import Path
import seml
__all__ = ("SETTINGS",)

SETTINGS = munchify(
    {
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
                'mem': 8000,
                },
        },
        "SBATCH_OPTIONS_TEMPLATES": {
            # Extend this with your custom templates.
            "GPU": {
                'gres': "gpu:1",
                'mem': 16000,
                'nodes': 1,
                'cpus-per-task': 2,
            },
            "JUPYTER_JOB": {
                'time': '0-08:00',
                'nodes': 1,
                'cpus-per-task': 1,
                'mem': 16000,
                'gres': "gpu:1",
                'qos': 'interactive',
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
        "SLURM_ACTIVE_STATES": ['CONFIGURING', 'PENDING', 'RUNNING', 'REQUEUE_FED',
                                'REQUEUE_HOLD', 'REQUEUED', 'RESIZING', 'SUSPENDED'],

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
    },
)
