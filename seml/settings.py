from munch import munchify
from pathlib import Path

__all__ = ("SETTINGS",)


SETTINGS = munchify(
    {
        "DATABASE": {
            # location of the MongoDB config. Default: $HOME/.config/seml/monogdb.config
            "MONGODB_CONFIG_PATH": f'{str(Path.home())}/.config/seml/mongodb.config'
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
    }
)
