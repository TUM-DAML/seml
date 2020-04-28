from munch import munchify
import os
import seml
__all__ = ("SETTINGS",)
seml_base = os.path.dirname(os.path.abspath(seml.__file__))

SETTINGS = munchify(
    {
        "DATABASE": {
            # location of the MongoDB config. Default: /path/to/seml/monogdb.config
            "MONGODB_CONFIG_PATH": f'{seml_base}/mongodb.config'
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
