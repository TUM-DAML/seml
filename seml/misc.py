import subprocess
import logging
from sacred import Experiment


def sacred_arguments_from_config_dict(config):
    config_strings = [f'{key}="{val}"' for key, val in config.items()]
    return " ".join(config_strings)


def get_cmd_from_exp_dict(exp):
    if 'executable' not in exp['seml']:
        raise ValueError(f"No executable found for experiment {exp['_id']}. Aborting.")
    exe = exp['seml']['executable']
    exp['config']['db_collection'] = exp['seml']['db_collection']
    exp['config']['overwrite'] = exp['_id']
    configs_string = sacred_arguments_from_config_dict(exp['config'])
    return f"python {exe} with {configs_string}"


def get_slurm_jobs():
    squeue_out = subprocess.check_output("squeue -a -t pending,running -h -o %A", shell=True)
    return [int(line) for line in squeue_out.split(b'\n')[:-1]]


def setup_logger(ex: Experiment):
    # set up the logger
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
            fmt='%(asctime)s (%(levelname)s): %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')
    ex.logger = logger


def s_if(n):
    return '' if n == 1 else 's'


def get_default_sbatch_dict():
    sbatch_dict = {
        '--time': '0-08:00',
        '--cpus-per-task': 1,
        '--nodes': 1,
    }
    return sbatch_dict
