import sys
import importlib
import resource
import os
import subprocess
import logging
from pathlib import Path

from seml.settings import SETTINGS
import seml.database_utils as db_utils


def get_config_from_exp(exp, verbose=False, unobserved=False, post_mortem=False, debug=False, relative=False):
    if 'executable' not in exp['seml']:
        raise ValueError(f"No executable found for experiment {exp['_id']}. Aborting.")
    exe = exp['seml']['executable']
    if relative:
        exe = exp['seml']['executable_relative']

    config = exp['config']
    config['db_collection'] = exp['seml']['db_collection']
    if not unobserved:
        config['overwrite'] = exp['_id']
    config_strings = [f'{key}="{val}"' for key, val in config.items()]
    if not verbose:
        config_strings.append("--force")
    if unobserved:
        config_strings.append("--unobserved")
    if post_mortem:
        config_strings.append("--pdb")
    if debug:
        config_strings.append("--debug")

    return exe, config_strings


def get_slurm_jobs():
    try:
        squeue_out = subprocess.check_output("squeue -a -t pending,running -h -o %i -u `logname`", shell=True)
        return {int(job_str) for job_str in squeue_out.splitlines() if b'_' not in job_str}
    except subprocess.CalledProcessError:
        return set()


def get_slurm_arrays_tasks():
    """Get a dictionary of running/pending Slurm job arrays (as keys) and tasks (as values)

    job_dict
    -------
    job_dict: dict
        This dictionary has the job array IDs as keys and the values are
        a list of 1) the pending job task range and 2) a list of running job task IDs.

    """
    try:
        squeue_out = subprocess.check_output("squeue -a -t pending,running -h -o %i -u `logname`", shell=True)
        jobs = [job_str for job_str in squeue_out.splitlines() if b'_' in job_str]
        if len(jobs) > 0:
            array_ids_str, task_ids = zip(*[job_str.split(b'_') for job_str in jobs])
            job_dict = {}
            for i, task_range_str in enumerate(task_ids):
                array_id = int(array_ids_str[i])
                if array_id not in job_dict:
                    job_dict[array_id] = [range(0), []]
                if b'[' in task_range_str:
                    # There is only one task range, which is the overall pending job array
                    limits = task_range_str[1:-1].split(b'-')
                    task_range = range(int(limits[0]), int(limits[-1]) + 1)
                    job_dict[array_id][0] = task_range
                else:
                    # Single task IDs belong to running jobs
                    task_id = int(task_range_str)
                    job_dict[array_id][1].append(task_id)
            return job_dict
        else:
            return {}
    except subprocess.CalledProcessError:
        return {}


def setup_logger(ex, level='INFO'):
    """
    Set up logger for experiment.

    Parameters
    ----------
    ex: sacred.Experiment
    Sacred experiment to set the logger of.
    level: str or int
    Set the threshold for the logger to this level.

    Returns
    -------
    None

    """
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
            fmt='%(asctime)s (%(levelname)s): %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)
    ex.logger = logger


def s_if(n):
    return '' if n == 1 else 's'


def get_default_slurm_config():
    return {
            'experiments_per_job': 1,
            'sbatch_options': {
                'time': '0-08:00',
                'nodes': 1,
                'cpus-per-task': 1,
                'mem': 8000,
                },
            }


def unflatten(dictionary: dict, sep: str ='.'):
    """
    Turns a flattened dict into a nested one, e.g. {'a.b':2, 'c':3} becomes {'a':{'b': 2}, 'c': 3}
    From https://stackoverflow.com/questions/6037503/python-unflatten-dict

    Parameters
    ----------
    dictionary: dict to be un-flattened
    sep: separator with which the nested keys are separated

    Returns
    -------
    resultDict: the nested dictionary.
    """

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def flatten(dictionary: dict, parent_key: str='', sep: str='.'):
    """
    Flatten a nested dictionary, e.g. {'a':{'b': 2}, 'c': 3} becomes {'a.b':2, 'c':3}.
    From https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Parameters
    ----------
    dictionary: dict to be flattened
    parent_key: string to prepend the key with
    sep: level separator

    Returns
    -------
    flattened dictionary.
    """
    import collections

    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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


def chunker(seq, size):
    """
    Chunk a list into chunks of size `size`.
    From
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

    Parameters
    ----------
    seq: input list
    size: size of chunks

    Returns
    -------
    The list of lists of size `size`
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def collect_exp_stats(run):
    """
    Collect information such as CPU user time, maximum memory usage,
    and maximum GPU memory usage and save it in the MongoDB.

    Parameters
    ----------
    run: Sacred run
        Current Sacred run.

    Returns
    -------
    None
    """
    exp_id = run.config['overwrite']
    if exp_id is None or run.unobserved:
        return

    stats = {}

    stats['self'] = {}
    stats['self']['user_time'] = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    stats['self']['system_time'] = resource.getrusage(resource.RUSAGE_SELF).ru_stime
    stats['self']['max_memory_bytes'] = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats['children'] = {}
    stats['children']['user_time'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    stats['children']['system_time'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    stats['children']['max_memory_bytes'] = 1024 * resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    if 'torch' in sys.modules:
        import torch
        stats['pytorch'] = {}
        if torch.cuda.is_available():
            stats['pytorch']['gpu_max_memory_bytes'] = torch.cuda.max_memory_allocated()

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        stats['tensorflow'] = {}
        if int(tf.__version__.split('.')[0]) < 2:
            if tf.test.is_gpu_available():
                stats['tensorflow']['gpu_max_memory_bytes'] = tf.contrib.memory_stats.MaxBytesInUse()
        else:
            if len(tf.config.experimental.list_physical_devices('GPU')) >= 1:
                logging.info("SEML stats: There is currently no way to get actual GPU memory usage in TensorFlow 2.")

    collection = db_utils.get_collection(run.config['db_collection'])
    collection.update_one(
            {'_id': exp_id},
            {'$set': {'stats': stats}})


def is_local_source(filename, root_dir):
    """
    See https://github.com/IDSIA/sacred/blob/master/sacred/dependencies.py
    Parameters
    ----------
    filename
    root_dir

    Returns
    -------

    """
    filename = Path(os.path.abspath(os.path.realpath(filename)))
    root_path = Path(os.path.abspath(os.path.realpath(root_dir)))
    if root_path not in filename.parents:
        return False
    return True


def import_exe(executable, conda_env):
    """Import the given executable file.

    Parameters
    ----------
    executable: str
        The Python file containing the experiment.
    conda_env: str
        The experiment's Anaconda environment.

    Returns
    -------
    The module of the imported executable.

    """
    # Check if current environment matches experiment environment
    if conda_env is not None and conda_env != os.environ['CONDA_DEFAULT_ENV']:
        logging.warning(f"Current Anaconda environment does not match the experiment's environment ('{conda_env}').")

    # Get experiment as module (which causes Sacred not to start ex.automain)
    exe_path = os.path.abspath(executable)
    sys.path.insert(0, os.path.dirname(exe_path))
    exe_module = importlib.import_module(os.path.splitext(os.path.basename(executable))[0])
    del sys.path[0]

    return exe_module


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    From https://stackoverflow.com/a/35804945
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


addLoggingLevel('VERBOSE', 19)


class LoggingFormatter(logging.Formatter):
    FORMATS = {
        logging.INFO: "%(msg)s",
        logging.VERBOSE: "%(msg)s",
        logging.DEBUG: "DEBUG: %(module)s: %(lineno)d: %(msg)s",
        "DEFAULT": "%(levelname)s: %(msg)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
