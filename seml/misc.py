import sys
import resource
import subprocess
import logging
from seml.settings import SETTINGS
import seml.database_utils as db_utils


def get_config_from_exp(exp, log_verbose=False, unobserved=False, post_mortem=False, debug=False):
    if 'executable' not in exp['seml']:
        raise ValueError(f"No executable found for experiment {exp['_id']}. Aborting.")
    exe = exp['seml']['executable']

    config = exp['config']
    config['db_collection'] = exp['seml']['db_collection']
    if not unobserved:
        config['overwrite'] = exp['_id']
    config_strings = [f'{key}="{val}"' for key, val in config.items()]
    if not log_verbose:
        config_strings.append("--force")
    if unobserved:
        config_strings.append("--unobserved")
    if post_mortem:
        config_strings.append("--pdb")
    if debug:
        config_strings.append("--debug")

    return exe, config_strings


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
        print('Failed to create Slack observer.')
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
        print('No API token for Nepune provided. Trying to use environment variable NEPTUNE_API_TOKEN.')
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


def collect_exp_stats(exp):
    """
    Collect information such as CPU user time, maximum memory usage,
    and maximum GPU memory usage and save it in the MongoDB.

    Parameters
    ----------
    exp: Sacred Experiment
        Currently running Sacred experiment.

    Returns
    -------
    None
    """
    if exp.current_run is None:
        return
    exp_id = exp.current_run.config['overwrite']
    if exp_id is None or exp.current_run.unobserved:
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

    collection = db_utils.get_collection(exp.current_run.config['db_collection'])
    collection.update_one(
            {'_id': exp_id},
            {'$set': {'stats': stats}})
