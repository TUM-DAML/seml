import subprocess
import logging


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
    try:
        squeue_out = subprocess.check_output("squeue -a -t pending,running -h -o %A", shell=True)
        return [int(line) for line in squeue_out.split(b'\n')[:-1]]
    except subprocess.CalledProcessError:
        return []


def setup_logger(ex):
    """
    Set up logger for experiment.

    Parameters
    ----------
    ex: sacred.Experiment
    Sacred experiment to set the logger of.

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
    logger.setLevel('INFO')
    ex.logger = logger


def s_if(n):
    return '' if n == 1 else 's'


def get_default_slurm_config():
    return {
            'output_dir': '.',
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