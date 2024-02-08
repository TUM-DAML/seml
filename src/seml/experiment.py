import datetime
import logging
import resource
import sys
from typing import Optional, Sequence, List

from sacred import Ingredient
from sacred.utils import PathType
from sacred.host_info import HostInfoGetter
from sacred.commandline_options import CLIOption
from sacred import Experiment as ExperimentBase
from sacred import SETTINGS as SACRED_SETTINGS

from seml.database import get_collection
from seml.observers import create_mongodb_observer
from seml.settings import SETTINGS

__all__ = ['setup_logger', 'collect_exp_stats']


class Experiment(ExperimentBase):
    def __init__(
        self,
        name: Optional[str] = None,
        ingredients: Sequence[Ingredient] = (),
        interactive: bool = False,
        base_dir: Optional[PathType] = None,
        additional_host_info: Optional[List[HostInfoGetter]] = None,
        additional_cli_options: Optional[Sequence[CLIOption]] = None,
        save_git_info: bool = True,
        add_mongodb_observer: bool = True,
    ):
        super().__init__(
            name=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            additional_host_info=additional_host_info,
            additional_cli_options=additional_cli_options,
            save_git_info=save_git_info,
        )
        if add_mongodb_observer:
            self.setup_mongodb_observer()

    def run(
        self,
        command_name: Optional[str] = None,
        config_updates: Optional[dict] = None,
        named_configs: Sequence[str] = (),
        info: Optional[dict] = None,
        meta_info: Optional[dict] = None,
        options: Optional[dict] = None,
    ):
        if not SETTINGS.EXPERIMENT.CAPTURE_OUTPUT:
            SACRED_SETTINGS.CAPTURE_MODE = 'no'
        super().run(
            command_name=command_name,
            config_updates=config_updates,
            named_configs=named_configs,
            info=info,
            meta_info=meta_info,
            options=options,
        )

    def setup_mongodb_observer(self):
        global _experiment
        _experiment = self

        def mongodb_observer_config():
            global _experiment
            overwrite = None
            db_collection = None
            if db_collection is not None:
                _experiment.observers.append(
                    create_mongodb_observer(db_collection, overwrite=overwrite)
                )
            del _experiment

        self.config(mongodb_observer_config)


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
        fmt='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)
    ex.logger = logger


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

    stats['real_time'] = (datetime.datetime.utcnow() - run.start_time).total_seconds()

    stats['self'] = {}
    stats['self']['user_time'] = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    stats['self']['system_time'] = resource.getrusage(resource.RUSAGE_SELF).ru_stime
    stats['self']['max_memory_bytes'] = (
        1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    stats['children'] = {}
    stats['children']['user_time'] = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_utime
    stats['children']['system_time'] = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_stime
    stats['children']['max_memory_bytes'] = (
        1024 * resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )

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
                with tf.Session() as sess:
                    stats['tensorflow']['gpu_max_memory_bytes'] = int(
                        sess.run(tf.contrib.memory_stats.MaxBytesInUse())
                    )
        else:
            if len(tf.config.experimental.list_physical_devices('GPU')) >= 1:
                if int(tf.__version__.split('.')[1]) >= 5:
                    stats['tensorflow'][
                        'gpu_max_memory_bytes'
                    ] = tf.config.experimental.get_memory_info('GPU:0')['peak']
                else:
                    logging.info(
                        'SEML stats: There is no way to get actual peak GPU memory usage in TensorFlow 2.0-2.4.'
                    )

    collection = get_collection(run.config['db_collection'])
    collection.update_one({'_id': exp_id}, {'$set': {'stats': stats}})
