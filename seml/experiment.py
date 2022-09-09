import sys
import logging
import resource
import datetime

from seml.database import get_collection

__all__ = ['setup_logger', 'collect_exp_stats']


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
                with tf.Session() as sess:
                    stats['tensorflow']['gpu_max_memory_bytes'] = int(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
        else:
            if len(tf.config.experimental.list_physical_devices('GPU')) >= 1:
                if int(tf.__version__.split('.')[1]) >= 5:
                    stats['tensorflow']['gpu_max_memory_bytes'] = tf.config.experimental.get_memory_info('GPU:0')['peak']
                else:
                    logging.info("SEML stats: There is no way to get actual peak GPU memory usage in TensorFlow 2.0-2.4.")

    collection = get_collection(run.config['db_collection'])
    collection.update_one(
            {'_id': exp_id},
            {'$set': {'stats': stats}})
