import importlib.metadata
from seml.observers import *  # noqa
from seml.evaluation import *  # noqa


def setup_logger(ex, level='INFO'):
    from seml.experiment import setup_logger

    setup_logger(ex, level=level)


def collect_exp_stats(run):
    from seml.experiment import collect_exp_stats

    collect_exp_stats(run)


__version__ = importlib.metadata.version(__package__ or __name__)
