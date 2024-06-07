import os
from typing import TYPE_CHECKING

from seml.evaluation import *  # noqa
from seml.experiment.observers import *  # noqa


def setup_logger(ex, level='INFO'):
    import logging

    from seml.experiment.experiment import setup_logger

    logging.warn(
        'Importing setup_logger directly from seml is deprecated.\n'
        'Use from seml.experiment import setup_logger instead.\n'
        'Note that seml.experiment.Experiment already includes the logger setup.\n'
        'See https://github.com/TUM-DAML/seml/blob/master/examples/example_experiment.py'
    )
    setup_logger(ex, level=level)


def collect_exp_stats(run):
    from seml.experiment.experiment import collect_exp_stats

    collect_exp_stats(run)


if TYPE_CHECKING:
    from functools import wraps

    from seml.experiment.experiment import Experiment as SemlExperiment

    @wraps(SemlExperiment)
    def Experiment(*args, **kwargs) -> SemlExperiment:
        return SemlExperiment(*args, **kwargs)
else:

    def Experiment(*args, **kwargs) -> 'SemlExperiment':
        from seml.experiment.experiment import Experiment as SemlExperiment

        return SemlExperiment(*args, **kwargs)


if not bool(os.environ.get('_SEML_COMPLETE', False)):
    import importlib.metadata

    __version__ = importlib.metadata.version(__package__ or __name__)
else:
    __version__ = '0.0.0'
