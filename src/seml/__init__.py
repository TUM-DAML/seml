import importlib.metadata
from seml.observers import *  # noqa
from seml.evaluation import *  # noqa


def setup_logger(ex, level='INFO'):
    import logging
    from seml.experiment import setup_logger

    logging.warn(
        'seml.setup_logger is deprecated.\n'
        'Use seml.experiment.Experiment instead of sacred.Experiment.\n'
        'seml.experiment.Experiment already includes the logger setup.\n'
        'See https://github.com/TUM-DAML/seml/blob/master/examples/example_experiment.py'
    )
    setup_logger(ex, level=level)


def collect_exp_stats(run):
    import logging
    from seml.experiment import collect_exp_stats

    logging.warn(
        'seml.collect_exp_stats is deprecated.\n'
        'Use seml.experiment.Experiment instead of sacred.Experiment.\n'
        'seml.experiment.Experiment already includes the statistics collection.\n'
        'See https://github.com/TUM-DAML/seml/blob/master/examples/example_experiment.py'
    )
    collect_exp_stats(run)


__version__ = importlib.metadata.version(__package__ or __name__)
