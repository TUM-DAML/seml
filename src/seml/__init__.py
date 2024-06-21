import os

from seml.evaluation import *  # noqa
from seml.experiment import Experiment  # noqa
from seml.experiment.observers import *  # noqa


if not bool(os.environ.get('_SEML_COMPLETE', False)):
    import importlib.metadata

    __version__ = importlib.metadata.version(__package__ or __name__)
else:
    __version__ = '0.0.0'
