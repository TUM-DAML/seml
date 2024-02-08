import importlib.metadata
from seml.experiment import *  # noqa
from seml.observers import *  # noqa
from seml.evaluation import *  # noqa
from seml.experiment import Experiment  # noqa


__version__ = importlib.metadata.version(__package__ or __name__)
