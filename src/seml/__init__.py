import importlib.metadata
from seml.observers import *  # noqa
from seml.evaluation import *  # noqa


__version__ = importlib.metadata.version(__package__ or __name__)
