from seml.evaluation import *  # noqa
from seml.experiment import Experiment  # noqa
from seml.experiment.observers import *  # noqa
from seml.utils.module_hider import AUTOCOMPLETING


if AUTOCOMPLETING:
    __version__ = '0.0.0'
else:
    import importlib.metadata

    __version__ = importlib.metadata.version(__package__ or __name__)
