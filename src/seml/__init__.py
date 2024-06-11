import os
from typing import TYPE_CHECKING

from seml.evaluation import *  # noqa


if TYPE_CHECKING:
    from functools import wraps

    from seml.experiment.experiment import Experiment as SemlExperiment

    @wraps(SemlExperiment)
    def Experiment(*args, **kwargs) -> SemlExperiment:
        return SemlExperiment(*args, **kwargs)
else:

    def Experiment(*args, **kwargs):
        from seml.experiment.experiment import Experiment as SemlExperiment

        return SemlExperiment(*args, **kwargs)


if not bool(os.environ.get('_SEML_COMPLETE', False)):
    import importlib.metadata

    __version__ = importlib.metadata.version(__package__ or __name__)
else:
    __version__ = '0.0.0'
