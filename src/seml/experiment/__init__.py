from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from functools import wraps

    from .experiment import Experiment as SemlExperiment

    @wraps(SemlExperiment)
    def Experiment(*args, **kwargs) -> SemlExperiment:
        return SemlExperiment(*args, **kwargs)
else:

    def Experiment(*args, **kwargs):
        from .experiment import Experiment as SemlExperiment

        return SemlExperiment(*args, **kwargs)
