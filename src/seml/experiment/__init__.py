from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import Experiment  # noqa
else:

    def Experiment(*args, **kwargs):
        # THis proxy class is used to avoid importing the real thing
        from .experiment import Experiment as SemlExperiment

        return SemlExperiment(*args, **kwargs)
