import os

from seml.module_hider import ModuleHider

# If we are in a completion shell, we don't want to import rich.
# due to its high import time.
with ModuleHider("rich", hide=bool(os.environ.get("_SEML_COMPLETE"))):
    from typer import *  # noqa
