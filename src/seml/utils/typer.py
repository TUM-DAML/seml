from seml.utils.module_hider import ModuleHider

# If we are in a completion shell, we don't want to import rich.
# due to its high import time.
with ModuleHider('rich'):
    from typer import *  # noqa
