import os

from seml.module_hider import ModuleHider

if 'SEML_NO_RICH' in os.environ and os.environ['SEML_NO_RICH']:
    ModuleHider('rich').__enter__()

# If we are in a completion shell, we don't want to import rich.
# due to its high import time.
with ModuleHider('rich', hide=bool(os.environ.get('_SEML_COMPLETE'))):
    from typer import *
