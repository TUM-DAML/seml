import os

from seml.module_hider import CompletionModuleHider

# If we are in a completion shell, we don't want to import rich.
# due to its high import time.
with CompletionModuleHider('rich', bool(os.environ.get('_SEML_COMPLETE'))):
    from typer import *
