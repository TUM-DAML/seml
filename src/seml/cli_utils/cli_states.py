# When autocompleting, we don't want to read the settings.py to retrieve the actual states.
# Instead, we use a dummy class that behaves like a the actual states class but always returns
# empty lists. This way we can still use the autocompletion without having to load the settings.py.
# When seml returns type hints, we load the actual states from the settings.py.
from seml.cli_utils import AUTOCOMPLETING

if AUTOCOMPLETING:

    class DummyStates:
        def __getitem__(self, item):
            return []

        def __getattr__(self, item):
            return []

        def values(self):
            return []

    CliStates = DummyStates()
else:
    from seml.settings import SETTINGS

    CliStates = SETTINGS.STATES
