import sys
from importlib.abc import MetaPathFinder


class ModuleHider(MetaPathFinder):
    def __init__(self, *hidden_modules: str, hide: bool = True) -> None:
        super().__init__()
        self.hidden_modules = set(hidden_modules)
        self.hide = hide

    def find_spec(self, fullname, path, target=None):
        if fullname in self.hidden_modules:
            raise ImportError('No module named {}'.format(fullname))

    def __enter__(self):
        if self.hide:
            sys.meta_path.insert(0, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide:
            sys.meta_path.remove(self)
