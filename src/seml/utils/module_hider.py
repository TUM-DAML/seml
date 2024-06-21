import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader

AUTOCOMPLETING = bool(os.environ.get('_SEML_COMPLETE', False))


class PackageNotFoundError(Exception): ...


class FakeImportlibMetadata(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.PackageNotFoundError = PackageNotFoundError

        def version(package):
            return '0.0.0'

        module.version = version
        module.metadata = lambda name: {'version': '0.0.0'}


class ModuleHider(MetaPathFinder):
    def __init__(self, *hidden_modules: str, hide: bool = AUTOCOMPLETING) -> None:
        super().__init__()
        self.hidden_modules = set(hidden_modules)
        self.hide = hide

    def find_spec(self, fullname, path, target=None):
        if fullname in self.hidden_modules:
            # a special case for munch <3
            if fullname == 'importlib_metadata':
                return spec_from_loader(fullname, FakeImportlibMetadata())
            raise ImportError('No module named {}'.format(fullname))

    def __enter__(self):
        if self.hide:
            sys.meta_path.insert(0, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide:
            sys.meta_path.remove(self)
