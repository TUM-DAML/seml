from typing import Callable, Generic, TypeVar

R = TypeVar('R')


class DiskCachedFunction(Generic[R]):
    def __init__(
        self,
        fun: Callable[[], R],
        name: str,
        time_to_live: float | None,
    ):
        self.time_to_live = time_to_live
        self.fun = fun
        self.name = name

    @property
    def cache_path(self):
        import hashlib
        import os
        from pathlib import Path

        from seml.settings import SETTINGS

        user = os.environ.get('USER', 'unknown')
        install_hash = hashlib.md5(self.name.encode('utf-8')).hexdigest()
        file_name = f'seml_{user}_{self.name}_{install_hash}.json'
        return Path(SETTINGS.TMP_DIRECTORY) / file_name

    def __call__(self) -> R:
        import json
        import time

        from seml.settings import SETTINGS

        # Load from cache
        # if it fails or is expired we will compute it again
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    cache = json.load(f)
                if cache['expire'] > time.time():
                    return cache['result']
            except OSError:
                pass
            except json.JSONDecodeError:
                pass
        # Compute and save to cache
        result = self.fun()
        time_to_live = self.time_to_live or SETTINGS.AUTOCOMPLETE_CACHE_ALIVE_TIME
        cache = {'result': result, 'expire': time.time() + time_to_live}
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(cache, f)
        except OSError:
            # If the writing fails for any reason we can just continue.
            pass
        return result

    def clear_cache(self):
        import os

        if self.cache_path.exists():
            try:
                os.remove(self.cache_path)
                return True
            except OSError:
                return False
        return False

    def recompute_cache(self):
        if self.clear_cache():
            self()
            return True
        return False


def cache_to_disk(name: str, time_to_live: float | None = None):
    """
    Cache the result of a function to disk.

    Parameters
    ----------
    name: str
        Name of the cache file.
    time_to_live: float
        Time to live of the cache in seconds.

    Returns
    -------
    A function decorator.
    """

    def wrapper(fun: Callable[[], R]):
        return DiskCachedFunction(fun, name, time_to_live)

    return wrapper
