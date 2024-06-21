import copy
import functools
import hashlib
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


def s_if(n: int) -> str:
    return '' if n == 1 else 's'


def unflatten(dictionary: dict, sep: str = '.', recursive: bool = False, levels=None):
    """
    Turns a flattened dict into a nested one, e.g. {'a.b':2, 'c':3} becomes {'a':{'b': 2}, 'c': 3}
    From https://stackoverflow.com/questions/6037503/python-unflatten-dict.

    Parameters
    ----------
    dictionary: dict to be un-flattened
    sep: separator with which the nested keys are separated
    recursive: bool, default: False
        Whether to also un-flatten sub-dictionaries recursively. NOTE: if recursive is True, there can be key
        collisions, e.g.: {'a.b': 3, 'a': {'b': 5}}. In these cases, keys which are later in the insertion order
        overwrite former ones, i.e. the example above returns {'a': {'b': 5}}.
    levels: int or list of ints (optional).
        If specified, only un-flatten the desired levels. E.g., if levels= [0, -1], then {'a.b.c.d': 111} becomes
        {'a': {'b.c': {'d': 111}}}.

    Returns
    -------
    result_dict: the nested dictionary.
    """

    duplicate_key_warning_str = (
        'Duplicate key detected in recursive dictionary unflattening. '
        "Overwriting previous entries of '{}'."
    )

    if levels is not None:
        if not isinstance(levels, tuple) and not isinstance(levels, list):
            levels = [levels]
        if len(levels) == 0:
            raise ValueError(
                'Need at least one level to unflatten when levels != None.'
            )
        if not isinstance(levels[0], int):
            raise TypeError(
                f'Levels must be list or set of integers, got type {type(levels[0])}.'
            )

    result_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict) and recursive:
            value = unflatten(value, sep=sep, recursive=True, levels=levels)

        parts = key.split(sep)
        if levels is not None:
            key_levels = levels.copy()
            for ix in range(len(key_levels)):
                if key_levels[ix] < 0:
                    new_ix = len(parts) + key_levels[ix] - 1
                    if (
                        key_levels[ix] == -1
                    ):  # special case so that indexing with -1 never throws an error.
                        new_ix = max(0, new_ix)
                    if new_ix < 0:
                        raise IndexError(
                            f'Dictionary key level out of bounds. ({new_ix} < 0).'
                        )
                    key_levels[ix] = new_ix
                if key_levels[ix] >= len(parts):
                    raise IndexError(
                        f'Dictionary key level {key_levels[ix]} out of bounds for size {len(parts)}.'
                    )
            key_levels = sorted(key_levels)

            key_levels = list(set(key_levels))
            new_parts = []
            ix_current = 0
            for level in key_levels:
                new_parts.append(sep.join(parts[ix_current : level + 1]))
                ix_current = level + 1

            if ix_current < len(parts):
                new_parts.append(sep.join(parts[ix_current::]))
            parts = new_parts

        d = result_dict
        # Index the existing dictionary in a nested way via the separated key levels. Create empty dicts if necessary.
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            elif not isinstance(d[part], dict):
                # Here we have a case such as: {'a.b': ['not_dict'], 'a': {'b': {'c': 111}}}
                # Since later keys overwrite former ones, we replace the value for {'a.b'} with {'c': 111}.
                logging.warning(duplicate_key_warning_str.format(part))
                d[part] = dict()
            # Select the sub-dictionary for the key level.
            d = d[part]
        last_key = parts[-1]
        if last_key in d:
            if isinstance(value, dict):
                intersection = set(d[last_key].keys()).intersection(value.keys())
                if len(intersection) > 0:
                    logging.warning(duplicate_key_warning_str.format(last_key))
                # Merge dictionaries, overwriting any existing values for duplicate keys.
                d[last_key] = merge_dicts(d[last_key], value)
            else:
                logging.warning(duplicate_key_warning_str.format(last_key))
                d[last_key] = value
        else:
            d[last_key] = value
    return result_dict


def flatten(dictionary: dict, parent_key: str = '', sep: str = '.'):
    """
    Flatten a nested dictionary, e.g. {'a':{'b': 2}, 'c': 3} becomes {'a.b':2, 'c':3}.
    From https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Parameters
    ----------
    dictionary: dict to be flattened
    parent_key: string to prepend the key with
    sep: level separator

    Returns
    -------
    flattened dictionary.
    """
    import collections

    items = []
    for k, v in dictionary.items():
        k = str(k)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            # This covers the edge case that someone supplies an empty dictionary as parameter
            if len(v) == 0:
                items.append((new_key, v))
            else:
                items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_from_nested(d: Dict, key: str, sep: str = '.') -> Any:
    """Gets a value from an unflattened dict, e.g. allows to use strings like `config.data` on a nesteddict

    Parameters
    ----------
    d : Dict
        The dict from which to get
    key : str
        A path to the value, separated by `sep`
    sep : str, optional
        The separator for levels in the nested dict, by default '.'

    Returns
    -------
    Any
        The nested value
    """
    for k in key.split(sep):
        d = d[k]
    return d


def list_is_prefix(first: List, second: List) -> bool:
    return len(first) <= len(second) and all(x1 == x2 for x1, x2 in zip(first, second))


def resolve_projection_path_conflicts(
    projection: Dict[str, Union[bool, int]], sep: str = '.'
) -> Dict[str, bool]:
    """Removes path conflicts in a MongoDB projection dict. E.g. if you pass the dict
    `{'config' : 1, 'config.dataset' : 1}`, MongoDB will throw an error. This method will ensure that
    always the "bigger" projection is returned, i.e. `"config"` in the aforementioned example.
    Note that this resolution will not work if you pass e.g. `{'config' : 1, 'config.dataset' : 0}`.

    Parameters
    ----------
    projection : Dict[str, bool]
        The projection to resolve
    sep : str, optional
        The separator for nested config values, by default '.'

    Returns
    -------
    Dict[str, bool]
        The resolved projection
    """
    result = {}
    for k, v in projection.items():
        k = tuple(k.split(sep))
        add_k = True
        for other in list(result.keys()):
            if list_is_prefix(k, other):
                # If `k` is a prefix of any path in `result`, this path will be removed
                if result[other] != v:
                    raise ValueError(
                        f'Can not resolve projection {(k, v), (other, result[other])}'
                    )
                del result[other]
            elif list_is_prefix(other, k):
                # If any other path in `result` is a prefix of `k` we do not add k
                if result[other] != v:
                    raise ValueError(
                        f'Can not resolve projection {(k, v), (other, result[other])}'
                    )
                add_k = False
        if add_k:
            result[k] = v
    return {sep.join(k): v for k, v in result.items()}


def chunker(seq, size):
    """
    Chunk a list into chunks of size `size`.
    From
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

    Parameters
    ----------
    seq: input list
    size: size of chunks

    Returns
    -------
    The list of lists of size `size`
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries.

    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.

    """
    if not isinstance(dict1, dict):
        raise ValueError(f'Expecting dict1 to be dict, found {type(dict1)}.')
    if not isinstance(dict2, dict):
        raise ValueError(f'Expecting dict2 to be dict, found {type(dict2)}.')

    return_dict = copy.deepcopy(dict1)

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            else:
                return_dict[k] = dict2[k]

    return return_dict


def remove_keys_from_nested(d: Dict, keys: List[str] = []) -> Dict:
    """Removes keys from a nested dictionary

    Parameters
    ----------
    d : Dict
        the dict to remove keys from
    keys : List[str], optional
        the keys to remove, by default []. Prefixes are also allowed.

    Returns
    -------
    Dict
        a copy of the dict without the values in `keys`
    """
    return unflatten(
        {
            k: v
            for k, v in flatten(d).items()
            if not any(k.startswith(key) for key in keys)
        }
    )


def make_hash(
    d: Dict,
    exclude_keys: List[str] = [
        'seed',
    ],
):
    """
    Generate a hash for the input dictionary.
    From: https://stackoverflow.com/a/22003440
    Parameters
    ----------
    d : Dict
        The dictionary to hash
    exclude_keys : List[str]
        Keys to not hash.

    Returns
    -------
    hash (hex encoded) of the input dictionary.
    """
    import hashlib
    import json

    return hashlib.md5(
        json.dumps(remove_keys_from_nested(d, exclude_keys), sort_keys=True).encode(
            'utf-8'
        )
    ).hexdigest()


def add_logging_level(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    From https://stackoverflow.com/a/35804945
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError(f'{levelName} already defined in logging module.')
    if hasattr(logging, methodName):
        raise AttributeError(f'{methodName} already defined in logging module.')
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError(f'{methodName} already defined in logger class.')

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


add_logging_level('VERBOSE', 19)


class LoggingFormatter(logging.Formatter):
    FORMATS = {
        logging.INFO: '%(msg)s',
        logging.VERBOSE: '%(msg)s',
        logging.DEBUG: 'DEBUG: %(module)s: %(lineno)d: %(msg)s',
        'DEFAULT': '%(levelname)s: %(msg)s',
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Hashabledict(dict):
    def __hash__(self):
        import json

        return hash(json.dumps(self, sort_keys=True))


@contextmanager
def working_directory(path: Path):
    """
    Context manager to temporarily change the working directory.

    Parameters
    ----------
    path: Path
        Path to the new working directory.
    """
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


R = TypeVar('R')


class DiskCachedFunction(Generic[R]):
    def __init__(
        self,
        fun: Callable[[], R],
        name: str,
        time_to_live: float,
    ):
        from seml.settings import SETTINGS

        user = os.environ.get('USER', 'unknown')
        install_hash = hashlib.md5(name.encode('utf-8')).hexdigest()
        file_name = f'seml_{user}_{name}_{install_hash}.json'
        self.cache_path = Path(SETTINGS.TMP_DIRECTORY) / file_name
        self.time_to_live = time_to_live
        self.fun = fun

    def __call__(self) -> R:
        import json
        import time

        # Load from cache
        # if it fails or is expired we will compute it again
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    cache = json.load(f)
                if cache['expire'] > time.time():
                    return cache['result']
            except IOError:
                pass
            except json.JSONDecodeError:
                pass
        # Compute and save to cache
        result = self.fun()
        cache = {'result': result, 'expire': time.time() + self.time_to_live}
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(cache, f)
        except IOError:
            # If the writing fails for any reason we can just continue.
            pass
        return result

    def clear_cache(self):
        if self.cache_path.exists():
            try:
                os.remove(self.cache_path)
                return True
            except IOError:
                return False
        return False

    def recompute_cache(self):
        if self.clear_cache():
            self()
            return True
        return False


def cache_to_disk(name: str, time_to_live: float):
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


def to_slices(items: List[int]) -> List[Tuple[int, int]]:
    """
    Convert a list of integers to a list of slices.

    Parameters
    ----------
    items: List[int]
        List of integers.

    Returns
    -------
    List[Tuple[int, int]]
        List of slices.
    """
    slices = []
    if len(items) == 0:
        return slices
    items = sorted(items)
    start, end = items[0], items[0]
    for i in items[1:]:
        if i == end + 1:
            end = i
        else:
            slices.append((start, end))
            start, end = i, i
    # last slice
    slices.append((start, end))
    return slices


def slice_to_str(s: Tuple[int, int]) -> str:
    """
    Convert a slice to a string.

    Parameters
    ----------
    s: Tuple[int, int]
        The slice.

    Returns
    -------
    str
        The slice as a string.
    """
    if s[0] == s[1]:
        return str(s[0])
    else:
        return f'{s[0]}-{s[1]}'


def to_hashable(x: Any) -> Any:
    """Returns a hashable representation of an object. Currently supports dicts and other iterables (which will be
    transformed into tuples)

    Parameters
    ----------
    x : Any
        the object to transform

    Returns
    -------
    Any
        the hashable representation
    """
    if isinstance(x, Hashable):
        return x
    elif isinstance(x, Dict):
        Hashabledict((k, to_hashable(v)) for k, v in x.items())
    elif isinstance(x, Iterable):
        return tuple(map(to_hashable, x))
    else:
        raise ValueError(f'{x} of type {type(x)} is not hashable.')


T = TypeVar('T', bound=Callable)


def warn_multiple_calls(warning: str, warn_after: int = 1):
    """
    Decorator to warn if a function is called multiple times.

    Parameters
    ----------
    warning: str
        The warning message.
    warn_after: int
        The number of calls after which to warn.
    """

    def decorator(f: T) -> T:
        num_calls = 0

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal num_calls
            num_calls += 1
            if num_calls > warn_after:
                logging.warning(warning.format(num_calls=num_calls))
            return f(*args, **kwargs)

        return wrapper

    return decorator


def load_text_resource(path: Union[str, Path]):
    """
    Read a text resource from the package.

    Parameters
    ----------
    path: str | Path
        Path to the resource.

    Returns
    -------
    str
        The resource content.
    """
    try:
        import importlib.resources

        full_path = importlib.resources.files('seml') / path
    except (AttributeError, ImportError):
        # Python 3.8
        import importlib_resources

        full_path = importlib_resources.files('seml') / path

    with open(full_path) as inp:
        return inp.read()


def assert_package_installed(package: str, error: str):
    """
    Assert that a package is installed.

    Parameters
    ----------
    package: str
        The package name.
    """
    import importlib

    try:
        importlib.import_module(package)
    except ImportError:
        logging.error(error)
        exit(1)


def src_layout_to_flat_layout(original_path: Union[Path, str]):
    """
    Removes the first "src" directory from the path, handling any position.
    For the imports to prefer our loaded seml version, we need to convert the src-layout to the flat-layout.
    https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

    Parameters:
    ----------
    original_path: Path | str
        The original path.

    Returns:
        A new path object with "src" removed.
    """
    path = Path(original_path)
    if path.parts[0] == 'src':
        return Path(*path.parts[1:])
    return path


def find_jupyter_host(
    log_file: Union[str, Path], wait: bool
) -> Tuple[Optional[str], Optional[bool]]:
    """
    Extracts the hostname from the jupyter log file and returns the URL.

    Parameters
    ----------
    log_file: str | Path
        The path to the log file.
    wait: bool
        Whether to wait until the jupyter server is running.

    Returns
    -------
    Optional[str]
        The URL of the jupyter server.
    Optional[bool]
        Whether the hostname is known. If None is returned, an error occured.
    """
    hosts_str = subprocess.run(
        'sinfo -h -o "%N|%o"', shell=True, check=True, capture_output=True
    ).stdout.decode('utf-8')
    hosts = {
        h.split('|')[0]: h.split('|')[1] for h in hosts_str.split('\n') if len(h) > 1
    }
    # Wait until jupyter is running
    if wait:
        log_file_contents = ''
        while ' is running at' not in log_file_contents:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_file_contents = f.read()
            time.sleep(0.5)
    else:
        if not os.path.exists(log_file):
            return None, None
        with open(log_file, 'r') as f:
            log_file_contents = f.read()
        if ' is running at' not in log_file_contents:
            return None, None
    # Determine hostname
    JUPYTER_LOG_HOSTNAME_PREFIX = 'SLURM assigned me the node(s): '
    hostname = (
        [x for x in log_file_contents.split('\n') if JUPYTER_LOG_HOSTNAME_PREFIX in x][
            0
        ]
        .split(':')[1]
        .strip()
    )
    if hostname in hosts:
        hostname = hosts[hostname]
        known_host = True
    else:
        known_host = False
    # Obtain general URL
    log_file_split = log_file_contents.split('\n')
    url_lines = [x for x in log_file_split if 'http' in x]
    url = url_lines[0].split(' ')
    url_str = None
    for s in url:
        if s.startswith('http://') or s.startswith('https://'):
            url_str = s
            break
    if url_str is None:
        return log_file_contents, None
    url_str = hostname + ':' + url_str.split(':')[-1]
    url_str = url_str.rstrip('/')
    if url_str.endswith('/lab'):
        url_str = url_str[:-4]
    return url_str, known_host


def is_local_file(filename, root_dir):
    """
    See https://github.com/IDSIA/sacred/blob/master/sacred/dependencies.py
    Parameters
    ----------
    filename
    root_dir

    Returns
    -------

    """
    file_path = Path(filename).expanduser().resolve()
    root_path = Path(root_dir).expanduser().resolve()
    return root_path in file_path.parents


def smaller_than_version_filter(version: Tuple[int, int, int]):
    """
    Returns a mongodb filter that selects experiments where the version number
    is small or equal to the supplied version.

    Parameters
    ----------
    version: Tuple[int, int, int]
        The version number to compare to.

    Returns
    -------
    Dict
        The filter.
    """
    from seml.settings import SETTINGS

    version_prefix = f'seml.{SETTINGS.SEML_CONFIG_VALUE_VERSION}'
    return {
        '$or': [
            {f'{version_prefix}.0': {'$lt': version[0]}},
            {
                f'{version_prefix}.0': {'$eq': version[0]},
                f'{version_prefix}.1': {'$lt': version[1]},
            },
            {
                f'{version_prefix}.0': {'$eq': version[0]},
                f'{version_prefix}.1': {'$eq': version[1]},
                f'{version_prefix}.2': {'$lt': version[2]},
            },
        ]
    }


def utcnow():
    """
    Wrapper around datetime.datetime.now(datetime.UTC) but supports older python versions.

    Returns
    -------
    datetime.datetime
        The current datetime.
    """
    import datetime

    try:
        return datetime.datetime.now(datetime.UTC)
    except AttributeError:
        return datetime.datetime.utcnow()
