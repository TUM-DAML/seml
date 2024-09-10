from __future__ import annotations

import copy
import functools
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    overload,
)


def s_if(n: int) -> str:
    return '' if n == 1 else 's'


def unflatten(
    dictionary: dict,
    sep: str = '.',
    recursive: bool = False,
    levels: int | Sequence[int] | None = None,
):
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
    import collections.abc

    duplicate_key_warning_str = (
        'Duplicate key detected in recursive dictionary unflattening. '
        "Overwriting previous entries of '{}'."
    )

    if levels is not None:
        if isinstance(levels, collections.abc.Sequence):
            levels = list(levels)
        else:
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


def flatten(dictionary: Mapping[str, Any], parent_key: str = '', sep: str = '.'):
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
    import collections.abc

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


def get_from_nested(d: Mapping[str, Any], key: str, sep: str = '.') -> Any:
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


def list_is_prefix(first: Sequence, second: Sequence) -> bool:
    return len(first) <= len(second) and all(x1 == x2 for x1, x2 in zip(first, second))


def resolve_projection_path_conflicts(
    projection: dict[str, bool | int], sep: str = '.'
) -> dict[str, bool]:
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
    result: dict[tuple[str, ...], bool] = {}
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
            result[k] = bool(v)
    return {sep.join(k): v for k, v in result.items()}


S = TypeVar('S', bound=Sequence)


def chunker(seq: S, size: int) -> Generator[S]:
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
    yield from (cast(S, seq[pos : pos + size]) for pos in range(0, len(seq), size))


D = TypeVar('D', bound=Mapping)


@overload
def merge_dicts(dict1: D, dict2: D) -> D: ...


# NG: I don't have a good idea how to type this properly.
# The idea is that if the two types are identical, we
# return the same type (for TypedDicts). Otherwise, we
# want to return just a dict without any assumptions.
@overload
def merge_dicts(dict1: dict, dict2: dict) -> dict: ...  # type: ignore


def merge_dicts(dict1: Mapping, dict2: Mapping) -> Mapping:
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


def remove_keys_from_nested(d: dict, keys: Iterable[str] = ()) -> dict:
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


def make_hash(d: dict, exclude_keys: Sequence[str] = ()):
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


class Hashabledict(dict):
    def __hash__(self):  # type: ignore - I don't think we can satisfy this. This is indeed a hack.
        import json

        return hash(json.dumps(self, sort_keys=True))


@contextmanager
def working_directory(path: Path | str):
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


def to_slices(items: list[int]) -> list[tuple[int, int]]:
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


def slice_to_str(s: tuple[int, int]) -> str:
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

        return cast(T, wrapper)

    return decorator


def load_text_resource(path: str | Path):
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
    path = Path(path)
    try:
        import importlib.resources

        full_path = importlib.resources.files('seml') / path  # type: ignore
    except (AttributeError, ImportError):
        # Python 3.8
        import importlib_resources

        full_path = importlib_resources.files('seml') / path

    with open(str(full_path)) as inp:
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


def find_jupyter_host(
    log_file: str | Path, wait: bool
) -> tuple[str | None, bool | None]:
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
    import subprocess

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
                with open(log_file) as f:
                    log_file_contents = f.read()
            time.sleep(0.5)
    else:
        if not os.path.exists(log_file):
            return None, None
        with open(log_file) as f:
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


@functools.cache
def get_virtual_env_path():
    """
    Get the path to the virtual environment.

    Returns
    -------
    str
        The path to the virtual environment.
    """
    if path := os.environ.get('VIRTUAL_ENV', os.environ.get('CONDA_PREFIX')):
        return Path(path).expanduser().resolve()
    return None


def is_local_file(
    filename: str | Path,
    root_dir: str | Path,
    ignore_site_packages_folder: bool = True,
):
    """
    See https://github.com/IDSIA/sacred/blob/master/sacred/dependencies.py
    Parameters
    ----------
    filename
    root_dir

    Returns
    -------
    bool
    """
    import site

    file_path = Path(filename).expanduser().resolve()
    root_path = Path(root_dir).expanduser().resolve()
    # We do the simple check first to avoid the expensive loop check
    # Check if file lies within the root directory
    if not file_path.is_relative_to(root_path):
        return False
    # Reject all files that are in some-diretory called `site-packages`
    if not ignore_site_packages_folder and 'site-packages' in str(file_path):
        return False
    if (venv := get_virtual_env_path()) and file_path.is_relative_to(venv):
        return False
    # Check if the file is in any environment site-packages
    for site_dir in map(Path, site.getsitepackages()):
        if file_path.is_relative_to(site_dir):
            return False
    # We are in the root_dir and not in any site-packages
    return True


def smaller_than_version_filter(version: tuple[int, int, int]):
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
        return datetime.datetime.now(datetime.UTC)  # type: ignore - here the type checker may fail in old python version
    except AttributeError:
        return datetime.datetime.utcnow()


TD = TypeVar('TD', bound=Mapping[str, Any])


def to_typeddict(d: Mapping[str, Any], cls: type[TD], missing_ok: bool = True) -> TD:
    """
    Returns a new TypedDict where only keys that are in the class type are kept.

    If the class has an `__extra_items__` attribute, the input dict is returned as is.

    If one wants to explicitly drop the delta between two typed dicts, use `cast_and_drop`.

    Parameters
    ----------
    d: Mapping[str, Any]
        The object to cast.
    cls: type[TD]
        The target class.
    missing_ok: bool, default: True
        Whether to allow missing keys in `d`.

    Returns
    -------
    TD
        The new object.
    """
    from copy import deepcopy

    if getattr(cls, '__extra_items__', None):
        return cast(TD, deepcopy(d))

    result = dict()
    for key in cls.__annotations__:
        if key in d:
            result[key] = d[key]
        elif not missing_ok:
            raise ValueError(f'Missing key {key} in {d}.')
    return cast(TD, result)


TD1 = TypeVar('TD1', bound=Mapping[str, Any])
TD2 = TypeVar('TD2', bound=Mapping[str, Any])


def drop_typeddict_difference(obj: TD1, cls: type[TD1], cls2: type[TD2]):
    """
    Returns a new TypedDict where all keys that cls has but cls2 does not have are dropped.

    Parameters
    ----------
    obj: TD1
        The object to cast.
    cls: type[TD1]
        The current class.
    cls2: type[TD2]
        The target class.

    Returns
    -------
    TD2
        The new object.
    """
    from copy import deepcopy

    result = dict(deepcopy(obj))
    to_drop = [key for key in cls.__annotations__ if key not in cls2.__annotations__]
    for k in to_drop:
        if k in result:
            del result[k]
    return cast(cls2, result)
