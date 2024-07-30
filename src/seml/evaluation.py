from __future__ import annotations

import logging
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Sequence,
    cast,
    overload,
)

from seml.database import get_collection
from seml.document import ExperimentDoc
from seml.settings import SETTINGS

if TYPE_CHECKING:
    import pandas as pd

States = SETTINGS.STATES

__all__ = ['get_results']


def parse_jsonpickle(db_entry: ExperimentDoc):
    import json

    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy

    jsonpickle_numpy.register_handlers()
    try:
        p = jsonpickle.pickler.Pickler(keys=False)
        parsed = jsonpickle.loads(json.dumps(db_entry, default=p.flatten), keys=False)
        parsed = cast(ExperimentDoc, parsed)
    except IndexError:
        parsed = db_entry
    return parsed


@overload
def get_results(
    db_collection_name: str,
    fields: Sequence[str] | None = None,
    to_data_frame: Literal[False] = False,
    mongodb_config: dict[str, Any] | None = None,
    states: Sequence[str] | None = None,
    filter_dict: dict[str, Any] | None = None,
    parallel: bool = False,
    progress: bool = True,
) -> list[ExperimentDoc]: ...


@overload
def get_results(
    db_collection_name: str,
    fields: Sequence[str] | None = None,
    to_data_frame: Literal[True] = True,
    mongodb_config: dict[str, Any] | None = None,
    states: Sequence[str] | None = None,
    filter_dict: dict[str, Any] | None = None,
    parallel: bool = False,
    progress: bool = True,
) -> pd.DataFrame: ...


def get_results(
    db_collection_name: str,
    fields: Sequence[str] | None = None,
    to_data_frame: bool = False,
    mongodb_config: dict[str, Any] | None = None,
    states: Sequence[str] | None = None,
    filter_dict: dict[str, Any] | None = None,
    parallel: bool = False,
    progress: bool = True,
):
    """
    Get experiment results from the MongoDB.
    Parameters
    ----------
    db_collection_name: str
        Name of the MongoDB collection.
    fields: list (optional).
        Database attributes to extract. Default: ['config', 'result'].
    to_data_frame: bool, default: False
        Whether to convert the results into a Pandas DataFrame.
    mongodb_config: dict (optional)
        MongoDB credential dictionary. If None, uses the credentials specified by `seml configure`.
    states: list of strings (optional)
        Extract only experiments with certain states. Default: ['COMPLETED'].
    filter_dict: dict (optional)
        Custom dictionary for filtering results from the MongoDB.
    parallel: bool, default: False
        If True, unserialize entries in parallel. Use for very large experiment collections.
    progress: bool, default: False
        If True, show a progress bar.

    Returns
    -------

    """
    import functools

    import pandas as pd

    from seml.console import track

    if fields is None:
        fields = ['config', 'result']

    if states is None:
        states = States.COMPLETED

    if filter_dict is None:
        filter_dict = {}

    track = functools.partial(track, disable=not progress)

    collection = get_collection(
        db_collection_name,
        mongodb_config=mongodb_config,
    )

    if len(states) > 0:
        if 'status' in filter_dict:
            logging.warning(
                "'states' argument is not empty and will overwrite 'filter_dict['status']'."
            )
        filter_dict = deepcopy(filter_dict)
        filter_dict['status'] = {'$in': states}

    cursor = collection.find(filter_dict, fields)
    results = [x for x in track(cursor, total=collection.count_documents(filter_dict))]

    if parallel:
        from multiprocessing import Pool

        with Pool() as p:
            parsed = list(track(p.imap(parse_jsonpickle, results), total=len(results)))
        parsed = cast(List[ExperimentDoc], parsed)
    else:
        parsed = [parse_jsonpickle(entry) for entry in track(results)]
    if to_data_frame:
        parsed = pd.json_normalize(parsed, sep='.')  # type: ignore
    return parsed
