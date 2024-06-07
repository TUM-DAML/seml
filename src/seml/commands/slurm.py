import logging
import subprocess
from collections import defaultdict
from typing import Dict, Optional

from seml.commands.manage import detect_killed
from seml.database import build_filter_dict, get_collection
from seml.settings import SETTINGS
from seml.utils import s_if


def hold_or_release_experiments(
    hold: bool,
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
):
    """
    Holds or releases experiments that are currently in the SLURM queue.

    Parameters
    ----------
    hold : bool
        Whether to hold or release the experiments
    db_collection_name : str
        The collection to hold or release experiments from
    sacred_id : Optional[int], optional
        The ID of the experiment to hold or release, by default None
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    filter_dict : Optional[Dict], optional
        Additional filters, by default None
    """
    import shlex

    detect_killed(db_collection_name, False)

    filter_dict = build_filter_dict(
        [*SETTINGS.STATES.PENDING], batch_id, filter_dict, sacred_id
    )
    collection = get_collection(db_collection_name)
    experiments = list(
        collection.find(filter_dict, {'slurm.array_id': 1, 'slurm.task_id': 1})
    )

    arrays = defaultdict(list)
    n_experiments = len(experiments)
    for exp in experiments:
        arrays[exp['slurm']['array_id']].append(exp['slurm']['task_id'])

    slurm_ids = [
        f"{array_id}_[{','.join(map(str, task_ids))}]"
        for array_id, task_ids in arrays.items()
    ]
    opteration = 'hold' if hold else 'release'
    subprocess.run(
        f'scontrol {opteration} {shlex.quote(" ".join(slurm_ids))}',
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # User feedback
    op_name = 'Held' if hold else 'Released'
    logging.info(f'{op_name} {n_experiments} experiment{s_if(len(arrays))}.')
