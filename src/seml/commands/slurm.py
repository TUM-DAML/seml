import logging
import subprocess
from typing import Optional

from seml.commands.manage import detect_killed
from seml.database import build_filter_dict, get_collection
from seml.settings import SETTINGS
from seml.utils import s_if


def hold_or_release_experiments(
    hold: bool,
    db_collection_name: str,
    batch_id: Optional[int] = None,
):
    """
    Holds or releases experiments that are currently in the SLURM queue.

    Parameters
    ----------
    hold : bool
        Whether to hold or release the experiments
    db_collection_name : str
        The collection to hold or release experiments from
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    """
    import shlex

    detect_killed(db_collection_name, False)

    filter_dict = build_filter_dict([*SETTINGS.STATES.PENDING], batch_id, None, None)
    collection = get_collection(db_collection_name)
    experiments = list(collection.find(filter_dict, {'slurm': 1}))

    arrays = set()
    for exp in experiments:
        for slurm in exp['slurm']:
            if (
                'array_id' not in slurm
            ):  # Skip experiments that are not in the SLURM queue
                continue
            arrays.add(slurm['array_id'])

    opteration = 'hold' if hold else 'release'
    subprocess.run(
        f'scontrol {opteration} {shlex.quote(" ".join(map(str, arrays)))}',
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # User feedback
    op_name = 'Held' if hold else 'Released'
    n_exp = len(experiments)
    n_job = len(arrays)
    logging.info(
        f'{op_name} {n_exp} experiment{s_if(n_exp)} in {n_job} job{s_if(n_job)}.'
    )
