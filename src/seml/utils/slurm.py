import functools
import os
import subprocess
import time
from typing import Dict, Optional, Union

from seml.settings import SETTINGS


@functools.lru_cache()
def get_cluster_name():
    """
    Retrieves the name of the cluster from the Slurm configuration.

    Returns
    -------
    str
        The name of the cluster
    """
    try:
        return (
            subprocess.run(
                "scontrol show config | grep ClusterName | awk '{print $3}'",
                shell=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    except subprocess.SubprocessError:
        return 'unknown'


def get_slurm_jobs(*job_ids: str):
    """
    Returns a list of dictionaries containing information about the Slurm jobs with the given job IDs.
    If no job IDs are provided, information about all jobs is returned.

    Parameters
    ----------
    job_ids : Optional[Sequence[str]]
        The job IDs of the jobs to get information about

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries containing information about the jobs
    """
    if len(job_ids) == 0:
        job_info_str = subprocess.run(
            'scontrol show job',
            shell=True,
            check=True,
            capture_output=True,
        ).stdout.decode('utf-8')
        job_info_strs = job_info_str.split('\n\n')
    else:
        job_info_strs = []
        for job_id in job_ids:
            job_info_str = subprocess.run(
                f'scontrol show job {job_id}',
                shell=True,
                check=True,
                capture_output=True,
            ).stdout.decode('utf-8')
            job_info_strs.append(job_info_str)
    job_info_strs = list(filter(None, job_info_strs))
    job_infos = list(map(parse_scontrol_job_info, job_info_strs))
    return job_infos


def parse_scontrol_job_info(job_info: str):
    """
    Converts the return value of `scontrol show job <jobid>` into a python dictionary.

    Parameters
    ----------
    job_info : str
        The output of `scontrol show job <jobid>`

    Returns
    -------
    dict
        The job information as a dictionary
    """
    job_info_dict: Dict[str, str] = {}
    # we may split to many times, e.g., if a value contains a space
    unfiltered_lines = job_info.split()
    filtered_lines = []
    for line in unfiltered_lines:
        if line:
            if '=' in line:
                # new variable
                filtered_lines.append(line)
            else:
                # just append to the previous variable
                filtered_lines[-1] += ' ' + line

    # Now every line must contain a '=' sign and we can simply split here
    for line in filtered_lines:
        key, value = line.split('=', 1)
        job_info_dict[key] = value
    return job_info_dict


def get_slurm_arrays_tasks(filter_by_user: bool = False):
    """Get a dictionary of running/pending Slurm job arrays (as keys) and tasks (as values)

    Parameters:
    -----------
    filter_by_user : bool
        Whether to only check jobs by the current user, by default False
    """
    try:
        squeue_cmd = f"SLURM_BITSTR_LEN=1024 squeue -a -t {','.join(SETTINGS.SLURM_STATES.ACTIVE)} -h -o %i"
        if filter_by_user:
            squeue_cmd += ' -u `whoami`'
        squeue_out = subprocess.run(
            squeue_cmd, shell=True, check=True, capture_output=True
        ).stdout
        jobs = [job_str for job_str in squeue_out.splitlines() if b'_' in job_str]
        if len(jobs) > 0:
            array_ids_str, task_ids = zip(*[job_str.split(b'_') for job_str in jobs])
            # `job_dict`: This dictionary has the job array IDs as keys and the values are
            # a list of 1) the pending job task range and 2) a list of running job task IDs.
            job_dict = {}
            for i, task_range_str in enumerate(task_ids):
                array_id = int(array_ids_str[i])
                if array_id not in job_dict:
                    job_dict[array_id] = [[range(0)], []]

                if b'[' in task_range_str:
                    # Remove brackets and maximum number of simultaneous jobs
                    task_range_str = task_range_str[1:-1].split(b'%')[0]
                    # The overall pending tasks array can be split into multiple arrays by cancelling jobs
                    job_id_ranges = task_range_str.split(b',')
                    for r in job_id_ranges:
                        if b'-' in r:
                            lower, upper = r.split(b'-')
                        else:
                            lower = upper = r
                        job_dict[array_id][0].append(range(int(lower), int(upper) + 1))
                else:
                    # Single task IDs belong to running jobs
                    task_id = int(task_range_str)
                    job_dict[array_id][1].append(task_id)

            return job_dict
        else:
            return {}
    except subprocess.CalledProcessError:
        return {}


def get_current_slurm_array_id():
    slurm_array_id = os.environ.get('SLURM_ARRAY_JOB_ID', None)
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
    return slurm_array_id, slurm_task_id


def get_current_slurm_job_id():
    return os.environ.get('SLURM_JOB_ID', None)


def cancel_slurm_jobs(*job_ids: str, state: Optional[str] = None):
    """
    Cancels the Slurm jobs with the given job IDs.

    Parameters
    ----------
    job_ids : Sequence[str]
        The job IDs of the jobs to cancel
    """
    job_str = ' '.join(map(str, job_ids))
    if state is not None:
        subprocess.run(f'scancel -t {state} {job_str}', shell=True, check=False)
    else:
        subprocess.run(f'scancel {job_str}', shell=True, check=False)


def are_slurm_jobs_running(*job_ids: str):
    """
    Checks the Slurm queue to see if the jobs with the given job IDs are still running.

    Parameters
    ----------
    job_ids : Sequence[str]
        The job IDs of the jobs to check

    Returns
    -------
    bool
        True if the jobs are still running, False otherwise
    """
    return (
        len(
            subprocess.run(
                f"squeue -h -o '%A' --jobs={','.join(job_ids)}",
                shell=True,
                check=True,
                capture_output=True,
            ).stdout
        )
        > 0
    )


def wait_until_slurm_jobs_finished(*job_ids: str, timeout: Union[int, float]):
    """
    Waits until all jobs are finished or until the timeout is reached.

    Parameters
    ----------
    job_ids: Sequence[str]
        The job IDs of the jobs to wait for
    timeout: Union[int, float]
        The maximum time to wait in seconds

    Returns
    -------
    bool
        True if the jobs finished before the timeout, False otherwise
    """
    end_time = time.time() + timeout
    while are_slurm_jobs_running(*job_ids):
        time.sleep(0.1)
        if time.time() > end_time:
            return False
    return True
