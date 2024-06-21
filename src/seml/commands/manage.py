import copy
import itertools
import logging
import re
import subprocess
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from seml.database import (
    build_filter_dict,
    delete_files,
    get_collection,
    get_database,
    get_mongodb_config,
)
from seml.experiment.config import (
    check_config,
    config_get_exclude_keys,
    generate_named_configs,
    resolve_configs,
    resolve_interpolations,
)
from seml.experiment.sources import delete_orphaned_sources, upload_sources
from seml.settings import SETTINGS
from seml.utils import (
    chunker,
    make_hash,
    s_if,
    utcnow,
)
from seml.utils.errors import MongoDBError
from seml.utils.slurm import (
    are_slurm_jobs_running,
    cancel_slurm_jobs,
    get_cluster_name,
    get_slurm_arrays_tasks,
    get_slurm_jobs,
    wait_until_slurm_jobs_finished,
)

States = SETTINGS.STATES

if TYPE_CHECKING:
    from pymongo.collection import Collection


def should_check_killed(filter_states: Optional[List[str]]) -> bool:
    """Checks whether killed experiments should be checked

    Parameters
    ----------
    filter_states : Optional[List[str]]
        The states to filter on

    Returns
    -------
    bool
        Whether killed experiments should be checked
    """
    return (
        filter_states is not None
        and len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states))
        > 0
    )


def cancel_empty_pending_jobs(db_collection_name: str, *sacred_ids: int):
    """Cancels pending jobs that are not associated with any experiment

    Parameters
    ----------
    db_collection_name : str
        The collection to check for pending jobs
    sacred_ids : int
        The IDs of the experiments to check
    """
    if len(sacred_ids) == 0:
        raise ValueError('At least one sacred ID must be provided.')
    collection = get_collection(db_collection_name)
    num_pending = collection.count_documents(
        {'status': {'$in': [*States.PENDING]}, '_id': {'$in': sacred_ids}}
    )
    if num_pending > 0:
        # There are still pending experiments, we don't want to cancel the jobs.
        return
    pending_exps = list(collection.find({'_id': {'$in': sacred_ids}}, {'slurm'}))
    array_ids = set(
        conf['array_id']
        for exp in pending_exps
        for conf in exp['slurm']
        if 'array_id' in conf
    )
    # Only cancel the pending jobs
    cancel_slurm_jobs(*array_ids, state='PENDING')


def cancel_jobs_without_experiments(*slurm_array_ids: str):
    """
    Cancels Slurm jobs that are not associated with any experiment that is still pending/running.

    Parameters
    ----------
    slurm_array_ids : str
        The array IDs of the Slurm jobs to check.
    """
    if len(slurm_array_ids) == 0:
        return []

    canceled_ids = []
    for array_id in slurm_array_ids:
        try:
            job_info = get_slurm_jobs(str(array_id))[0]
        except subprocess.CalledProcessError:
            # Job is not running, so we can skip this.
            continue
        col_name = job_info.get('Comment', None)
        if col_name is None:
            continue
        collection = get_collection(col_name)
        is_needed = (
            collection.count_documents(
                {
                    'slurm': {'$elemMatch': {'array_id': array_id}},
                    'status': {'$in': [*States.RUNNING, *States.PENDING]},
                },
                limit=1,
            )
            > 0
        )
        if not is_needed:
            cancel_slurm_jobs(str(array_id))
            canceled_ids.append(array_id)
    return canceled_ids


def cancel_experiment_by_id(
    collection: 'Collection',
    exp_id: int,
    set_interrupted: bool = True,
    slurm_dict: Optional[Dict] = None,
    wait: bool = False,
    timeout: int = SETTINGS.CANCEL_TIMEOUT,
):
    """Cancels a single experiment by its id

    Parameters
    ----------
    collection : str
        The collection this experiment belongs to
    exp_id : int
        The experiment id
    set_interrupted : bool, optional
        Whether to set the state of the experiment to INTERRUPTED, by default True
    slurm_dict : Optional[Dict], optional
        Optional updates to the slurm dict of the experiments, by default None
    wait : bool, optional
        Whether to wait for the cancellation by checking the slurm queue, by default False
    timeout : int, optional
        The timeout in seconds to wait for the cancellation, by default SETTINGS.CANCEL_TIMEOUT
    """

    exp = collection.find_one({'_id': exp_id})
    if exp is None:
        logging.error(f'No experiment found with ID {exp_id}.')
        return

    if slurm_dict:
        for s_conf in exp['slurm']:
            s_conf.update(slurm_dict)

    # check if the job has been scheduled at all
    array_ids = [conf.get('array_id', None) for conf in exp['slurm']]
    if any(array_id is None for array_id in array_ids):
        logging.error(f'Experiment with ID {exp_id} has not been started using Slurm.')
        return

    # check if the job has been claimed and associated with a concrete job
    is_running = 'array_id' in exp['execution']
    if is_running:
        job_str = f"{exp['execution']['array_id']}_{exp['execution']['task_id']}"
        job_strings = [job_str]
        filter_dict = {
            'execution.array_id': exp['execution']['array_id'],
            'execution.task_id': exp['execution']['task_id'],
        }
    else:
        job_strings = list(map(str, array_ids))

    # Check if slurm job exists
    if not are_slurm_jobs_running(*job_strings):
        logging.error(
            f'Slurm job {job_strings} of experiment '
            f'with ID {exp_id} is not pending/running in Slurm.'
        )
        return

    cancel_update = {
        '$set': {
            'status': States.INTERRUPTED[0],
            'stop_time': utcnow(),
        }
    }
    if set_interrupted:
        # Set the database state to INTERRUPTED
        collection.update_one({'_id': exp_id}, cancel_update)

    if is_running:
        # Check if other experiments are running in the same job
        other_exps_filter = filter_dict.copy()
        other_exps_filter['_id'] = {'$ne': exp_id}
        other_exps_filter['status'] = {'$in': [*States.RUNNING, *States.PENDING]}
        other_exp_running = collection.count_documents(other_exps_filter) >= 1

        # Cancel if no other experiments are running in the same job
        if not other_exp_running:
            cancel_slurm_jobs(job_str)
            # Wait until the job is actually gone
            if wait and not wait_until_slurm_jobs_finished(job_str, timeout=timeout):
                logging.error('Job did not cancel in time.')
                exit(1)

            if set_interrupted:
                # set state to interrupted again (might have been overwritten by Sacred in the meantime).
                collection.update_many(filter_dict, cancel_update)

    # Cancel jobs that will not execute anything
    cancel_jobs_without_experiments(*array_ids)


def cancel_experiments(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
    yes: bool = False,
    wait: bool = False,
    confirm_threshold: int = SETTINGS.CONFIRM_THRESHOLD.CANCEL,
    timeout: int = SETTINGS.CANCEL_TIMEOUT,
):
    """Cancels experiment(s)

    Parameters
    ----------
    db_collection_name : str
        The collection to cancel experiments of
    sacred_id : Optional[int], optional
        ID of the experiment to delete. Overrides other filters.
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    batch_id : Optional[int], optional
        Filter on experiment batch ids, by default None
    filter_dict : Optional[Dict], optional
        Additional filters on experiments, by default None
    yes : bool, optional
        Whether to override confirmation prompts, by default False
    wait : bool, optional
        Whether to wait for all experiments be cancelled (by checking the slurm queue), by default False
    confirm_threshold : int, optional
        The threshold for the number of experiments to cancel before asking for confirmation, by default SETTINGS.CONFIRM_THRESHOLD.CANCEL
    timeout : int, optional
        The timeout in seconds to wait for the cancellation, by default SETTINGS.CANCEL_TIMEOUT
    """
    from seml.console import prompt

    collection = get_collection(db_collection_name)
    # We check whether there are slurm jobs for which after this action no
    # RUNNING experiment remains. These slurm jobs can be killed altogether.
    # However, it is NOT possible right now to cancel a single experiment in a Slurm job with multiple
    # running experiments.
    try:
        if should_check_killed(filter_states):
            detect_killed(db_collection_name, print_detected=False)

        db_filter_dict = build_filter_dict(
            filter_states, batch_id, filter_dict, sacred_id=sacred_id
        )

        to_cancel_arr_ids = list(collection.find(db_filter_dict, {'slurm': 1}))
        ncancel = len(to_cancel_arr_ids)
        if sacred_id is not None and ncancel == 0:
            logging.error(f'No experiment found with ID {sacred_id}.')

        logging.info(f'Cancelling {ncancel} experiment{s_if(ncancel)}.')
        if ncancel >= confirm_threshold:
            if not yes and not prompt('Are you sure? (y/n)', type=bool):
                exit(1)

        running_filter = copy.deepcopy(db_filter_dict)
        running_filter = {
            **running_filter,
            'execution.cluster': get_cluster_name(),
            'execution.array_id': {'$exists': True},
        }
        running_exps = list(
            collection.find(
                running_filter,
                {'_id': 1, 'status': 1, 'execution': 1},
            )
        )
        # update database status and write the stop_time
        # this should be done early so that the experiments are not picked up by other processes
        cancel_update = {
            '$set': {
                'status': States.INTERRUPTED[0],
                'stop_time': utcnow(),
            }
        }
        collection.update_many(db_filter_dict, cancel_update)

        # Cancel pending jobs that will not execute anything
        array_ids = set(
            conf['array_id']
            for exp in to_cancel_arr_ids
            for conf in exp['slurm']
            if 'array_id' in conf
        )
        canceled = cancel_jobs_without_experiments(*array_ids)

        # set of slurm IDs in the database
        slurm_ids = set(
            [
                (e['execution']['array_id'], e['execution']['task_id'])
                for e in running_exps
            ]
        )
        # set of experiment IDs to be cancelled.
        exp_ids = set([e['_id'] for e in running_exps])
        to_cancel = set()

        # iterate over slurm IDs to check which slurm jobs can be cancelled altogether
        for a_id, t_id in slurm_ids:
            # find experiments RUNNING under the slurm job
            jobs_running = [
                e
                for e in running_exps
                if (
                    e['execution']['array_id'] == a_id
                    and e['execution']['task_id'] == t_id
                    and e['status'] in States.RUNNING
                )
            ]
            running_exp_ids = set(e['_id'] for e in jobs_running)
            if len(running_exp_ids.difference(exp_ids)) == 0:
                # there are no running jobs in this slurm job that should not be canceled.
                to_cancel.add(f'{a_id}_{t_id}')

        # cancel all Slurm jobs for which no running experiment remains.
        if len(to_cancel) > 0:
            chunk_size = 100
            chunks = list(chunker(list(to_cancel), chunk_size))
            [cancel_slurm_jobs(*chunk) for chunk in chunks]
            # Wait until all jobs are actually stopped.
            for chunk in chunks:
                if wait and not wait_until_slurm_jobs_finished(*chunk, timeout=timeout):
                    logging.error('Job did not cancel in time.')
                    exit(1)

        canceled = list(map(str, canceled + list(to_cancel)))
        n_canceled = len(canceled)
        if n_canceled > 0:
            logging.info(
                f'Canceled job{s_if(n_canceled)} with the following ID{s_if(n_canceled)}: '
                + ', '.join(canceled)
            )
        # Let's repeat this in case a job cleaned itself up and overwrote the status.
        collection.update_many(db_filter_dict, cancel_update)
    except subprocess.CalledProcessError:
        logging.warning(
            'One or multiple Slurm jobs were no longer running when I tried to cancel them.'
        )


def delete_experiments(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
    yes: bool = False,
    cancel: bool = True,
):
    """Deletes experiment(s).

    Parameters
    ----------
    db_collection_name : str
        The collection name to which to delete experiments from
    sacred_id : Optional[int], optional
        ID of the experiment to delete. Overrides other filters.
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    batch_id : Optional[int], optional
        Filter on experiment batch ids, by default None
    filter_dict : Optional[Dict], optional
        Additional filters on experiments, by default None
    yes : bool, optional
        Whether to override confirmation prompts, by default False
    """
    from seml.console import prompt

    # Before deleting, we should first cancel the experiments that are still running.
    if cancel:
        cancel_states = set(States.PENDING + States.RUNNING)
        if filter_states is not None and len(filter_states) > 0:
            cancel_states = cancel_states.intersection(filter_states)

        if len(cancel_states) > 0:
            cancel_experiments(
                db_collection_name,
                sacred_id,
                list(cancel_states),
                batch_id,
                filter_dict,
                yes=False,
                confirm_threshold=1,
                wait=True,
            )

    collection = get_collection(db_collection_name)
    experiment_files_to_delete = []

    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    ndelete = collection.count_documents(filter_dict)
    if sacred_id is not None and ndelete == 0:
        raise MongoDBError(f'No experiment found with ID {sacred_id}.')
    batch_ids = collection.find(filter_dict, {'batch_id'})
    batch_ids_in_del = set([x.get('batch_id', -1) for x in batch_ids])

    logging.info(
        f'Deleting {ndelete} configuration{s_if(ndelete)} from database collection.'
    )
    if ndelete >= SETTINGS.CONFIRM_THRESHOLD.DELETE:
        if not yes and not prompt('Are you sure? (y/n)', type=bool):
            exit(1)

    # Collect sources uploaded by sacred.
    exp_sources_list = collection.find(
        filter_dict, {'experiment.sources': 1, 'artifacts': 1}
    )
    for exp in exp_sources_list:
        experiment_files_to_delete.extend(get_experiment_files(exp))
    result = collection.delete_many(filter_dict)
    if not result.deleted_count == ndelete:
        logging.error(
            f'Only {result.deleted_count} of {ndelete} experiments were deleted.'
        )

    # Delete sources uploaded by sacred.
    delete_files(collection.database, experiment_files_to_delete)
    logging.info(
        f'Deleted {len(experiment_files_to_delete)} files associated with deleted experiments.'
    )

    if len(batch_ids_in_del) > 0:
        # clean up the uploaded sources if no experiments of a batch remain
        delete_orphaned_sources(collection, batch_ids_in_del)

    if collection.count_documents({}) == 0:
        collection.drop()


def drop_collections(
    pattern: str, mongodb_config: Optional[Dict] = None, yes: bool = False
):
    """
    Drops collections matching the given pattern.

    Parameters
    ----------
    pattern : str
        The regex collection names have to match against
    mongodb_config : dict or None
        A configuration for the mongodb. If None, the standard config is used.
    yes : bool
        Whether to override confirmation prompts
    """
    from seml.console import list_items, prompt

    # Get the database
    if mongodb_config is None:
        mongodb_config = get_mongodb_config()
    db = get_database(**mongodb_config)
    expression = re.compile(pattern)
    collection_names = [
        name
        for name in db.list_collection_names()
        if name not in ('fs.chunks', 'fs.files') and expression.match(name)
    ]
    if len(collection_names) == 0:
        logging.info('No collections found.')
        return
    if not yes:
        logging.info(
            f'The following {len(collection_names)} collection will be deleted:'
        )
        list_items(collection_names)
        if not prompt('Are you sure? (y/n)', type=bool):
            return
    for name in collection_names:
        delete_experiments(name, yes=True)


def reset_slurm_dict(exp: Dict):
    """Resets the slurm dict of an experiment

    Parameters
    ----------
    exp : Dict
        The experiment of which to reset the slurm dict
    """
    keep_slurm = set()
    keep_slurm.update(SETTINGS.VALID_SLURM_CONFIG_VALUES)
    for sub_conf in exp['slurm']:
        slurm_keys = set(sub_conf.keys())
        for key in slurm_keys - keep_slurm:
            del sub_conf[key]

        # Clean up sbatch_options dictionary
        remove_sbatch = {'job-name', 'output', 'array', 'comment'}
        sbatch_keys = set(sub_conf['sbatch_options'].keys())
        for key in remove_sbatch & sbatch_keys:
            del sub_conf['sbatch_options'][key]


def reset_single_experiment(collection: 'Collection', exp: Dict):
    """Resets a single experiment

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to which the experiment belongs to
    exp : Dict
        The experiment dict
    """
    exp['status'] = States.STAGED[0]
    # queue_time for backward compatibility.
    keep_entries = [
        'batch_id',
        'status',
        'seml',
        'slurm',
        'config',
        'config_hash',
        'add_time',
        'queue_time',
        'git',
        'config_unresolved',
    ]

    # Clean up SEML dictionary
    keep_seml = set(['source_files', 'working_dir', SETTINGS.SEML_CONFIG_VALUE_VERSION])
    keep_seml.update(SETTINGS.VALID_SEML_CONFIG_VALUES)
    seml_keys = set(exp['seml'].keys())
    for key in seml_keys - keep_seml:
        del exp['seml'][key]

    reset_slurm_dict(exp)

    collection.replace_one(
        {'_id': exp['_id']},
        {entry: exp[entry] for entry in keep_entries if entry in exp},
        upsert=False,
    )


def reset_experiments(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
    yes: bool = False,
):
    """Resets experiments

    Parameters
    ----------
    db_collection_name : str
        The name of the collection to resets experiments from
    sacred_id : Optional[int], optional
        If given, the id of the experiment to reset. Overrides other filters, by default None
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    filter_dict : Optional[Dict], optional
        Additional filters, by default None
    yes : bool, optional
        Whether to override confirmation prompts, by default False
    """
    from seml.console import prompt

    collection = get_collection(db_collection_name)
    if should_check_killed(filter_states):
        detect_killed(db_collection_name, print_detected=False)
    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    nreset = collection.count_documents(filter_dict)
    exps = collection.find(filter_dict)
    if sacred_id is not None and nreset == 0:
        raise MongoDBError(f'No experiment found with ID {sacred_id}.')

    logging.info(f'Resetting the state of {nreset} experiment{s_if(nreset)}.')
    if nreset >= SETTINGS.CONFIRM_THRESHOLD.RESET:
        if not yes and not prompt('Are you sure? (y/n)', type=bool):
            exit(1)
    for exp in exps:
        reset_single_experiment(collection, exp)


def detect_killed(db_collection_name: str, print_detected: bool = True):
    """Detects killed experiments by checking the slurm status

    Parameters
    ----------
    db_collection_name : str
        The collection to check killed experiments from
    print_detected : bool, optional
        Whether to print how many killed experiments have been detected, by default True
    """
    collection = get_collection(db_collection_name)
    cluster = get_cluster_name()
    exps = collection.find(
        {
            'status': {'$in': [*States.PENDING, *States.RUNNING]},
            'execution.cluster': cluster,  # only check experiments that are running on the current cluster
            # Previously we only checked for started experiments by including the following line:
            # 'host': {'$exists': True},  # only check experiments that have been started
            # Though, this does not catch the case where a user cancels pending experiments with scancel.
            # I (Nicholas) am not 100% sure about the implications of removing the check but it at least
            # resolves the issue around manually canceled jobs.
        }
    )
    running_jobs = get_slurm_arrays_tasks()
    nkilled = 0
    for exp in exps:
        # detect whether the experiment is running in slurm
        exp_running = exp['execution'].get('array_id', -1) in running_jobs and (
            any(
                exp['execution']['task_id'] in r
                for r in running_jobs[exp['execution']['array_id']][0]
            )
            or exp['execution']['task_id']
            in running_jobs[exp['execution']['array_id']][1]
        )
        # detect whether any job that could execute it is pending
        array_ids = [conf['array_id'] for conf in exp['slurm']]
        # Any of these jobs may still pull the experiment and run it
        exp_pending = any(array_id in running_jobs for array_id in array_ids)

        if not exp_running and not exp_pending:
            if 'stop_time' in exp:
                # the experiment is already over but failed to report properly
                collection.update_one(
                    {'_id': exp['_id']}, {'$set': {'status': States.INTERRUPTED[0]}}
                )
            else:
                # the experiment was externally killed
                nkilled += 1
                collection.update_one(
                    {'_id': exp['_id']}, {'$set': {'status': States.KILLED[0]}}
                )
                try:
                    with open(exp['seml']['output_file'], 'r', errors='replace') as f:
                        all_lines = f.readlines()
                    collection.update_one(
                        {'_id': exp['_id']}, {'$set': {'fail_trace': all_lines[-4:]}}
                    )
                except IOError:
                    # If the experiment is cancelled before starting (e.g. when still queued), there is not output file.
                    logging.verbose(
                        f"File {exp['seml']['output_file']} could not be read."
                    )
                except KeyError:
                    logging.verbose(
                        f"Output file not found in experiment {exp['_id']}."
                    )
    if print_detected:
        logging.info(f'Detected {nkilled} externally killed experiment{s_if(nkilled)}.')


def get_experiment_files(experiment: Dict) -> List[str]:
    """Gets the file ids of files associated with an experiment

    Parameters
    ----------
    experiment : Dict
        The experiment dict

    Returns
    -------
    List[str]
        A list of file ids
    """
    experiment_files = []
    if 'experiment' in experiment:
        if 'sources' in experiment['experiment']:
            exp_sources = experiment['experiment']['sources']
            experiment_files.extend([x[1] for x in exp_sources])
    if 'artifacts' in experiment:
        experiment_files.extend([x['file_id'] for x in experiment['artifacts']])
    return experiment_files


def reload_sources(
    db_collection_name: str,
    batch_ids: Optional[List[int]] = None,
    keep_old: bool = False,
    yes: bool = False,
):
    """Reloads the sources of experiment(s)

    Parameters
    ----------
    db_collection_name : str
        The collection to reload sources from
    batch_ids : Optional[List[int]], optional
        Filter on the batch ids, by default None
    keep_old : bool, optional
        Whether to keep old source files in the fs, by default False
    yes : bool, optional
        Whether to override confirmation prompts, by default False
    resolve : bool, optional
        Whether to re-resolve the config values
    """
    from importlib.metadata import version

    import gridfs
    from pymongo import UpdateOne

    from seml.console import prompt

    collection = get_collection(db_collection_name)

    if batch_ids is not None and len(batch_ids) > 0:
        filter_dict = {'batch_id': {'$in': list(batch_ids)}}
    else:
        filter_dict = {}
    db_results = list(
        collection.find(
            filter_dict, {'batch_id', 'seml', 'config', 'status', 'config_unresolved'}
        )
    )
    id_to_document = {}
    for bid, documents in itertools.groupby(db_results, lambda x: x['batch_id']):
        id_to_document[bid] = list(documents)
    states = {x['status'] for x in db_results}

    if any([s in (States.RUNNING + States.PENDING + States.COMPLETED) for s in states]):
        logging.info(
            'Some of the experiments is still in RUNNING, PENDING or COMPLETED.'
        )
        if not yes and not prompt('Are you sure? (y/n)', type=bool):
            exit(1)

    for batch_id, documents in id_to_document.items():
        seml_config = documents[0]['seml']

        version_seml_config = seml_config.get(
            SETTINGS.SEML_CONFIG_VALUE_VERSION, (0, 0, 0)
        )
        version_str = '.'.join(map(str, version_seml_config))
        if version_str != version('seml'):
            logging.warn(
                f'Batch {batch_id} was added with seml version "{version_str}" '
                f'which mismatches the current version {version("seml")}'
            )

        if 'working_dir' not in seml_config or not seml_config['working_dir']:
            logging.error(f'Batch {batch_id}: No source files to refresh.')
            continue

        if any(
            document.get('config_unresolved', None) is None for document in documents
        ):
            logging.warn(
                f'Batch {batch_id}: Some experiments do not have an unresolved configuration. '
                'The resolved configuration "config" will be used for resolution instead.'
            )
        configs_unresolved = [
            document.get('config_unresolved', document['config'])
            for document in documents
        ]
        configs, named_configs = generate_named_configs(configs_unresolved)
        configs = resolve_configs(
            seml_config['executable'],
            seml_config['conda_environment'],
            configs,
            named_configs,
            seml_config['working_dir'],
        )

        # If the seed was explicited, it should be kept for the new resolved config when reloading resources
        for config, config_unresolved in zip(configs, configs_unresolved):
            if SETTINGS.CONFIG_KEY_SEED in configs_unresolved:
                config[SETTINGS.CONFIG_KEY_SEED] = config_unresolved[
                    SETTINGS.CONFIG_KEY_SEED
                ]

        documents = [
            resolve_interpolations(
                {
                    **{**document, 'config': config},
                    'config_unresolved': config_unresolved,
                }
            )
            for document, config, config_unresolved in zip(
                documents, configs, configs_unresolved
            )
        ]

        result = collection.bulk_write(
            [
                UpdateOne(
                    {'_id': document['_id']},
                    {
                        '$set': {
                            'config': document['config'],
                            'config_unresolved': document['config_unresolved'],
                            'config_hash': make_hash(
                                document['config'],
                                config_get_exclude_keys(
                                    document['config'], document['config_unresolved']
                                ),
                            ),
                        }
                    },
                )
                for document in documents
            ]
        )
        logging.info(
            f'Batch {batch_id}: Resolved configurations of {result.matched_count} experiments against new source files ({result.modified_count} changed).'
        )

        # Check whether the configurations aligns with the current source code
        check_config(
            seml_config['executable'],
            seml_config['conda_environment'],
            [document['config'] for document in documents],
            seml_config['working_dir'],
        )

        # Find the currently used source files
        db = collection.database
        fs = gridfs.GridFS(db)
        fs_filter_dict = {
            'metadata.batch_id': batch_id,
            'metadata.collection_name': f'{collection.name}',
            'metadata.deprecated': {'$exists': False},
        }
        current_source_files = db['fs.files'].find(filter_dict, '_id')
        current_ids = [x['_id'] for x in current_source_files]
        fs_filter_dict = {'_id': {'$in': current_ids}}
        # Deprecate them
        db['fs.files'].update_many(
            fs_filter_dict, {'$set': {'metadata.deprecated': True}}
        )
        try:
            # Try to upload the new ones
            source_files = upload_sources(seml_config, collection, batch_id)
        except Exception as e:
            # If it fails we reconstruct the old ones
            logging.error(
                f'Batch {batch_id}: Source import failed. Restoring old files.'
            )
            db['fs.files'].update_many(
                fs_filter_dict, {'$unset': {'metadata.deprecated': ''}}
            )
            raise e

        try:
            # Try to assign the new ones to the experiments
            filter_dict = {'batch_id': batch_id}
            collection.update_many(
                filter_dict, {'$set': {'seml.source_files': source_files}}
            )
            logging.info(f'Batch {batch_id}: Successfully reloaded source code.')
        except Exception as e:
            logging.error(f'Batch {batch_id}: Failed to set new source files.')
            # Delete new source files from DB
            for to_delete in source_files:
                fs.delete(to_delete[1])
            raise e

        # Delete the old source files
        if not keep_old:
            fs_filter_dict = {
                'metadata.batch_id': batch_id,
                'metadata.collection_name': f'{collection.name}',
                'metadata.deprecated': True,
            }
            source_files_old = [
                x['_id'] for x in db['fs.files'].find(fs_filter_dict, {'_id'})
            ]
            for to_delete in source_files_old:
                fs.delete(to_delete)


def detect_duplicates(
    db_collection_name: str, filter_dict: Optional[Dict] = None
) -> List[Set[int]]:
    """Finds duplicate configurations based on their hashes.

    Parameters
    ----------
    db_collection_name : str
        The collection to check

    Returns
    -------
    List[Set[int]]
        All duplicate experiments.
    """
    collection = get_collection(db_collection_name)
    pipeline = [
        {
            '$group': {
                '_id': '$config_hash',
                'ids': {'$addToSet': '$_id'},
                'count': {'$sum': 1},
            }
        },
        {
            '$match': {
                'count': {'$gt': 1},
            }
        },
    ]
    if filter_dict is not None:
        pipeline = [{'$match': filter_dict}] + pipeline
    duplicates = collection.aggregate(pipeline)
    return [set(duplicate['ids']) for duplicate in duplicates]
