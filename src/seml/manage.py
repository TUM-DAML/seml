import copy
import datetime
import itertools
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

from seml.config import (
    check_config,
    generate_named_configs,
    resolve_configs,
    config_get_exclude_keys,
    resolve_interpolations,
)
from seml.database import (
    build_filter_dict,
    get_collection,
    get_database,
    get_mongodb_config,
)
from seml.errors import MongoDBError
from seml.settings import SETTINGS
from seml.sources import delete_files, delete_orphaned_sources, upload_sources
from seml.utils import (
    chunker,
    find_jupyter_host,
    flatten,
    get_from_nested,
    make_hash,
    resolve_projection_path_conflicts,
    s_if,
    slice_to_str,
    to_hashable,
    to_slices,
)

States = SETTINGS.STATES


def cancel_experiment_by_id(
    collection: str,
    exp_id: int,
    set_interrupted: bool = True,
    slurm_dict: Optional[Dict] = None,
    wait: bool = False,
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
    """

    exp = collection.find_one({'_id': exp_id})
    if slurm_dict:
        exp['slurm'].update(slurm_dict)

    if exp is not None:
        if 'array_id' in exp['slurm']:
            job_str = f"{exp['slurm']['array_id']}_{exp['slurm']['task_id']}"
            filter_dict = {
                'slurm.array_id': exp['slurm']['array_id'],
                'slurm.task_id': exp['slurm']['task_id'],
            }
        else:
            logging.error(
                f'Experiment with ID {exp_id} has not been started using Slurm.'
            )
            return

        try:
            # Check if job exists
            subprocess.run(
                f'scontrol show jobid -dd {job_str}',
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            if set_interrupted:
                # Set the database state to INTERRUPTED
                collection.update_one(
                    {'_id': exp_id}, {'$set': {'status': States.INTERRUPTED[0]}}
                )

            # Check if other experiments are running in the same job
            other_exps_filter = filter_dict.copy()
            other_exps_filter['_id'] = {'$ne': exp_id}
            other_exps_filter['status'] = {'$in': [*States.RUNNING, *States.PENDING]}
            other_exp_running = collection.count_documents(other_exps_filter) >= 1

            # Cancel if no other experiments are running in the same job
            if not other_exp_running:
                subprocess.run(f'scancel {job_str}', shell=True, check=True)
                # Wait until the job is actually gone
                if wait:
                    while (
                        len(
                            subprocess.run(
                                f"squeue -h -o '%A' -j{job_str}",
                                shell=True,
                                check=True,
                                capture_output=True,
                            ).stdout
                        )
                        > 0
                    ):
                        time.sleep(0.5)
                if set_interrupted:
                    # set state to interrupted again (might have been overwritten by Sacred in the meantime).
                    collection.update_many(
                        filter_dict,
                        {
                            '$set': {
                                'status': States.INTERRUPTED[0],
                                'stop_time': datetime.datetime.utcnow(),
                            }
                        },
                    )

        except subprocess.CalledProcessError:
            logging.error(
                f'Slurm job {job_str} of experiment '
                f'with ID {exp_id} is not pending/running in Slurm.'
            )
    else:
        logging.error(f'No experiment found with ID {exp_id}.')


def cancel_experiments(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
    yes: bool = False,
    wait: bool = False,
    confirm_threshold: int = SETTINGS.CONFIRM_THRESHOLD.CANCEL,
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
    """
    from seml.console import prompt

    collection = get_collection(db_collection_name)
    # We check whether there are slurm jobs for which after this action no
    # RUNNING experiment remains. These slurm jobs can be killed altogether.
    # However, it is NOT possible right now to cancel a single experiment in a Slurm job with multiple
    # running experiments.
    try:
        if (
            len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states))
            > 0
        ):
            detect_killed(db_collection_name, print_detected=False)

        filter_dict = build_filter_dict(
            filter_states, batch_id, filter_dict, sacred_id=sacred_id
        )

        ncancel = collection.count_documents(filter_dict)
        if sacred_id is not None and ncancel == 0:
            logging.error(f'No experiment found with ID {sacred_id}.')

        logging.info(f'Cancelling {ncancel} experiment{s_if(ncancel)}.')
        if ncancel >= confirm_threshold:
            if not yes and not prompt('Are you sure? (y/n)', type=bool):
                exit(1)

        filter_dict_new = copy.deepcopy(filter_dict)
        filter_dict_new.update({'slurm.array_id': {'$exists': True}})
        exps = list(
            collection.find(
                filter_dict_new,
                {'_id': 1, 'status': 1, 'slurm.array_id': 1, 'slurm.task_id': 1},
            )
        )
        # set of slurm IDs in the database
        slurm_ids = set([(e['slurm']['array_id'], e['slurm']['task_id']) for e in exps])
        # set of experiment IDs to be cancelled.
        exp_ids = set([e['_id'] for e in exps])
        to_cancel = set()

        # iterate over slurm IDs to check which slurm jobs can be cancelled altogether
        for a_id, t_id in slurm_ids:
            # find experiments RUNNING under the slurm job
            jobs_running = [
                e
                for e in exps
                if (
                    e['slurm']['array_id'] == a_id
                    and e['slurm']['task_id'] == t_id
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
            [
                subprocess.run(f"scancel {' '.join(chunk)}", shell=True, check=True)
                for chunk in chunks
            ]
            # Wait until all jobs are actually stopped.
            if wait:
                for chunk in chunks:
                    while (
                        len(
                            subprocess.run(
                                f"squeue -h -o '%A' --jobs={','.join(chunk)}",
                                shell=True,
                                check=True,
                                capture_output=True,
                            ).stdout
                        )
                        > 0
                    ):
                        time.sleep(0.5)

        # update database status and write the stop_time
        collection.update_many(
            filter_dict,
            {
                '$set': {
                    'status': States.INTERRUPTED[0],
                    'stop_time': datetime.datetime.utcnow(),
                }
            },
        )
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
    batch_ids_in_del = set([x['batch_id'] for x in batch_ids])

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
    slurm_keys = set(exp['slurm'].keys())
    for key in slurm_keys - keep_slurm:
        del exp['slurm'][key]

    # Clean up sbatch_options dictionary
    remove_sbatch = {'job-name', 'output', 'array'}
    sbatch_keys = set(exp['slurm']['sbatch_options'].keys())
    for key in remove_sbatch & sbatch_keys:
        del exp['slurm']['sbatch_options'][key]


def reset_single_experiment(collection: str, exp: Dict):
    """Resets a single experiment

    Parameters
    ----------
    collection : str
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
    if len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states)) > 0:
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
    exps = collection.find(
        {
            'status': {'$in': [*States.PENDING, *States.RUNNING]},
            # Previously we only checked for started experiments by including the following line:
            # 'host': {'$exists': True},  # only check experiments that have been started
            # Though, this does not catch the case where a user cancels pending experiments with scanel.
            # I (Nicholas) am not 100% sure about the implications of removing the check but it at least
            # resolves the issue around manually canceled jobs.
        }
    )
    running_jobs = get_slurm_arrays_tasks()
    nkilled = 0
    for exp in exps:
        exp_running = (
            'array_id' in exp['slurm']
            and exp['slurm']['array_id'] in running_jobs
            and (
                any(
                    exp['slurm']['task_id'] in r
                    for r in running_jobs[exp['slurm']['array_id']][0]
                )
                or exp['slurm']['task_id'] in running_jobs[exp['slurm']['array_id']][1]
            )
        )
        if not exp_running:
            if 'stop_time' in exp:
                collection.update_one(
                    {'_id': exp['_id']}, {'$set': {'status': States.INTERRUPTED[0]}}
                )
            else:
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
    import gridfs
    from pymongo import UpdateOne
    from importlib.metadata import version
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

        version_seml_config = seml_config.get(SETTINGS.SEML_CONFIG_VALUE_VERSION, None)
        if version_seml_config != version('seml'):
            logging.warn(
                f'Batch {batch_id} was added with seml version "{version_seml_config}" '
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


def print_fail_trace(
    db_collection_name: str,
    sacred_id: Optional[int],
    filter_states: Optional[List[str]],
    batch_id: Optional[int],
    filter_dict: Optional[Dict],
    projection: Optional[List[str]] = None,
):
    """Convenience function that prints the fail trace of experiments

    Parameters
    ----------
    db_collection_name : str
        Name of the collection to print traces of
    sacred_id : Optional[int]
        Optional filter on the experiment ID
    filter_states : Optional[List[str]]
        Optional filter on the experiment states
    batch_id : Optional[int]
        Optional filter on the experiment batch ID
    filter_dict : Optional[Dict]
        Optional filters
    projection : Optional[List[str]]
        Additional values to print per failed experiment, by default None
    """
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    from seml.console import console, Table

    detect_killed(db_collection_name, print_detected=False)
    collection = get_collection(db_collection_name)
    if projection is None:
        projection = []
    mongo_db_projection = resolve_projection_path_conflicts(
        {
            **{
                '_id': 1,
                'status': 1,
                'slurm.array_id': 1,
                'slurm.task_id': 1,
                'fail_trace': 1,
                'seml.description': 1,
                'batch_id': 1,
            },
            **{key: 1 for key in projection},
        }
    )

    if sacred_id is None:
        filter_dict = build_filter_dict(filter_states, batch_id, filter_dict)
        exps = list(collection.find(filter_dict, mongo_db_projection))
    else:
        exps = [collection.find_one({'_id': sacred_id}, mongo_db_projection)]
    for exp in exps:
        exp_id = exp.get('_id')
        status = exp.get('status')
        batch_id = exp.get('batch_id')
        slurm_array_id = exp.get('slurm', {}).get('array_id', None)
        slurm_task_id = exp.get('slurm', {}).get('task_id', None)
        fail_trace = exp.get('fail_trace', [])
        description = exp.get('seml', {}).get('description', None)
        header = (
            f'Experiment ID {exp_id}, '
            f'Batch ID {batch_id}, '
            f'Status: "{status}", '
            f'Slurm Array-Task id: {slurm_array_id}-{slurm_task_id}'
        )

        renderables = []
        if description is not None:
            text_description = Text()
            text_description.append('Description: ', style='bold magenta')
            text_description.append(description)
            renderables += [text_description, Rule(Text('Fail-Trace', style='bold'))]
        renderables.append(''.join(['\t' + line for line in fail_trace] + []).strip())
        if len(projection) > 0:
            table_projection = Table(show_header=False)
            projection_keys_flat = [
                key
                for key in flatten(exp)
                if any(key.startswith(p) for p in projection)
            ]
            for key in projection_keys_flat:
                table_projection.add_row(key, str(get_from_nested(exp, key)))
            renderables += [
                Rule(Text('Projection', style='bold')),
                Align(table_projection, align='left'),
            ]
        panel = Panel(
            Group(*renderables),
            title=console.render_str(header, highlight=True),
            highlight=True,
            border_style='red',
        )
        console.print(panel)
    logging.info(f'Printed the fail traces of {len(exps)} experiment(s).')


def print_status(
    db_collection_name: str,
    update_status: bool = True,
    projection: Optional[List[str]] = None,
):
    """Prints the status of an experiment collection

    Parameters
    ----------
    db_collection_name : str
        Name of the collection to print status of
    update_status : bool, optional
        Whehter to detect killed experiments, by default True
    projection : Optional[List[str]], optional
        Additional attributes from the MongoDB to print, by default None
    """

    from rich.align import Align
    from rich.table import Column
    from seml.console import console, Table

    collection = get_collection(db_collection_name)

    # Handle status updates
    if update_status:
        detect_killed(db_collection_name, print_detected=False)
    else:
        logging.warning(
            f'Status of {States.RUNNING[0]} experiments may not reflect if they have died or been canceled. Use the `--update-status` flag instead.'
        )

    if projection is None:
        projection = []
    projection = list(resolve_projection_path_conflicts({key: 1 for key in projection}))

    result = collection.aggregate(
        [
            {
                '$group': {
                    '_id': '$status',
                    'ids': {'$addToSet': '$_id'},
                    'batch_ids': {'$addToSet': '$batch_id'},
                    'descriptions': {'$addToSet': '$seml.description'},
                    'count': {'$sum': 1},
                    **{
                        f'projection_{idx}': {'$addToSet': f'${key}'}
                        for idx, key in enumerate(projection)
                    },
                }
            }
        ]
    )
    result = sorted(result, key=lambda x: list(States.keys()).index(x['_id']))
    show_descriptions = any(len(row['descriptions']) > 0 for row in result)
    # Unpack the (nested) projections
    # We keep prefixes encoded as ${id} to preserve the order of the projection keys
    result_projection = []
    for record in result:
        result_projection.append(defaultdict(set))
        for idx, key in enumerate(projection):
            for values in record[f'projection_{idx}']:
                for x, y in flatten({f'${idx}': values}).items():
                    result_projection[-1][x].add(to_hashable(y))
    projection_columns = sorted(set(k for record in result_projection for k in record))
    # For the column headers, we replace ${id} with the projection key
    columns = []
    for projection_column in projection_columns:
        projection_key_idx = int(
            re.match(r'.*\$([0-9]+)(\..*|$)', projection_column).groups()[0]
        )
        columns.append(
            projection_column.replace(
                f'${projection_key_idx}', projection[projection_key_idx]
            )
        )
    duplicate_experiment_ids = set(
        experiment_id
        for dups in detect_duplicates(db_collection_name)
        for experiment_id in dups
    )

    if show_descriptions:
        columns.insert(0, 'Descriptions')
    table = Table(
        Column('Status', justify='left', footer='Total'),
        Column(
            'Count',
            justify='left',
            footer=str(sum(record['count'] for record in result)),
        ),
        Column('Experiment IDs', justify='left'),
        Column('Batch IDs', justify='left'),
        Column('Duplicates', footer=str(len(duplicate_experiment_ids))),
        *[Column(key, justify='left') for key in columns],
        show_header=True,
        show_footer=len(result) > 1,
    )
    for record, record_projection in zip(result, result_projection):
        row = [
            record['_id'],
            str(record['count']),
            ', '.join(map(slice_to_str, to_slices(record['ids']))),
            ', '.join(map(slice_to_str, to_slices(record['batch_ids']))),
            str(len(set(record['ids']) & duplicate_experiment_ids)),
        ]
        if show_descriptions:
            row.append(
                ', '.join(
                    [f'"{description}"' for description in record['descriptions']]
                    if len(record['descriptions']) > 1
                    else record['descriptions']
                )
            )
        row += [
            ', '.join(map(str, record_projection.get(key, {})))
            for key in projection_columns
        ]
        table.add_row(*row)
    console.print(Align(table, align='center'))


def list_database(
    pattern: str,
    mongodb_config: Optional[Dict] = None,
    progress: bool = False,
    list_empty: bool = False,
    update_status: bool = False,
    print_full_description: bool = False,
):
    """
    Prints a tabular version of multiple collections and their states (without resolving RUNNING experiments that may have been canceled manually).

    Parameters
    ----------
    pattern : str
        The regex collection names have to match against
    mongodb_config : dict or None
        A configuration for the mongodb. If None, the standard config is used.
    progress : bool
        Whether to use a progress bar for fetching
    list_empty : bool
        Whether to list collections that have no documents associated with any state
    update_status : bool
        Whether to update the status of experiments by checking log files. This may take a while.
    print_full_description : bool
        Whether to print full descriptions (wrap-arround) or truncate the descriptions otherwise.
    """
    import pandas as pd
    from rich.align import Align
    from rich.table import Column
    from seml.console import track

    from seml.console import console, Table

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
    # Handle status updates
    if update_status:
        for collection in collection_names:
            detect_killed(collection, print_detected=False)
    else:
        logging.warning(
            f'Status of {States.RUNNING[0]} experiments may not reflect if they have died or been canceled. Use the `--update-status` flag instead.'
        )

    # Count the number of experiments in each state
    name_to_counts = defaultdict(lambda: {state: 0 for state in States.keys()})
    name_to_descriptions = defaultdict(lambda: '')
    it = track(collection_names, disable=not progress)

    inv_states = {v: k for k, states in States.items() for v in states}
    show_description = False
    for collection_name in it:
        counts_by_status = db[collection_name].aggregate(
            [
                {
                    '$group': {
                        '_id': '$status',
                        '_count': {'$sum': 1},
                        'description': {'$addToSet': '$seml.description'},
                    }
                }
            ]
        )
        descriptions = db[collection_name].aggregate(
            [{'$group': {'_id': '$seml.description'}}]
        )
        descriptions = [
            result['_id'] for result in descriptions if result['_id'] is not None
        ]
        name_to_counts[collection_name].update(
            {
                inv_states[result['_id']]: result['_count']
                for result in counts_by_status
                if result['_id'] in inv_states
            }
        )
        if len(descriptions) > 1:
            descriptions = [f'"{description}"' for description in descriptions]
        if len(descriptions) > 0:
            show_description = True
        name_to_descriptions[collection_name] = ', '.join(descriptions)

    if len(name_to_counts) == 0:
        logging.info(f'Found no collection matching "{pattern}"!')
        return

    df = pd.DataFrame.from_dict(name_to_counts, dtype=int).transpose()
    # Remove empty collections
    if not list_empty:
        df = df[df.sum(axis=1) > 0]
    # sort rows and columns
    df = df.sort_index()[States.keys()]
    # add a column with the total
    df['Total'] = df.sum(axis=1)

    totals = df.sum(axis=0)
    max_len = max(map(len, collection_names))
    columns = [
        Column('Collection', justify='left', footer='Total', min_width=max_len),
    ] + [
        Column(state.capitalize(), justify='right', footer=str(totals[state]))
        for state in df.columns
    ]
    if show_description:
        columns.append(
            Column(
                'Description(s)',
                justify='left',
                max_width=console.width - max_len - sum(map(len, df.columns)) + 1,
                no_wrap=not print_full_description,
                overflow='ellipsis',
            )
        )
    table = Table(*columns, show_footer=df.shape[0] > 1)
    for collection_name, row in df.iterrows():
        row = [collection_name, *[str(x) for x in row.to_list()]]
        if show_description:
            row.append(name_to_descriptions[collection_name])
        table.add_row(*row)
    # For some reason the table thinks the terminal is larger than it is
    table = Align(table, align='center', width=console.width - max_len + 1)
    console.print(Align(table, align='center'), soft_wrap=True)


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


def print_duplicates(
    db_collection_name: str,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
):
    """Detects and lists duplicate experiment configurations

    Parameters
    ----------
    db_collection_name : str
        The collection to detect duplicates in
    filter_states : Optional[List[str]], optional
        Optional filter on states, by default None
    batch_id : Optional[int], optional
        Optional filter on batch IDs, by default None
    filter_dict : Optional[Dict], optional
        Optional additional user filters, by default None
    """

    from seml.console import console
    from rich.panel import Panel
    from rich.text import Text

    if len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states)) > 0:
        detect_killed(db_collection_name, print_detected=False)
    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=None
    )
    duplicates = detect_duplicates(db_collection_name, filter_dict)
    num_duplicates = sum(map(len, duplicates))
    sorted_duplicates = sorted(list(map(lambda d: tuple(sorted(d)), duplicates)))
    panel = Panel(
        Text.assemble(
            ('Duplicate experiment ID groups: ', 'bold'),
            (', '.join(map(str, sorted_duplicates))),
        ),
        title=console.render_str(
            f'Found {num_duplicates} duplicate experiment configurations ({len(duplicates)} groups)'
        ),
        highlight=True,
        border_style='red',
    )
    console.print(panel)


def print_output(
    db_collection_name: str,
    sacred_id: Optional[int] = None,
    filter_states: Optional[List[str]] = None,
    batch_id: Optional[int] = None,
    filter_dict: Optional[Dict] = None,
):
    """
    Prints the output of experiments

    Parameters
    ----------
    db_collection_name : str
        The collection to print the output of
    sacred_id : Optional[int], optional
        The ID of the experiment to print the output of, by default None
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    filter_dict : Optional[Dict], optional
        Additional filters, by default None
    """
    from seml.console import console, Heading, pause_live_widget

    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    collection = get_collection(db_collection_name)
    experiments = collection.find(
        filter_dict, {'seml.output_file': 1, '_id': 1, 'batch_id': 1, 'captured_out': 1}
    )
    count = 0
    for exp in experiments:
        count += 1
        console.print(Heading(f'Experiment {exp["_id"]} (batch {exp["batch_id"]})'))
        with pause_live_widget():
            try:
                with open(
                    exp['seml']['output_file'], mode='r', newline='', errors='replace'
                ) as f:
                    for line in f:
                        console.print(line[:-1], end=line[-1])
                    console.print()  # new line
            except IOError:
                logging.info(f"File {exp['seml']['output_file']} could not be read.")
                if 'captured_out' in exp and exp['captured_out']:
                    logging.info('Captured output from DB:')
                    console.print(exp['captured_out'])
                else:
                    logging.error('No output available.')
            except KeyError:
                logging.error(f"Output file not found in experiment {exp['_id']}.")

    if count == 0:
        logging.info('No experiments found.')


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
    job_info_dict = {}
    for line in job_info.split():
        if line:
            key, value = line.split('=', 1)
            job_info_dict[key] = value
    return job_info_dict


def generate_queue_table(
    db,
    job_ids: List[str],
    filter_states: Optional[List[str]],
    filter_by_user: bool = True,
):
    """
    Generates a table of the SEML collections of Slurm jobs.

    Parameters
    ----------
    job_ids : List[str]
        The job IDs to check
    filter_by_user : bool, optional
        Whether to only check jobs by the current user, by default True.

    Returns
    -------
    Align
        The table of the SEML collections of Slurm jobs.
    """
    from seml.console import Table
    from rich.align import Align

    # Run scontrol
    if job_ids is None or len(job_ids) == 0:
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

    # Convert to dictionary
    job_info_strs = list(filter(None, job_info_strs))
    job_infos = list(map(parse_scontrol_job_info, job_info_strs))

    # Find the collections
    all_collections = set(db.list_collection_names())

    collection_to_jobs = defaultdict(list)
    states = set()
    collections = set()
    for job in job_infos:
        state = job['JobState']
        if filter_states is not None and state not in filter_states:
            continue

        user_id = job.get('UserId', '').split('(')[0]
        if filter_by_user and user_id != os.environ['USER']:
            continue

        collection = job.get('Comment', None)
        if not (collection and collection in all_collections):
            collection = 'No collection found'
            if job['JobName'] == 'jupyter':
                collection = 'Jupyter'
                url, known_host = find_jupyter_host(job['StdOut'], False)
                if known_host is not None:
                    collection = f'Jupyter ({url})'

        collection_to_jobs[(collection, state)].append(job)
        states.add(job['JobState'])
        collections.add(collection)

    # Print the collections
    states = sorted(states)
    collections = sorted(collections)
    table = Table(
        'Collection',
        *states,
        show_header=True,
    )

    def format_job(job_info):
        if job_info is None:
            return ''
        nodelist = job_info['NodeList']
        job_id = job_info['JobId']
        task_id = job_info.get('ArrayTaskId', None)
        if task_id:
            job_id = f'{job_id}_{task_id}'
        if nodelist:
            return f"{job_id} ({job_info['RunTime']}, {nodelist})"
        else:
            return f"{job_id} ({job_info.get('Reason', '')})"

    for col in collections:
        row = [col]
        for state in states:
            jobs = collection_to_jobs[(col, state)]
            row.append('\n'.join(map(format_job, jobs)))
        table.add_row(*row)

    return Align(table, align='center')


def print_queue(
    job_ids: Optional[Sequence[str]],
    filter_states: Optional[Sequence[str]],
    filter_by_user: bool,
    watch: bool,
):
    """
    Prints the SEML collections of Slurm jobs.

    Parameters
    ----------
    job_ids : Optional[Sequence[str]], optional
        The job IDs to check, by default None (None meaning all jobs)
    filter_by_user : bool, optional
        Whether to only check jobs by the current user, by default True
    """
    from seml.console import console, pause_live_widget
    from rich.live import Live

    mongodb_config = get_mongodb_config()
    db = get_database(**mongodb_config)

    def generate_table_fn():
        return generate_queue_table(db, job_ids, filter_states, filter_by_user)

    table = generate_table_fn()
    if watch:
        console.clear()
        with pause_live_widget():
            with Live(table, refresh_per_second=0.5) as live:
                while True:
                    time.sleep(2)
                    live.update(generate_table_fn())
    else:
        console.print(table)
