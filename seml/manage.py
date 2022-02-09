import itertools
import logging
import os
import subprocess
import datetime
from getpass import getpass
import copy
import time
import gridfs

from seml.config import check_config
from seml.database import get_collection, build_filter_dict
from seml.sources import delete_orphaned_sources, upload_sources
from seml.utils import s_if, chunker
from seml.settings import SETTINGS
from seml.errors import ArgumentError, MongoDBError

States = SETTINGS.STATES


def report_status(db_collection_name):
    detect_killed(db_collection_name, print_detected=False)
    collection = get_collection(db_collection_name)
    staged = collection.count_documents({'status': {'$in': States.STAGED}})
    pending = collection.count_documents({'status': {'$in': States.PENDING}})
    failed = collection.count_documents({'status': {'$in': States.FAILED}})
    killed = collection.count_documents({'status': {'$in': States.KILLED}})
    interrupted = collection.count_documents({'status': {'$in': States.INTERRUPTED}})
    running = collection.count_documents({'status': {'$in': States.RUNNING}})
    completed = collection.count_documents({'status': {'$in': States.COMPLETED}})
    title = f"********** Report for database collection '{db_collection_name}' **********"
    logging.info(title)
    logging.info(f"*     - {staged:3d} staged experiment{s_if(staged)}")
    logging.info(f"*     - {pending:3d} pending experiment{s_if(pending)}")
    logging.info(f"*     - {running:3d} running experiment{s_if(running)}")
    logging.info(f"*     - {completed:3d} completed experiment{s_if(completed)}")
    logging.info(f"*     - {interrupted:3d} interrupted experiment{s_if(interrupted)}")
    logging.info(f"*     - {failed:3d} failed experiment{s_if(failed)}")
    logging.info(f"*     - {killed:3d} killed experiment{s_if(killed)}")
    logging.info("*" * len(title))


def cancel_experiment_by_id(collection, exp_id, set_interrupted=True, slurm_dict=None, wait=False):
    exp = collection.find_one({'_id': exp_id})
    if slurm_dict:
        exp['slurm'].update(slurm_dict)

    if exp is not None:
        if 'array_id' in exp['slurm']:
            job_str = f"{exp['slurm']['array_id']}_{exp['slurm']['task_id']}"
            filter_dict = {'slurm.array_id': exp['slurm']['array_id'],
                           'slurm.task_id': exp['slurm']['task_id']}
        else:
            logging.error(f"Experiment with ID {exp_id} has not been started using Slurm.")
            return

        try:
            # Check if job exists
            subprocess.run(f"scontrol show jobid -dd {job_str}", shell=True, check=True, stdout=subprocess.DEVNULL)
            if set_interrupted:
                # Set the database state to INTERRUPTED
                collection.update_one({'_id': exp_id}, {'$set': {'status': States.INTERRUPTED[0]}})

            # Check if other experiments are running in the same job
            other_exps_filter = filter_dict.copy()
            other_exps_filter['_id'] = {'$ne': exp_id}
            other_exps_filter['status'] = {'$in': [*States.RUNNING, *States.PENDING]}
            other_exp_running = (collection.count_documents(other_exps_filter) >= 1)

            # Cancel if no other experiments are running in the same job
            if not other_exp_running:
                subprocess.run(f"scancel {job_str}", shell=True, check=True)
                # Wait until the job is actually gone
                if wait:
                    while len(subprocess.run(f"squeue -h -o '%A' -j{job_str}", shell=True, check=True, capture_output=True).stdout) > 0:
                        time.sleep(0.5)
                if set_interrupted:
                    # set state to interrupted again (might have been overwritten by Sacred in the meantime).
                    collection.update_many(filter_dict,
                                           {'$set': {'status': States.INTERRUPTED[0],
                                                     'stop_time': datetime.datetime.utcnow()}})

        except subprocess.CalledProcessError:
            logging.error(f"Slurm job {job_str} of experiment "
                          f"with ID {exp_id} is not pending/running in Slurm.")
    else:
        logging.error(f"No experiment found with ID {exp_id}.")


def cancel_experiments(db_collection_name, sacred_id, filter_states, batch_id, filter_dict, yes, wait=False):
    """
    Cancel experiments.

    Parameters
    ----------
    db_collection_name: str
        Database collection name.
    sacred_id: int or None
        ID of the experiment to cancel. If None, will use the other arguments to cancel possible multiple experiments.
    filter_states: list of strings or None
        List of statuses to filter for. Will cancel all jobs from the database collection
        with one of the given statuses.
    batch_id: int or None
        The ID of the batch of experiments to cancel. All experiments that are staged together (i.e. within the same
        command line call) have the same batch ID.
    filter_dict: dict or None
        Arbitrary filter dictionary to use for cancelling experiments. Any experiments whose database entries match all
        keys/values of the dictionary will be cancelled.

    Returns
    -------
    None

    """
    collection = get_collection(db_collection_name)
    if sacred_id is None:
        # no ID is provided: we check whether there are slurm jobs for which after this action no
        # RUNNING experiment remains. These slurm jobs can be killed altogether.
        # However, it is NOT possible right now to cancel a single experiment in a Slurm job with multiple
        # running experiments.
        try:
            if len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states)) > 0:
                detect_killed(db_collection_name, print_detected=False)

            filter_dict = build_filter_dict(filter_states, batch_id, filter_dict)

            ncancel = collection.count_documents(filter_dict)
            logging.info(f"Cancelling {ncancel} experiment{s_if(ncancel)}.")
            if ncancel >= SETTINGS.CONFIRM_CANCEL_THRESHOLD:
                if not yes and input(f"Are you sure? (y/n) ").lower() != "y":
                    exit()

            filter_dict_new = copy.deepcopy(filter_dict)
            filter_dict_new.update({'slurm.array_id': {'$exists': True}})
            exps = list(collection.find(filter_dict_new,
                                        {'_id': 1, 'status': 1, 'slurm.array_id': 1, 'slurm.task_id': 1}))
            # set of slurm IDs in the database
            slurm_ids = set([(e['slurm']['array_id'], e['slurm']['task_id']) for e in exps])
            # set of experiment IDs to be cancelled.
            exp_ids = set([e['_id'] for e in exps])
            to_cancel = set()

            # iterate over slurm IDs to check which slurm jobs can be cancelled altogether
            for (a_id, t_id) in slurm_ids:
                # find experiments RUNNING under the slurm job
                jobs_running = [e for e in exps
                                if (e['slurm']['array_id'] == a_id and e['slurm']['task_id'] == t_id
                                    and e['status'] in States.RUNNING)]
                running_exp_ids = set(e['_id'] for e in jobs_running)
                if len(running_exp_ids.difference(exp_ids)) == 0:
                    # there are no running jobs in this slurm job that should not be canceled.
                    to_cancel.add(f"{a_id}_{t_id}")

            # cancel all Slurm jobs for which no running experiment remains.
            if len(to_cancel) > 0:
                chunk_size = 100
                chunks = list(chunker(list(to_cancel), chunk_size))
                [subprocess.run(f"scancel {' '.join(chunk)}", shell=True, check=True) for chunk in chunks]
                # Wait until all jobs are actually stopped.
                if wait:
                    for chunk in chunks:
                        while len(subprocess.run(f"squeue -h -o '%A' --jobs={','.join(chunk)}", 
                                                 shell=True, 
                                                 check=True, 
                                                 capture_output=True).stdout) > 0:
                            time.sleep(0.5)

            # update database status and write the stop_time
            collection.update_many(filter_dict, {'$set': {"status": States.INTERRUPTED[0],
                                                          "stop_time": datetime.datetime.utcnow()}})
        except subprocess.CalledProcessError:
            logging.warning(f"One or multiple Slurm jobs were no longer running when I tried to cancel them.")
    else:
        logging.info(f"Cancelling experiment with ID {sacred_id}.")
        if SETTINGS.CONFIRM_CANCEL_THRESHOLD <= 1:
            if not yes and input('Are you sure? (y/n)').lower() != 'y':
                exit()
        cancel_experiment_by_id(collection, sacred_id, wait=wait)


def delete_experiments(db_collection_name, sacred_id, filter_states, batch_id, filter_dict, yes=False):
    collection = get_collection(db_collection_name)
    if sacred_id is None:
        if len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states)) > 0:
            detect_killed(db_collection_name, print_detected=False)

        filter_dict = build_filter_dict(filter_states, batch_id, filter_dict)
        ndelete = collection.count_documents(filter_dict)
        batch_ids = collection.find(filter_dict, {'batch_id'})
        batch_ids_in_del = set([x['batch_id'] for x in batch_ids])

        logging.info(f"Deleting {ndelete} configuration{s_if(ndelete)} from database collection.")
        if ndelete >= SETTINGS.CONFIRM_DELETE_THRESHOLD:
            if not yes and input(f"Are you sure? (y/n) ").lower() != "y":
                exit()
        collection.delete_many(filter_dict)
    else:
        exp = collection.find_one({'_id': sacred_id})
        if exp is None:
            raise MongoDBError(f"No experiment found with ID {sacred_id}.")
        else:
            logging.info(f"Deleting experiment with ID {sacred_id}.")
            if SETTINGS.CONFIRM_DELETE_THRESHOLD <= 1:
                if not yes and input('Are you sure? (y/n)').lower() != 'y':
                    exit()
            batch_ids_in_del = set([exp['batch_id']])
            collection.delete_one({'_id': sacred_id})

    if len(batch_ids_in_del) > 0:
        # clean up the uploaded sources if no experiments of a batch remain
        delete_orphaned_sources(collection, batch_ids_in_del)


def reset_slurm_dict(exp):
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


def reset_single_experiment(collection, exp):
    exp['status'] = States.STAGED[0]
    # queue_time for backward compatibility.
    keep_entries = ['batch_id', 'status', 'seml', 'slurm', 'config', 'config_hash', 'add_time', 'queue_time', 'git']

    # Clean up SEML dictionary
    keep_seml = set(['source_files', 'working_dir'])
    keep_seml.update(SETTINGS.VALID_SEML_CONFIG_VALUES)
    seml_keys = set(exp['seml'].keys())
    for key in seml_keys - keep_seml:
        del exp['seml'][key]

    reset_slurm_dict(exp)

    collection.replace_one({'_id': exp['_id']}, {entry: exp[entry] for entry in keep_entries if entry in exp},
                           upsert=False)


def reset_experiments(db_collection_name, sacred_id, filter_states, batch_id, filter_dict, yes=False):
    collection = get_collection(db_collection_name)

    if sacred_id is None:
        if len({*States.PENDING, *States.RUNNING, *States.KILLED} & set(filter_states)) > 0:
            detect_killed(db_collection_name, print_detected=False)

        if isinstance(filter_states, str):
            filter_states = [filter_states]

        filter_dict = build_filter_dict(filter_states, batch_id, filter_dict)

        nreset = collection.count_documents(filter_dict)
        exps = collection.find(filter_dict)

        logging.info(f"Resetting the state of {nreset} experiment{s_if(nreset)}.")
        if nreset >= SETTINGS.CONFIRM_RESET_THRESHOLD:
            if not yes and input(f"Are you sure? (y/n) ").lower() != "y":
                exit()
        for exp in exps:
            reset_single_experiment(collection, exp)
    else:
        exp = collection.find_one({'_id': sacred_id})
        if exp is None:
            raise MongoDBError(f"No experiment found with ID {sacred_id}.")
        else:
            logging.info(f"Resetting the state of experiment with ID {sacred_id}.")
            if SETTINGS.CONFIRM_RESET_THRESHOLD <= 1:
                if not yes and input('Are you sure? (y/n)').lower() != 'y':
                    exit()
            reset_single_experiment(collection, exp)


def detect_killed(db_collection_name, print_detected=True):
    collection = get_collection(db_collection_name)
    exps = collection.find({'status': {'$in': [*States.PENDING, *States.RUNNING]},
                            '$or': [{'slurm.array_id': {'$exists': True}}, {'slurm.id': {'$exists': True}}]})
    running_jobs = get_slurm_arrays_tasks()
    nkilled = 0
    for exp in exps:
        exp_running = ('array_id' in exp['slurm'] and exp['slurm']['array_id'] in running_jobs
                       and (any(exp['slurm']['task_id'] in r for r in running_jobs[exp['slurm']['array_id']][0])
                            or exp['slurm']['task_id'] in running_jobs[exp['slurm']['array_id']][1]))
        if not exp_running:
            if 'stop_time' in exp:
                collection.update_one({'_id': exp['_id']}, {'$set': {'status': States.INTERRUPTED[0]}})
            else:
                nkilled += 1
                collection.update_one({'_id': exp['_id']}, {'$set': {'status': States.KILLED[0]}})
                try:
                    with open(exp['seml']['output_file'], 'r') as f:
                        all_lines = f.readlines()
                    collection.update_one({'_id': exp['_id']}, {'$set': {'fail_trace': all_lines[-4:]}})
                except IOError:
                    # If the experiment is cancelled before starting (e.g. when still queued), there is not output file.
                    logging.verbose(f"File {exp['seml']['output_file']} could not be read.")
    if print_detected:
        logging.info(f"Detected {nkilled} externally killed experiment{s_if(nkilled)}.")


def get_slurm_arrays_tasks(filter_by_user=False):
    """Get a dictionary of running/pending Slurm job arrays (as keys) and tasks (as values)

    job_dict
    -------
    job_dict: dict
        This dictionary has the job array IDs as keys and the values are
        a list of 1) the pending job task range and 2) a list of running job task IDs.

    """
    try:
        squeue_cmd = f"SLURM_BITSTR_LEN=256 squeue -a -t {','.join(SETTINGS.SLURM_STATES.ACTIVE)} -h -o %i"
        if filter_by_user:
            squeue_cmd += " -u `whoami`"
        squeue_out = subprocess.run(
                squeue_cmd,
                shell=True, check=True, capture_output=True).stdout
        jobs = [job_str for job_str in squeue_out.splitlines() if b'_' in job_str]
        if len(jobs) > 0:
            array_ids_str, task_ids = zip(*[job_str.split(b'_') for job_str in jobs])
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


def get_nonempty_input(field_name, num_trials=3):
    get_input = getpass if "password" in field_name else input
    field = get_input(f"Please input the {field_name}: ")
    trials = 1
    while (field is None or len(field) == 0) and trials < num_trials:
        logging.error(f'{field_name} was empty.')
        field = get_input(f"Please input the {field_name}: ")
        trials += 1
    if field is None or len(field) == 0:
        raise ArgumentError(f"Did not receive an input for {num_trials} times. Aborting.")
    return field


def mongodb_credentials_prompt():
    logging.info('Configuring SEML. Warning: Password will be stored in plain text.')
    host = get_nonempty_input("MongoDB host")
    port = input('Port (default: 27017):')
    port = "27017" if port == "" else port
    database = get_nonempty_input("database name")
    username = get_nonempty_input("user name")
    password = get_nonempty_input("password")
    file_path = SETTINGS.DATABASE.MONGODB_CONFIG_PATH
    config_string = (f'username: {username}\n'
                     f'password: {password}\n'
                     f'port: {port}\n'
                     f'database: {database}\n'
                     f'host: {host}')
    logging.info(f"Saving the following configuration to {file_path}:\n"
                 f"{config_string.replace(f'password: {password}', 'password: ********')}"
                 )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(config_string)


def reload_sources(db_collection_name, batch_ids=None, keep_old=False, yes=False):
    collection = get_collection(db_collection_name)
    
    if batch_ids is not None:
        filter_dict = {'batch_id': {'$in': list(batch_ids)}}
    else:
        filter_dict = {}
    db_results = list(collection.find(filter_dict, {'batch_id', 'seml', 'config', 'status'}))
    id_to_config = {
        bid: (next(iter(configs))['seml'], [x['config'] for x in configs])
        for bid, configs in
        itertools.groupby(db_results, lambda x: x['batch_id'])
    }
    states = {x['status'] for x in db_results}

    if any([s in (States.RUNNING + States.PENDING + States.COMPLETED) for s in states]):
        logging.info(f'Some of the experiments is still in RUNNING, PENDING or COMPLETED.')
        if not yes and input(f'Are you sure you want to continue? (y/n)').lower() != 'y':
            exit()

    for batch_id, (seml_config, configs) in id_to_config.items():
        if 'working_dir' not in seml_config or not seml_config['working_dir']:
            logging.error(f'Batch {batch_id}: No source files to refresh.')
            continue
        # Cache the old working directory and move to the specified one
        cwd = os.getcwd()
        os.chdir(seml_config['working_dir'])

        # Check whether the configurations aligns with the current source code
        check_config(seml_config['executable'], seml_config['conda_environment'], configs)

        # Find the currently used source files
        db = collection.database
        fs = gridfs.GridFS(db)
        fs_filter_dict = {
            'metadata.batch_id': batch_id,
            'metadata.collection_name': f'{collection.name}',
            'metadata.deprecated': {'$exists': False}
        }
        current_source_files = db['fs.files'].find(filter_dict, '_id')
        current_ids = [x['_id'] for x in current_source_files]
        fs_filter_dict = {
            '_id': {'$in': current_ids}
        }
        # Deprecate them
        db['fs.files'].update_many(fs_filter_dict, {'$set': {'metadata.deprecated': True}})
        try:
            # Try to upload the new ones
            source_files = upload_sources(seml_config, collection, batch_id)
        except Exception as e:
            # If it fails we reconstruct the old ones
            logging.error(f"Batch {batch_id}: Source import failed. Restoring old files.")
            db['fs.files'].update_many(fs_filter_dict, {'$unset': {'metadata.deprecated': ""}})
            raise e
        try:
            # Try to assign the new ones to the experiments
            filter_dict = {
                'batch_id': batch_id
            }
            collection.update_many(filter_dict, {
                '$set': {
                    'seml.source_files': source_files
                }
            })
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
                'metadata.deprecated': True
            }
            source_files = [x['_id'] for x in db['fs.files'].find(fs_filter_dict, {'_id'})]
            for to_delete in source_files:
                fs.delete(to_delete)
        # Move to the old working directory
        os.chdir(cwd)
