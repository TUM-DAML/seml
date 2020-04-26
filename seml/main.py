import sys
import argparse
import subprocess
import warnings
import datetime
import json
import logging

from seml.misc import get_slurm_arrays_tasks, s_if, chunker, LoggingFormatter
from seml import database_utils as db_utils
from seml.queue_experiments import queue_experiments
from seml.start_experiments import start_experiments

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


def report_status(config_file):
    detect_killed(config_file, print_detected=False)
    collection = db_utils.get_collection_from_config(config_file)
    queued = collection.count_documents({'status': 'QUEUED'})
    pending = collection.count_documents({'status': 'PENDING'})
    failed = collection.count_documents({'status': 'FAILED'})
    killed = collection.count_documents({'status': 'KILLED'})
    interrupted = collection.count_documents({'status': 'INTERRUPTED'})
    running = collection.count_documents({'status': 'RUNNING'})
    completed = collection.count_documents({'status': 'COMPLETED'})
    title = f"********** Report for database collection '{collection.name}' **********"
    logging.info(title)
    logging.info(f"*     - {queued:3d} queued experiment{s_if(queued)}")
    logging.info(f"*     - {pending:3d} pending experiment{s_if(pending)}")
    logging.info(f"*     - {running:3d} running experiment{s_if(running)}")
    logging.info(f"*     - {completed:3d} completed experiment{s_if(completed)}")
    logging.info(f"*     - {interrupted:3d} interrupted experiment{s_if(interrupted)}")
    logging.info(f"*     - {failed:3d} failed experiment{s_if(failed)}")
    logging.info(f"*     - {killed:3d} killed experiment{s_if(killed)}")
    logging.info("*" * len(title))


def cancel_experiment_by_id(collection, exp_id):
    exp = collection.find_one({'_id': exp_id})
    if exp is not None:
        job_str = f"{exp['slurm']['array_id']}_{exp['slurm']['task_id']}"
        try:
            # Check if job exists
            subprocess.check_output(f"scontrol show jobid -dd {job_str}", shell=True)
            # Set the database state to INTERRUPTED
            collection.update_one({'_id': exp_id}, {'$set': {'status': 'INTERRUPTED'}})

            # Check if other experiments are running in the same job
            other_exps = collection.find({'slurm.array_id': exp['slurm']['array_id'],
                                          'slurm.task_id': exp['slurm']['task_id']})
            other_exp_running = False
            for e in other_exps:
                if e['status'] in ["RUNNING", "PENDING"]:
                    other_exp_running = True

            # Cancel if no other experiments are running in the same job
            if not other_exp_running:
                subprocess.check_output(f"scancel {job_str}", shell=True)
                # set state to interrupted again (might have been overwritten by Sacred in the meantime).
                collection.update_many({'slurm.array_id': exp['slurm']['array_id'],
                                        'slurm.task_id': exp['slurm']['task_id']},
                                       {'$set': {'status': 'INTERRUPTED',
                                                 'stop_time': datetime.datetime.utcnow()}})

        except subprocess.CalledProcessError:
            warnings.warn(f"Slurm job {job_str} of experiment "
                          f"with ID {exp_id} is not pending/running in Slurm.")
    else:
        raise LookupError(f"No experiment found with ID {exp_id}.")


def cancel_experiments(config_file, sacred_id, filter_states, batch_id, filter_dict):
    """
    Cancel experiments.

    Parameters
    ----------
    config_file: str
        Path to the configuration YAML file.
    sacred_id: int or None
        ID of the experiment to cancel. If None, will use the other arguments to cancel possible multiple experiments.
    filter_states: list of strings or None
        List of statuses to filter for. Will cancel all jobs from the database collection
        with one of the given statuses.
    batch_id: int or None
        The ID of the batch of experiments to cancel. All experiments that are queued together (i.e. within the same
        command line call) have the same batch ID.
    filter_dict: dict or None
        Arbitrary filter dictionary to use for cancelling experiments. Any experiments whose database entries match all
        keys/values of the dictionary will be cancelled.

    Returns
    -------
    None

    """
    collection = db_utils.get_collection_from_config(config_file)
    if sacred_id is None:
        # no ID is provided: we check whether there are slurm jobs for which after this action no
        # RUNNING experiment remains. These slurm jobs can be killed altogether.
        # However, it is NOT possible right now to cancel a single experiment in a Slurm job with multiple
        # running experiments.
        try:
            if len({'PENDING', 'RUNNING', 'KILLED'} & set(filter_states)) > 0:
                detect_killed(config_file, print_detected=False)

            filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)

            ncancel = collection.count_documents(filter_dict)
            if ncancel >= 10:
                if input(f"Cancelling {ncancel} experiment{s_if(ncancel)}. "
                         f"Are you sure? (y/n) ").lower() != "y":
                    exit()
            else:
                logging.info(f"Cancelling {ncancel} experiment{s_if(ncancel)}.")

            exps = list(collection.find(filter_dict, {'_id': 1, 'status': 1, 'slurm.array_id': 1, 'slurm.task_id': 1}))
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
                                    and e['status'] in ['RUNNING'])]
                running_exp_ids = set(e['_id'] for e in jobs_running)
                if len(running_exp_ids.difference(exp_ids)) == 0:
                    # there are no running jobs in this slurm job that should not be canceled.
                    to_cancel.add(f"{a_id}_{t_id}")

            # cancel all Slurm jobs for which no running experiment remains.
            if len(to_cancel) > 0:
                chunk_size = 100
                chunks = chunker(list(to_cancel), chunk_size)
                [subprocess.check_output(f"scancel {' '.join(chunk)}", shell=True) for chunk in chunks]

            # update database status and write the stop_time
            collection.update_many(filter_dict, {'$set': {"status": "INTERRUPTED",
                                                          "stop_time": datetime.datetime.utcnow()}})
        except subprocess.CalledProcessError:
            warnings.warn(f"One or multiple Slurm jobs were no longer running when I tried to cancel them.")
    else:
        logging.info(f"Cancelling experiment with ID {sacred_id}.")
        cancel_experiment_by_id(collection, sacred_id)


def delete_experiments(config_file, sacred_id, filter_states, batch_id, filter_dict):
    collection = db_utils.get_collection_from_config(config_file)
    if sacred_id is None:
        if len({'PENDING', 'RUNNING', 'KILLED'} & set(filter_states)) > 0:
            detect_killed(config_file, print_detected=False)

        filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)
        ndelete = collection.count_documents(filter_dict)
        batch_ids = collection.find(filter_dict, {'batch_id'})
        batch_ids_in_del = set([x['batch_id'] for x in batch_ids])

        if ndelete >= 10:
            if input(f"Deleting {ndelete} configuration{s_if(ndelete)} from database collection. "
                     f"Are you sure? (y/n) ").lower() != "y":
                exit()
        else:
            logging.info(f"Deleting {ndelete} configuration{s_if(ndelete)} from database collection.")
        collection.delete_many(filter_dict)
    else:
        exp = collection.find_one({'_id': sacred_id})
        if exp is None:
            raise LookupError(f"No experiment found with ID {sacred_id}.")
        else:
            logging.info(f"Deleting experiment with ID {sacred_id}.")
            batch_ids_in_del = set(exp['batch_id'])
            collection.delete_one({'_id': sacred_id})

    if len(batch_ids_in_del) > 0:
        # clean up the uploaded sources if no experiments of a batch remain
        db_utils.check_empty_batches(collection, batch_ids_in_del)


def reset_experiment(collection, exp):
    exp['status'] = 'QUEUED'
    keep_entries = ['batch_id', 'status', 'seml', 'slurm', 'config', 'config_hash', 'queue_time', 'git']

    # Clean up SEML dictionary
    keep_seml = {'db_collection', 'executable', 'conda_environment', 'output_dir', 'source_files', 'project_root_dir',
                 'executable_relative'}
    seml_keys = set(exp['seml'].keys())
    for key in seml_keys - keep_seml:
        del exp['seml'][key]

    # Clean up Slurm dictionary
    keep_slurm = {'name', 'output_dir', 'experiments_per_job', 'sbatch_options'}
    slurm_keys = set(exp['slurm'].keys())
    for key in slurm_keys - keep_slurm:
        del exp['slurm'][key]

    # Clean up sbatch_options dictionary
    remove_sbatch = {'job-name', 'output'}
    sbatch_keys = set(exp['slurm']['sbatch_options'].keys())
    for key in remove_sbatch & sbatch_keys:
        del exp['slurm']['sbatch_options'][key]

    collection.replace_one({'_id': exp['_id']}, {entry: exp[entry] for entry in keep_entries}, upsert=False)


def reset_states(config_file, sacred_id, filter_states, batch_id, filter_dict):
    collection = db_utils.get_collection_from_config(config_file)

    if sacred_id is None:
        if len({'PENDING', 'RUNNING', 'KILLED'} & set(filter_states)) > 0:
            detect_killed(config_file, print_detected=False)

        filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)

        nreset = collection.count_documents(filter_dict)
        exps = collection.find(filter_dict)

        if nreset >= 10:
            if input(f"Resetting the state of {nreset} experiment{s_if(nreset)}. "
                     f"Are you sure? (y/n) ").lower() != "y":
                exit()
        else:
            logging.info(f"Resetting the state of {nreset} experiment{s_if(nreset)}.")
        for exp in exps:
            reset_experiment(collection, exp)
    else:
        exp = collection.find_one({'_id': sacred_id})
        if exp is None:
            raise LookupError(f"No experiment found with ID {sacred_id}.")
        else:
            logging.info(f"Resetting the state of experiment with ID {sacred_id}.")
            reset_experiment(collection, exp)


def detect_killed(config_file, print_detected=True):
    collection = db_utils.get_collection_from_config(config_file)
    exps = collection.find({'status': {'$in': ['PENDING', 'RUNNING']}, 'slurm.array_id': {'$exists': True}})
    running_jobs = get_slurm_arrays_tasks()
    nkilled = 0
    for exp in exps:
        exp_running = (exp['slurm']['array_id'] in running_jobs
                       and (exp['slurm']['task_id'] in running_jobs[exp['slurm']['array_id']][0]
                            or exp['slurm']['task_id'] in running_jobs[exp['slurm']['array_id']][1]))
        if not exp_running:
            if 'stop_time' in exp:
                collection.update_one({'_id': exp['_id']}, {'$set': {'status': 'INTERRUPTED'}})
            else:
                nkilled += 1
                collection.update_one({'_id': exp['_id']}, {'$set': {'status': 'KILLED'}})
                try:
                    seml_config = exp['seml']
                    slurm_config = exp['slurm']
                    if 'output_file' in seml_config:
                        output_file = seml_config['output_file']
                    elif 'output_file' in slurm_config:     # backward compatibility, we used to store the path in 'slurm'
                        output_file = slurm_config['output_file']
                    else:
                        continue
                    with open(output_file, 'r') as f:
                        all_lines = f.readlines()
                    collection.update_one({'_id': exp['_id']}, {'$set': {'fail_trace': all_lines[-4:]}})
                except IOError:
                    if 'output_file' in seml_config:
                        output_file = seml_config['output_file']
                    elif 'output_file' in slurm_config:     # backward compatibility
                        output_file = slurm_config['output_file']
                    logging.warning(f"File {output_file} could not be read.")
    if print_detected:
        logging.info(f"Detected {nkilled} externally killed experiment{s_if(nkilled)}.")


def clean_unreferenced_artifacts(config_file, all_collections=False):
    """
    Delete orphaned artifacts from the database. That is, artifacts that were generated by experiments, but whose
    experiment's database entry has been removed. This leads to storage accumulation, and this function cleans this
    excess storage.
    Parameters
    ----------
    config_file: str
        config file containing the collection to be scanned.
    all_collections: bool
        If yes, will scan ALL collections (not just the one provided in the config file) for orphaned artifacts.

    Returns
    -------
    None
    """
    import gridfs
    if all_collections:
        config = db_utils.get_mongodb_config()
        db = db_utils.get_database(**config)
        collection_names = db.list_collection_names()
    else:
        collection = db_utils.get_collection_from_config(config_file)
        db = collection.database
        collection_names = [collection.name]
    collection_names = set(collection_names)
    collection_blacklist = {'fs.chunks', 'fs.files'}
    collection_names = collection_names - collection_blacklist

    fs = gridfs.GridFS(db)
    referenced_files = set()
    for collection_name in tqdm(collection_names):
        collection = db[collection_name]
        experiments = list(collection.find({}, {'artifacts': 1, 'experiment.sources': 1, 'source_files': 1}))
        for exp in experiments:
            if 'artifacts' in exp:
                referenced_files.update({x[1] for x in exp['artifacts']})
            if 'experiment' in exp and 'sources' in exp['experiment']:
                referenced_files.update({x[1] for x in exp['experiment']['sources']})
            if 'source_files' in exp:
                referenced_files.update({x[1] for x in exp['source_files']})

    all_files_in_db = list(db['fs.files'].find({}, {'_id': 1, 'filename': 1, 'metadata': 1}))
    filtered_file_ids = set()
    for file in all_files_in_db:
        if 'filename' in file:
            filename = file['filename']
            file_collection = None
            if filename.startswith("file://") and 'metadata' in file and file['metadata']:
                # seml-uploaded source
                metadata = file['metadata']
                file_collection = metadata['collection_name'] if 'collection_name' in metadata else None
            elif filename.startswith("artifact://"):
                # artifact uploaded by Sacred
                filename = filename[11:]
                file_collection = filename.split("/")[0]
            if file_collection is not None and file_collection in collection_names:
                # only delete files corresponding to collections we want to clean
                filtered_file_ids.add(file['_id'])

    not_referenced_artifacts = filtered_file_ids - referenced_files
    n_delete = len(not_referenced_artifacts)
    if n_delete == 0:
        logging.info("No unreferenced artifacts found.")
        return

    if input(f"Deleting {n_delete} not referenced artifact{s_if(n_delete)} from database {db.name}. "
             f"Are you sure? (y/n) ").lower() != "y":
        exit()
    logging.info('Deleting not referenced artifacts...')
    for to_delete in tqdm(not_referenced_artifacts):
        fs.delete(to_delete)
    logging.info(f'Successfully deleted {n_delete} not referenced artifact{s_if(n_delete)}.')


def main():
    parser = argparse.ArgumentParser(
            description="Manage experiments for the given configuration. "
                        "Each experiment is represented as a record in the database. "
                        "See examples/README.md for more details.",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            'config_file', type=str,
            help="Path to the YAML configuration file for the experiment.")
    parser.add_argument(
            '--verbose', '-v', action='store_true',
            help='Display more log messages.')
    subparsers = parser.add_subparsers(title="Possible operations")

    parser_queue = subparsers.add_parser(
            "queue",
            help="Queue the experiments as defined in the configuration.")
    parser_queue.add_argument(
            '-nh', '--no-hash', action='store_true',
            help="Do not use the hash of the config dictionary to filter out duplicates (by comparing all"
                 "dictionary values individually). This is much  slower, so use only if you have a good reason not to"
                 " use the hash.")
    parser_queue.add_argument(
            '-nc', '--no-config-check', action='store_true',
            help="Do not check the config for missing/unused arguments. "
                 "Use this if the check fails unexpectedly when using advanced Sacred features or to accelerate queueing.")
    parser_queue.add_argument(
            '-f', '--force-duplicates', action='store_true',
            help="Add experiments to the database even when experiments with identical configurations "
                 "are already in the database.")
    parser_queue.set_defaults(func=queue_experiments)

    parser_start = subparsers.add_parser(
            "start",
            help="Fetch queued experiments from the database and run them (by default via Slurm).")
    parser_start.add_argument(
            '-l', '--local', action='store_true',
            help="Run the experiments locally.")
    parser_start.add_argument(
            '-n', '--num-exps', type=int, default=-1,
            help="Only start the specified number of experiments.")
    parser_start.add_argument(
            '-u', '--unobserved', action='store_true',
            help="Run the experiments without Sacred observers (no changes to MongoDB). "
                 "This also disables output capturing by Sacred, facilitating the use of debuggers (pdb, ipdb).")
    parser_start.add_argument(
            '-pm', '--post-mortem', action='store_true',
            help="Activate post-mortem debugging with pdb.")
    parser_start.add_argument(
            '-d', '--debug', action='store_true',
            help="Run a single experiment locally without Sacred observers and with post-mortem debugging. "
                 "This is equivalent to `--verbose --local --num-exps 1 --unobserved --post-mortem --output-to-console`.")
    parser_start.add_argument(
            '-dr', '--dry-run', action='store_true',
            help="Only show the associated commands instead of running the experiments.")
    parser_start.add_argument(
            '-id', '--sacred-id', type=int,
            help="Sacred ID (_id in the database collection) of the experiment to cancel.")
    parser_start.add_argument(
            '-b', '--batch-id', type=int,
            help="Batch ID (batch_id in the database collection) of the experiments to be cancelled. Experiments that were "
                 "queued together have the same batch_id."
    )
    parser_start.add_argument(
        '-f', '--filter-dict', type=json.loads,
        help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter the experiments by."
    )
    parser_start.add_argument(
        '-o', '--output-to-console', action='store_true',
        help="Print output to console instead of writing it to a log file. Only possible if experiment is run locally."
    )
    parser_start.set_defaults(func=start_experiments)

    parser_status = subparsers.add_parser(
            "status",
            help="Report status of experiments in the database collection.")
    parser_status.set_defaults(func=report_status)

    parser_cancel = subparsers.add_parser(
            "cancel",
            help="Cancel the Slurm job/job step corresponding to experiments, filtered by ID or state.")
    parser_cancel.add_argument(
            '-id', '--sacred-id', type=int,
            help="Sacred ID (_id in the database collection) of the experiment to cancel.")
    parser_cancel.add_argument(
            '-s', '--filter-states', type=str, nargs='*', default=['PENDING', 'RUNNING'],
            help="List of states to filter experiments by. Cancels all experiments if an empty list is passed. "
                 "Default: Cancel all pending and running experiments.")
    parser_cancel.add_argument(
            '-b', '--batch-id', type=int,
            help="Batch ID (batch_id in the database collection) of the experiments to be cancelled. Experiments that were "
                 "queued together have the same batch_id."
    )
    parser_cancel.add_argument(
            '-f', '--filter-dict', type=json.loads,
            help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter the experiments by."
    )
    parser_cancel.set_defaults(func=cancel_experiments)

    parser_delete = subparsers.add_parser(
            "delete",
            help="Delete experiments by ID or state (does not cancel Slurm jobs).")
    parser_delete.add_argument(
            '-id', '--sacred-id', type=int,
            help="Sacred ID (_id in the database collection) of the experiment to delete.")
    parser_delete.add_argument(
            '-s', '--filter-states', type=str, nargs='*', default=['QUEUED', 'FAILED', 'KILLED', 'INTERRUPTED'],
            help="List of states to filter experiments by. Deletes all experiments if an empty list is passed. "
                 "Default: Delete all queued, failed, killed and interrupted experiments.")
    parser_delete.add_argument(
            '-b', '--batch-id', type=int,
            help="Batch ID (batch_id in the database collection) of the experiments to be deleted. Experiments that were "
                 "queued together have the same batch_id."
    )
    parser_delete.add_argument(
            '-f', '--filter-dict', type=json.loads,
            help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter the experiments by."
    )
    parser_delete.set_defaults(func=delete_experiments)

    parser_reset = subparsers.add_parser(
            "reset",
            help="Reset the state of experiments (set to QUEUED and clean database entry) "
                 "by ID or state (does not cancel Slurm jobs).")
    parser_reset.add_argument(
            '-id', '--sacred-id', type=int,
            help="Sacred ID (_id in the database collection) of the experiment to reset.")
    parser_reset.add_argument(
            '-s', '--filter-states', type=str, nargs='*', default=['FAILED', 'KILLED', 'INTERRUPTED'],
            help="List of states to filter experiments by. "
                 "Resets all experiments if an empty list is passed. "
                 "Default: Reset failed, killed and interrupted experiments.")
    parser_reset.add_argument(
            '-f', '--filter-dict', type=json.loads,
            help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter the experiments by."
    )
    parser_reset.add_argument(
            '-b', '--batch-id', type=int,
            help="Batch ID (batch_id in the database collection) of the experiments to be deleted. Experiments that were "
                 "queued together have the same batch_id."
    )
    parser_reset.set_defaults(func=reset_states)

    parser_detect = subparsers.add_parser(
            "detect-killed",
            help="Detect experiments where the corresponding Slurm jobs were killed externally.")
    parser_detect.set_defaults(func=detect_killed)

    parser_clean_db = subparsers.add_parser(
        "clean-db",
        help="Remove orphaned artifacts in the DB from runs which have been deleted.")
    parser_clean_db.add_argument(
        '-a', '--all-collections', action='store_true',
        help="Scan all collections for orphaned artifacts (not just the one provided in the config).")
    parser_clean_db.set_defaults(func=clean_unreferenced_artifacts)

    args = parser.parse_args()

    # Initialize logging
    if args.verbose:
        logging_level = logging.VERBOSE
    else:
        logging_level = logging.INFO
    hdlr = logging.StreamHandler(sys.stderr)
    hdlr.setFormatter(LoggingFormatter())
    logging.root.addHandler(hdlr)
    logging.root.setLevel(logging_level)

    f = args.func
    del args.func
    del args.verbose
    if 'filter_states' in args:
        args.filter_states = [state.upper() for state in args.filter_states]
    f(**args.__dict__)

if __name__ == "__main__":
    main()
