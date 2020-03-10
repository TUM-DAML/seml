import argparse
import subprocess
import warnings
import datetime
import json

from seml.misc import get_slurm_jobs, s_if, chunker
from seml import database_utils as db_utils
from seml.queue_experiments import queue_experiments
from seml.start_experiments import start_experiments

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


def report_status(config_file):
    detect_killed(config_file, verbose=False)
    collection = db_utils.get_collection_from_config(config_file)
    queued = collection.count_documents({'status': 'QUEUED'})
    pending = collection.count_documents({'status': 'PENDING'})
    failed = collection.count_documents({'status': 'FAILED'})
    killed = collection.count_documents({'status': 'KILLED'})
    interrupted = collection.count_documents({'status': 'INTERRUPTED'})
    running = collection.count_documents({'status': 'RUNNING'})
    completed = collection.count_documents({'status': 'COMPLETED'})
    title = "********** Experiment database collection report **********"
    print(title)
    print(f"*     - {queued:3d} queued experiment{s_if(queued)}")
    print(f"*     - {pending:3d} pending experiment{s_if(pending)}")
    print(f"*     - {running:3d} running experiment{s_if(running)}")
    print(f"*     - {completed:3d} completed experiment{s_if(completed)}")
    print(f"*     - {interrupted:3d} interrupted experiment{s_if(interrupted)}")
    print(f"*     - {failed:3d} failed experiment{s_if(failed)}")
    print(f"*     - {killed:3d} killed experiment{s_if(killed)}")
    print("*" * len(title))


def cancel_experiment_by_id(collection, exp_id):
    exp = collection.find_one({'_id': exp_id})
    if exp is not None:
        try:
            # Check if job exists
            subprocess.check_output(f"scontrol show jobid -dd {exp['slurm']['id']}", shell=True)
            # Set the database state to INTERRUPTED
            collection.update_one({'_id': exp_id}, {'$set': {'status': 'INTERRUPTED'}})

            other_exps = collection.find({'slurm.id': exp['slurm']['id']})
            any_exp_running = False
            for e in other_exps:
                if e['status'] in ["RUNNING", "COMPLETED"]:
                    any_exp_running = True

            if not any_exp_running:
                subprocess.check_output(f"scancel {exp['slurm']['id']}", shell=True)
                # set state to interrupted again (might have been overwritten by Sacred in the meantime).
                collection.update_many({'slurm.id': exp['slurm']['id']}, {'$set': {'status': 'INTERRUPTED'}})

        except subprocess.CalledProcessError:
            warnings.warn(f"Slurm job {exp['slurm']['id']} of experiment "
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
                detect_killed(config_file, verbose=False)

            filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)

            ncancel = collection.count_documents(filter_dict)
            if ncancel >= 10:
                if input(f"Cancelling {ncancel} experiment{s_if(ncancel)}. "
                         f"Are you sure? (y/n) ").lower() != "y":
                    exit()
            else:
                print(f"Cancelling {ncancel} experiment{s_if(ncancel)}.")

            exps = list(collection.find(filter_dict, {'slurm.id': 1, '_id': 1, 'status': 1}))
            # set of slurm IDs in the database
            slurm_ids = set([e['slurm']['id'] for e in exps if "slurm" in e and 'id' in e['slurm']])
            # set of experiment IDs to be cancelled.
            exp_ids = set([e['_id'] for e in exps])
            to_cancel = set()

            # iterate over slurm IDs to check which slurm jobs can be cancelled altogether
            for s_id in tqdm(slurm_ids):
                # find experiments RUNNING under the slurm job
                jobs_running = [x for x in exps if x['slurm']['id'] == s_id and x['status'] in ['RUNNING']]
                # jobs_running = list(collection.find({'slurm.id': s_id,
                #                                      'status'  : {"$in": ["RUNNING"]}},
                #                                     {"_id": 1}))
                running_exp_ids = set(e['_id'] for e in jobs_running)
                if len(running_exp_ids.difference(exp_ids)) == 0:
                    # there are no running jobs in this slurm job that should not be canceled.
                    to_cancel.add(str(s_id))

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
        print(f"Cancelling experiment with ID {sacred_id}.")
        cancel_experiment_by_id(collection, sacred_id)


def delete_experiments(config_file, sacred_id, filter_states, batch_id, filter_dict):
    collection = db_utils.get_collection_from_config(config_file)
    if sacred_id is None:
        if len({'PENDING', 'RUNNING', 'KILLED'} & set(filter_states)) > 0:
            detect_killed(config_file, verbose=False)

        filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)
        ndelete = collection.count_documents(filter_dict)

        if ndelete >= 10:
            if input(f"Deleting {ndelete} configuration{s_if(ndelete)} from database collection. "
                     f"Are you sure? (y/n) ").lower() != "y":
                exit()
        else:
            print(f"Deleting {ndelete} configuration{s_if(ndelete)} from database collection.")
        collection.delete_many(filter_dict)
    else:
        if collection.find_one({'_id': sacred_id}) is None:
            raise LookupError(f"No experiment found with ID {sacred_id}.")
        else:
            print(f"Deleting experiment with ID {sacred_id}.")
            collection.delete_one({'_id': sacred_id})


def reset_experiment(collection, exp):
    exp['status'] = 'QUEUED'
    keep_entries = ['batch_id', 'status', 'seml', 'slurm', 'config', 'queue_time']

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
            detect_killed(config_file, verbose=False)

        filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)

        nreset = collection.count_documents(filter_dict)
        exps = collection.find(filter_dict)

        if nreset >= 10:
            if input(f"Resetting the state of {nreset} experiment{s_if(nreset)}. "
                     f"Are you sure? (y/n) ").lower() != "y":
                exit()
        else:
            print(f"Resetting the state of {nreset} experiment{s_if(nreset)}.")
        for exp in exps:
            reset_experiment(collection, exp)
    else:
        exp = collection.find_one({'_id': sacred_id})
        if exp is None:
            raise LookupError(f"No experiment found with ID {sacred_id}.")
        else:
            print(f"Resetting the state of experiment with ID {sacred_id}.")
            reset_experiment(collection, exp)


def detect_killed(config_file, verbose=True):
    collection = db_utils.get_collection_from_config(config_file)
    exps = collection.find({'status': {'$in': ['PENDING', 'RUNNING']}})
    running_jobs = get_slurm_jobs()
    nkilled = 0
    for exp in exps:
        if 'slurm' in exp and 'id' in exp['slurm'] and exp['slurm']['id'] not in running_jobs:
            nkilled += 1
            collection.update_one({'_id': exp['_id']}, {'$set': {'status': 'KILLED'}})
            try:
                with open(exp['slurm']['output_file'], 'r') as f:
                    all_lines = f.readlines()
                collection.update_one({'_id': exp['_id']}, {'$set': {'fail_trace': all_lines[-4:]}})
            except IOError:
                print(f"Warning: file {exp['slurm']['output_file']} could not be read.")
    if verbose:
        print(f"Detected {nkilled} externally killed experiment{s_if(nkilled)}.")

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

    fs = gridfs.GridFS(db)
    referenced_artifact_ids = set()
    for collection_name in collection_names:
        collection = db[collection_name]
        all_artifacts = list(collection.find({}, {'artifacts': 1}))
        all_artifacts = [x['artifacts'] for x in all_artifacts if 'artifacts' in x]
        all_artifacts_flat = [art for x in all_artifacts for art in x]
        artifact_ids = set([x['file_id'] for x in all_artifacts_flat])
        referenced_artifact_ids = referenced_artifact_ids.union(artifact_ids)

    artifacts_in_db = list(db['fs.files'].find({}, {'_id': 1}))
    artifacts_in_db = set([x['_id'] for x in artifacts_in_db])
    not_referenced_artifacts = artifacts_in_db - referenced_artifact_ids
    n_delete = len(not_referenced_artifacts)
    if input(f"Deleting {n_delete} not referenced artifact{s_if(n_delete)} from database {db.name}. "
             f"Are you sure? (y/n) ").lower() != "y":
        exit()
    print('Deleting not referenced artifacts...')
    for to_delete in tqdm(not_referenced_artifacts):
        fs.delete(to_delete)
    print(f'Successfully deleted {n_delete} not referenced artifact{s_if(n_delete)}.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description="Manage experiments for the given configuration. "
                        "Each experiment is represented as a record in the database. "
                        "See examples/README.md for more details.",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            '-c', '--config-file',
            type=str, required=True,
            help="Path to the YAML configuration file for the experiment.")
    subparsers = parser.add_subparsers(title="Possible operations")

    parser_queue = subparsers.add_parser(
            "queue",
            help="Queue the experiments as defined in the configuration.")
    parser_queue.add_argument(
            '-f', '--force-duplicates', action='store_true',
            help="If True, will add experiments to the database even when experiments with identical configurations "
                 "are already in the database.")
    parser_queue.set_defaults(func=queue_experiments)

    parser_start = subparsers.add_parser(
            "start",
            help="Fetch queued experiments from the database and run them (by default via Slurm).")
    parser_start.add_argument(
            '-l', '--local', action='store_true',
            help="Run the experiments locally.")
    parser_start.add_argument(
            '--test', type=int, default=-1,
            help="Only run the specified number of experiments to try and see whether they work.")
    parser_start.add_argument(
            '--verbose', '-v', action='store_true',
            help='Display more log messages.')
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
        "clean_db",
        help="Remove orphaned artifacts in the DB from runs which have been deleted.")

    parser_clean_db.add_argument(
        '-a', '--all_collections', action='store_true',
        help="If True, will scan all collections for orphaned artifacts (not just the one provided in the config).")

    parser_clean_db.set_defaults(func=clean_unreferenced_artifacts)

    args = parser.parse_args()
    f = args.func
    del args.func
    if 'filter_states' in args:
        args.filter_states = [state.upper() for state in args.filter_states]
    f(**args.__dict__)
