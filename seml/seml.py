import argparse
import subprocess
import warnings
import datetime
import json

from seml.misc import get_slurm_jobs, s_if
from seml import database_utils as db_utils
from seml.queue_experiments import queue_experiments
from seml.start_experiments import start_experiments


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
            if batch_id is None and ncancel >= 10:
                if input(f"Cancelling {ncancel} experiment{s_if(ncancel)}. "
                         f"Are you sure? (y/n) ").lower() != "y":
                    exit()
            else:
                print(f"Cancelling {ncancel} experiment{s_if(ncancel)}.")

            exps = list(collection.find(filter_dict))
            # set of slurm IDs in the database
            slurm_ids = set([e['slurm']['id'] for e in exps if "slurm" in e and ["id"] in e['slurm']])
            # set of experiment IDs to be cancelled.
            exp_ids = set([e['_id'] for e in exps])
            to_cancel = set()

            # iterate over slurm IDs to check which slurm jobs can be cancelled altogether
            for s_id in slurm_ids:
                # find experiments RUNNING under the slurm job
                jobs_running = list(collection.find({'slurm.id': s_id,
                                                     'status'  : {"$in": ["RUNNING"]}},
                                                    {"_id": 1}))
                running_exp_ids = set(e['_id'] for e in jobs_running)
                if len(running_exp_ids.difference(exp_ids)) == 0:
                    # there are no running jobs in this slurm job that should not be canceled.
                    to_cancel.add(str(s_id))

            # cancel all Slurm jobs for which no running experiment remains.
            if len(to_cancel) > 0:
                subprocess.check_output(f"scancel {' '.join(list(to_cancel))}", shell=True)

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

        if batch_id is None and ndelete >= 10:
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
    keep_entries = ['seml', 'config', 'queue_time', 'batch_id']
    collection.replace_one({'_id': exp['_id']}, {entry: exp[entry] for entry in keep_entries}, upsert=False)
    collection.update_one({'_id': exp['_id']}, {"$set": {"status": 'QUEUED'}}, upsert=False)


def reset_states(config_file, sacred_id, filter_states, batch_id, filter_dict):
    collection = db_utils.get_collection_from_config(config_file)

    if sacred_id is None:
        if len({'PENDING', 'RUNNING', 'KILLED'} & set(filter_states)) > 0:
            detect_killed(config_file, verbose=False)

        filter_dict = db_utils.build_filter_dict(filter_states, batch_id, filter_dict)

        nreset = collection.count_documents(filter_dict)
        exps = collection.find(filter_dict)

        if batch_id is None and nreset >= 10:
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
            '--force-duplicates', action='store_true',
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

    args = parser.parse_args()
    f = args.func
    del args.func
    if 'filter_states' in args:
        args.filter_states = [state.upper() for state in args.filter_states]
    f(**args.__dict__)
