import os
import sys
import argparse
import json
import logging

from seml.manage import (report_status, cancel_experiments, delete_experiments, detect_killed, reset_experiments,
                         mongodb_credentials_prompt)
from seml.queuing import queue_experiments
from seml.start import start_experiments
from seml.config import read_config
from seml.database import clean_unreferenced_artifacts
from seml.utils import LoggingFormatter


def main():
    parser = argparse.ArgumentParser(
            description="Manage experiments for the given configuration. "
                        "Each experiment is represented as a record in the database. "
                        "See examples/README.md for more details.",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            'db_collection_name', type=str, nargs='?', default=None,
            help="Name of the database collection for the experiment.")
    parser.add_argument(
            '--verbose', '-v', action='store_true',
            help='Display more log messages.')

    subparsers = parser.add_subparsers(title="Possible operations")

    parser_configure = subparsers.add_parser(
        "configure",
        help="Provide your MongoDB credentials.")
    parser_configure.set_defaults(func=mongodb_credentials_prompt)

    parser_queue = subparsers.add_parser(
            "queue",
            help="Queue the experiments as defined in the configuration.")
    parser_queue.add_argument(
            'config_file', type=str, nargs='?', default=None,
            help="Path to the YAML configuration file for the experiment.")
    parser_queue.add_argument(
            '-nh', '--no-hash', action='store_true',
            help="Do not use the hash of the config dictionary to filter out duplicates (by comparing all"
                 "dictionary values individually). This is much  slower, so use only if you have a good reason not to"
                 " use the hash.")
    parser_queue.add_argument(
            '-nc', '--no-config-check', action='store_true',
            help="Do not check the config for missing/unused arguments. "
                 "Use this if the check fails unexpectedly when using "
                 "advanced Sacred features or to accelerate queueing.")
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
            help="Run the experiments locally (not via Slurm).")
    parser_start.add_argument(
            '-n', '--num-exps', type=int, default=-1,
            help="Only start the specified number of experiments.")
    parser_start.add_argument(
            '-u', '--unobserved', action='store_true',
            help="Run the experiments without Sacred observers (no changes to the database). "
                 "This also disables output capturing by Sacred, facilitating the use of debuggers (pdb, ipdb).")
    parser_start.add_argument(
            '-pm', '--post-mortem', action='store_true',
            help="Activate post-mortem debugging with pdb.")
    parser_start.add_argument(
            '-d', '--debug', action='store_true',
            help="Run a single experiment locally without Sacred observers and with post-mortem debugging. "
                 "This is equivalent to "
                 "`--verbose --local --num-exps 1 --unobserved --post-mortem --output-to-console`.")
    parser_start.add_argument(
            '-dr', '--dry-run', action='store_true',
            help="Only show the associated commands instead of running the experiments.")
    parser_start.add_argument(
            '-id', '--sacred-id', type=int,
            help="Sacred ID (_id in the database collection) of the experiment to cancel.")
    parser_start.add_argument(
            '-b', '--batch-id', type=int,
            help="Batch ID (batch_id in the database collection) of the experiments to be cancelled. "
                 "Experiments that were queued together have the same batch_id.")
    parser_start.add_argument(
        '-f', '--filter-dict', type=json.loads,
        help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter the experiments by.")
    parser_start.add_argument(
        '-o', '--output-to-console', action='store_true',
        help="Print output to console instead of writing it to a log file. Only possible if experiment is run locally.")
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
            help="Batch ID (batch_id in the database collection) of the experiments to be cancelled. "
                 "Experiments that were queued together have the same batch_id.")
    parser_cancel.add_argument(
            '-f', '--filter-dict', type=json.loads,
            help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') "
                 "to filter the experiments by.")
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
            help="Batch ID (batch_id in the database collection) of the experiments to be deleted. "
                 "Experiments that were queued together have the same batch_id.")
    parser_delete.add_argument(
            '-f', '--filter-dict', type=json.loads,
            help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') "
                 "to filter the experiments by.")
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
            help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') "
                 "to filter the experiments by.")
    parser_reset.add_argument(
            '-b', '--batch-id', type=int,
            help="Batch ID (batch_id in the database collection) of the experiments to be deleted. "
                 "Experiments that were queued together have the same batch_id.")
    parser_reset.set_defaults(func=reset_experiments)

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

    if args.func == mongodb_credentials_prompt:  # launch SEML configure.
        del args.db_collection_name
    else:  # otherwise remove the flag as it is not used elsewhere.
        if not args.db_collection_name:
            parser.error("the following arguments are required: db_collection_name")
        else:
            if os.path.isfile(args.db_collection_name):
                logging.warning("Loading the collection name from a config file. This has been deprecated. "
                                "Please instead provide a database collection name in the command line.")
                seml_config, _, _ = read_config(args.db_collection_name)
                if args.func == queue_experiments:
                    args.config_file = args.db_collection_name
                args.db_collection_name = seml_config['db_collection']
            elif args.func == queue_experiments and not args.config_file:
                parser_queue.error("the following arguments are required: config_file")

    f = args.func
    del args.func
    del args.verbose
    if 'filter_states' in args:
        args.filter_states = [state.upper() for state in args.filter_states]
    f(**args.__dict__)


if __name__ == "__main__":
    main()
