import sys
import argparse
import json
import logging

from seml.manage import (report_status, cancel_experiments, delete_experiments, detect_killed, reset_experiments,
                         mongodb_credentials_prompt)
from seml.add import add_experiments
from seml.start import start_experiments, start_jupyter_job
from seml.database import clean_unreferenced_artifacts
from seml.utils import LoggingFormatter
from seml.settings import SETTINGS

States = SETTINGS.STATES


def main():
    parser = argparse.ArgumentParser(
            description="Manage experiments for the given configuration. "
                        "Each experiment is represented as a record in the database. "
                        "See examples/README.md for more details.",
            formatter_class=argparse.RawTextHelpFormatter,
            add_help=True)
    parser.add_argument(
            'db_collection_name', type=str, nargs='?', default=None,
            help="Name of the database collection for the experiment.")
    parser.add_argument(
            '--verbose', '-v', action='store_true',
            help='Display more log messages.')

    subparsers = parser.add_subparsers(title="Possible operations")

    parser_jupyter = subparsers.add_parser(
            "jupyter",
            help="Start a Jupyter slurm job. Uses SBATCH options defined in settings.py under "
                 "BATCH_OPTIONS_TEMPLATES.JUPYTER_JOB")
    parser_jupyter.add_argument(
            "-l", "--lab", action='store_true',
            help="Start a jupyter-lab instance instead of jupyter notebook.")
    parser_jupyter.add_argument(
            "-c", "--conda-env", type=str, default=None,
            help="Start the Jupyter instance in a Conda environment.")
    parser_jupyter.add_argument(
            '-sb', '--sbatch-options', type=json.loads,
            help="Dictionary (passed as a string, e.g. '{\"gres\": \"gpu:2\"}') to request two GPUs.")
    parser_jupyter.set_defaults(func=start_jupyter_job)

    parser_configure = subparsers.add_parser(
            "configure",
            help="Provide your MongoDB credentials.")
    parser_configure.set_defaults(func=mongodb_credentials_prompt)

    parser_add = subparsers.add_parser(
            "add",
            help="Add the experiments to the database as defined in the configuration.")
    parser_add.add_argument(
            'config_file', type=str,
            help="Path to the YAML configuration file for the experiment.")
    parser_add.add_argument(
            '-nh', '--no-hash', action='store_true',
            help="Do not use the hash of the config dictionary to filter out duplicates (by comparing all "
                 "dictionary values individually). This is much slower, so use only if you have a good reason not to "
                 "use the hash.")
    parser_add.add_argument(
            '-nsc', '--no-sanity-check', action='store_true',
            help="Do not check the config for missing/unused arguments. "
                 "Use this if the check fails unexpectedly when using "
                 "advanced Sacred features or to accelerate adding.")
    parser_add.add_argument(
            '-ncc', '--no-code-checkpoint', action='store_true',
            help="Do not save the source code files in the MongoDB. "
                 "When a staged experiment is started, it will instead use the current version of the code "
                 "files (which might have been updated in the meantime or could fail when started).")
    parser_add.add_argument(
            '-f', '--force-duplicates', action='store_true',
            help="Add experiments to the database even when experiments with identical configurations "
                 "are already in the database.")
    parser_add.set_defaults(func=add_experiments)

    parser_start = subparsers.add_parser(
            "start",
            help="Fetch staged experiments from the database and run them (by default via Slurm).")
    parser_start.add_argument(
            '-pc', '--print-command', action='store_true',
            help="Only show the associated commands instead of running the experiments.")
    parser_start.add_argument(
            '-d', '--debug', action='store_true',
            help="Run a single interactive experiment without Sacred observers and with post-mortem debugging. "
                 "Implies `--verbose --num-exps 1 --post-mortem --output-to-console`.")
    parser_start.add_argument(
            '-ds', '--debug-server', action='store_true',
            help="Run the experiment with a debug server, to which you can remotely connect with e.g. VS Code. "
                 "Implies `--debug`.")
    parser_start_local = parser_start.add_argument_group("optional arguments for local jobs")
    parser_start_local.add_argument(
            '-l', '--local', action='store_true',
            help="Run the experiments locally (not via Slurm).")
    parser_start_local.add_argument(
            '-nw', '--no-worker', action='store_true',
            help="Do not launch a local worker after setting experiments' state to PENDING.")
    parser_start.set_defaults(func=start_experiments, set_to_pending=True)

    parser_launch_worker = subparsers.add_parser(
        "launch-worker",
        help="Launch a local worker that runs PENDING jobs.")
    parser_launch_worker.set_defaults(func=start_experiments, set_to_pending=False, no_worker=False, local=True,
                                      debug=False, debug_server=False, print_command=False)

    for subparser in [parser_start, parser_launch_worker]:
        subparser.add_argument(
                '-n', '--num-exps', type=int, default=0,
                help="Only start the specified number of experiments. 0: run all staged experiments.")
        subparser.add_argument(
                '-nf', '--no-file-output', action='store_true',
                help="Do not save the console output in a file.")

    for subparser in [parser_start_local, parser_launch_worker]:
        subparser.add_argument(
                '-ss', '--steal-slurm', action='store_true',
                help="Local jobs 'steal' from the Slurm queue, "
                     "i.e. also execute experiments waiting for execution via Slurm.")
        subparser.add_argument(
                '-wg', '--worker-gpus', type=str,
                help="The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.")
        subparser.add_argument(
                '-wc', '--worker-cpus', type=int,
                help="The number of CPU cores used by the local worker. Will be directly passed to OMP_NUM_THREADS.")
        subparser.add_argument(
                '-we', '--worker-environment-vars', type=json.loads,
                help="Further environment variables to be set for the local worker.")
        subparser.add_argument(
                '-pm', '--post-mortem', action='store_true',
                help="Activate post-mortem debugging with pdb.")
        subparser.add_argument(
                '-o', '--output-to-console', action='store_true',
                help="Print output to console.")

    parser_status = subparsers.add_parser(
            "status",
            help="Report status of experiments in the database collection.")
    parser_status.set_defaults(func=report_status)

    parser_cancel = subparsers.add_parser(
            "cancel",
            help="Cancel the Slurm job/job step corresponding to experiments, filtered by ID or state.")
    parser_cancel.add_argument(
            '-s', '--filter-states', type=str, nargs='*', default=[*States.PENDING, *States.RUNNING],
            help="List of states to filter experiments by. Cancels all experiments if an empty list is passed. "
                 "Default: Cancel all pending and running experiments.")
    parser_cancel.set_defaults(func=cancel_experiments)

    parser_delete = subparsers.add_parser(
            "delete",
            help="Delete experiments by ID or state (does not cancel Slurm jobs).")
    parser_delete.add_argument(
            '-s', '--filter-states', type=str, nargs='*', default=[*States.STAGED, *States.FAILED,
                                                                   *States.KILLED, *States.INTERRUPTED],
            help="List of states to filter experiments by. Deletes all experiments if an empty list is passed. "
                 "Default: Delete all staged, failed, killed and interrupted experiments.")
    parser_delete.set_defaults(func=delete_experiments)

    parser_reset = subparsers.add_parser(
            "reset",
            help="Reset the state of experiments by setting their state to staged and cleaning their database entry. "
                 "Does not cancel Slurm jobs.")
    parser_reset.add_argument(
            '-s', '--filter-states', type=str, nargs='*', default=[*States.FAILED, *States.KILLED,
                                                                   *States.INTERRUPTED],
            help="List of states to filter experiments by. "
                 "Resets all experiments if an empty list is passed. "
                 "Default: Reset failed, killed and interrupted experiments.")
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

    for subparser in [parser_start, parser_launch_worker, parser_cancel, parser_delete, parser_reset]:
        subparser.add_argument(
                '-id', '--sacred-id', type=int,
                help="Sacred ID (_id in the database collection) of the experiment to start.")
        subparser.add_argument(
                '-b', '--batch-id', type=int,
                help="Batch ID (batch_id in the database collection) of the experiments to be started. "
                     "Experiments that were staged together have the same batch_id.")
        subparser.add_argument(
                '-f', '--filter-dict', type=json.loads,
                help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter "
                     "the experiments by.")

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

    if args.func in [mongodb_credentials_prompt, start_jupyter_job]:
        # No collection name required
        del args.db_collection_name
    elif not args.db_collection_name:
        parser.error("the following arguments are required: db_collection_name")

    f = args.func
    del args.func
    del args.verbose
    if 'filter_states' in args:
        args.filter_states = [state.upper() for state in args.filter_states]
    f(**args.__dict__)


if __name__ == "__main__":
    main()
