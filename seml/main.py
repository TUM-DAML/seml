import sys
import argparse
import json
import logging

from seml.manage import (report_status, cancel_experiments, delete_experiments, detect_killed, reset_experiments,
                         mongodb_credentials_prompt, reload_sources)
from seml.add import add_experiments
from seml.start import start_experiments, start_jupyter_job, print_command
from seml.database import clean_unreferenced_artifacts
from seml.utils import LoggingFormatter
from seml.settings import SETTINGS

States = SETTINGS.STATES


def parse_args(parser, commands):
    # https://stackoverflow.com/a/43927360
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Parse only the top-level commands if there are no subcommands
    # We need to do this to ensure seml --help is working
    if len(split_argv) == 1:
        parser.parse_args(split_argv[0])
    # Parse all subcommands
    commands = []
    for argv in split_argv[1:]:
        # Copy the original arguments and the command specific ones
        n = parser.parse_args(split_argv[0] + argv)
        commands.append(n)
    return commands


class ParameterAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {
            value.split('=')[0]: eval('='.join(value.split('=')[1:]))
            for value in values
        })


def main():
    parser = argparse.ArgumentParser(
            description="Manage experiments for the given configuration. "
                        "Each experiment is represented as a record in the database. "
                        "See examples/README.md for more details.",
            formatter_class=argparse.RawTextHelpFormatter,
            add_help=True)
    parser.set_defaults(func=parser.print_usage)
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
                 "SBATCH_OPTIONS_TEMPLATES.JUPYTER")
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
    parser_add.add_argument(
            '-o', '--overwrite-params', action=ParameterAction, nargs='+', default={},
            help="Specifies parameters that overwrite their respective values in all configs."
                 "Format: <param>=<value>, use flat dictionary notation with key1.key2=value."
    )
    parser_add.set_defaults(func=add_experiments)

    parser_start = subparsers.add_parser(
            "start",
            help="Fetch staged experiments from the database and run them (by default via Slurm).")
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


    parser_reload = subparsers.add_parser(
            "reload-sources",
            help="Reload uploaded source files."
    )
    parser_reload.add_argument(
            '-k', '--keep-old', action='store_true',
            help="Keeps the old source files in the database. (You will have to manually delete them or reload again.)"
    )
    parser_reload.add_argument(
            '-b', '--batch-ids', type=int, default=None, nargs='*',
            help="Batch IDs (batch_id in the database collection) of the experiments. "
                 "Experiments that were staged together have the same batch_id."
    )
    parser_reload.set_defaults(func=reload_sources)

    parser_launch_worker = subparsers.add_parser(
        "launch-worker",
        help="Launch a local worker that runs PENDING jobs.")
    parser_launch_worker.set_defaults(func=start_experiments, set_to_pending=False, no_worker=False, local=True,
                                      debug=False, debug_server=False)

    parser_print_command = subparsers.add_parser(
        "print-command",
        help="Print the commands for running the experiments.")
    parser_print_command.set_defaults(func=print_command)

    for subparser in [parser_start, parser_launch_worker, parser_print_command]:
        subparser.add_argument(
                '-n', '--num-exps', type=int, default=0,
                help="Only start the specified number of experiments. 0: run all (staged) experiments.")

    for subparser in [parser_start, parser_launch_worker]:
        subparser.add_argument(
                '-nf', '--no-file-output', action='store_true',
                help="Do not save the console output in a file.")

    for subparser in [parser_start_local, parser_launch_worker]:
        subparser.add_argument(
                '-ss', '--steal-slurm', action='store_true',
                help="Local jobs 'steal' from the Slurm queue, "
                     "i.e. also execute experiments waiting for execution via Slurm.")
        subparser.add_argument(
                '-pm', '--post-mortem', action='store_true',
                help="Activate post-mortem debugging with pdb.")
        subparser.add_argument(
                '-o', '--output-to-console', action='store_true',
                help="Print output to console.")

    for subparser in [parser_start_local, parser_launch_worker, parser_print_command]:
        subparser.add_argument(
                '-wg', '--worker-gpus', type=str,
                help="The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.")
        subparser.add_argument(
                '-wc', '--worker-cpus', type=int,
                help="The number of CPU cores used by the local worker. Will be directly passed to OMP_NUM_THREADS.")
        subparser.add_argument(
                '-we', '--worker-environment-vars', type=json.loads,
                help="Further environment variables to be set for the local worker.")

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

    for subparser in [parser_start, parser_launch_worker, parser_print_command,
                      parser_cancel, parser_delete, parser_reset]:
        subparser.add_argument(
                '-id', '--sacred-id', type=int,
                help="Sacred ID (_id in the database collection) of the experiment. "
                     "Takes precedence over other filters.")
        subparser.add_argument(
                '-f', '--filter-dict', type=json.loads,
                help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter "
                     "the experiments by.")
        subparser.add_argument(
                '-b', '--batch-id', type=int,
                help="Batch ID (batch_id in the database collection) of the experiments. "
                     "Experiments that were staged together have the same batch_id.")
    
    for subparser in [parser_cancel, parser_reload, parser_clean_db, parser_delete, parser_reset]:
        subparser.add_argument(
            '-y', '--yes', action='store_true',
            help="Automatically confirm all dialogues with yes."
        )

    commands = parse_args(parser, subparsers)

    # Initialize logging
    hdlr = logging.StreamHandler(sys.stderr)
    hdlr.setFormatter(LoggingFormatter())
    logging.root.addHandler(hdlr)
    
    for command in commands:
        # Set logging level
        if command.verbose:
            logging_level = logging.VERBOSE
        else:
            logging_level = logging.INFO
        logging.root.setLevel(logging_level)

        if command.func in [mongodb_credentials_prompt, start_jupyter_job, parser.print_usage]:
            # No collection name required
            del command.db_collection_name
        elif not command.db_collection_name:
            parser.error("the following arguments are required: db_collection_name")

        f = command.func
        # If we chain commands we should wait until jobs are properly cancelled
        if f == cancel_experiments and len(commands) > 1:
            command.wait = True
        del command.func
        del command.verbose
        if 'filter_states' in command:
            command.filter_states = [state.upper() for state in command.filter_states]
        f(**vars(command))


if __name__ == "__main__":
    main()
