#!/usr/bin/env python
import functools
import json
import logging
import sys
from pathlib import Path
from typing import List

import typer
from typing_extensions import Annotated

from seml.add import add_config_files
from seml.configure import configure
from seml.database import (clean_unreferenced_artifacts,
                           get_collections_from_mongo_shell_or_pymongo,
                           get_mongodb_config, list_database)
from seml.manage import (cancel_experiments, delete_experiments, detect_killed,
                         print_fail_trace, reload_sources, report_status,
                         reset_experiments)
from seml.settings import SETTINGS
from seml.start import print_command, start_experiments, start_jupyter_job
from seml.utils import cache_to_disk, LoggingFormatter

States = SETTINGS.STATES


@cache_to_disk('db_config', SETTINGS.AUTOCOMPLETE_CACHE_ALIVE_TIME)
def db_collection_completer():
    """ CLI completion for db collections. """
    config = get_mongodb_config()
    return list(get_collections_from_mongo_shell_or_pymongo(**config))


app = typer.Typer(no_args_is_help=True)
YesAnnotation = Annotated[bool, typer.Option(
    '-y',
    '--yes',
    help="Automatically confirm all dialogues with yes.",
)]
SacredIdAnnotation = Annotated[int, typer.Option(
    '-id',
    '--sacred-id',
    help="Sacred ID (_id in the database collection) of the experiment. "
            "Takes precedence over other filters.",
)]
FilterDictAnnotation = Annotated[dict, typer.Option(
    '-f',
    '--filter-dict',
    help="Dictionary (passed as a string, e.g. '{\"config.dataset\": \"cora_ml\"}') to filter "
            "the experiments by.",
    metavar='JSON',
    parser=json.loads,
)]
BatchIdAnnotation = Annotated[int, typer.Option(
    '-b',
    '--batch-id',
    help="Batch ID (batch_id in the database collection) of the experiments. "
            "Experiments that were staged together have the same batch_id.",
)]

_STATE_LIST = [s for states in States.values() for s in states]
FilterStatesAnnotation = Annotated[List[str], typer.Option(
    '-s',
    '--filter-states',
    help='List of states to filter the experiments by. If empty (""), all states are considered.',
    metavar=f'[{"|".join(_STATE_LIST)}]',
    parser=lambda s: s.strip().upper(),
    callback=lambda values: [
        __x.strip().upper()
        for _x in values
        for __x in _x.replace(',', ' ').split()
        if __x
    ],
)]
SBatchOptionsAnnotation = Annotated[dict, typer.Option(
    '-sb',
    '--sbatch-options',
    help="Dictionary (passed as a string, e.g. '{\"gres\": \"gpu:2\"}') to request two GPUs.",
    metavar='JSON',
    parser=json.loads
)]
NumExperimentsAnnotation = Annotated[int, typer.Option(
    '-n',
    '--num-experiments',
    help="Number of experiments to start. "
            "0: all (staged) experiments ",
)]
FileOutputAnnotation = Annotated[
    bool,
    typer.Option(
        help="Write the experiment's output to a file.",
    )
]
OutputToConsoleAnnotation = Annotated[
    bool,
    typer.Option(
        help="Write the experiment's output to the console.",
    )
]
StealSlurmAnnotation = Annotated[
    bool,
    typer.Option(
        '-ss',
        '--steal-slurm',
        help="Local jobs 'steal' from the Slurm queue, "
            "i.e. also execute experiments waiting for execution via Slurm.",
    )
]
PostMortemAnnotation = Annotated[
    bool,
    typer.Option(
        '-pm',
        '--post-mortem',
        help="Activate post-mortem debugging with pdb.",
    )
]
WorkerGPUsAnnotation = Annotated[
    str,
    typer.Option(
        '-wg',
        '--worker-gpus',
        help="The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.",
    )
]
WorkerCPUsAnnotation = Annotated[
    int,
    typer.Option(
        '-wc',
        '--worker-cpus',
        help="The number of CPUs used by the local worker. Will be directly passed to OMP_NUM_THREADS.",
    )
]
WorkerEnvAnnotation = Annotated[
    dict,
    typer.Option(
        '-we',
        '--worker-env',
        help="Further environment variables to be set for the local worker.",
        metavar='JSON',
        parser=json.loads
    )
]



@app.callback()
def callback(
    ctx: typer.Context,
    collection: Annotated[
        str,
        typer.Argument(
            help="The name of the database collection to use.",
            autocompletion=db_collection_completer
        )
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            '-v',
            '--verbose',
            help="Whether to print debug messages."
        )
    ] = False
):
    """SEML - Slurm Experiment Management Library."""
    if len(logging.root.handlers) == 0:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(
                logging.VERBOSE if verbose else logging.INFO,
                show_path=False,
                show_level=True,
                show_time=False,
            )
            logging.basicConfig(
                level="NOTSET", format="%(message)s", handlers=[handler]
            )
        except ImportError:
            hdlr = logging.StreamHandler(sys.stderr)
            hdlr.setFormatter(LoggingFormatter())
            logging.root.addHandler(hdlr)
            if verbose:
                logging_level = logging.VERBOSE
            else:
                logging_level = logging.INFO
            logging.root.setLevel(logging_level)

    ctx.obj = dict(
        collection=collection,
        verbose=verbose
    )


@app.command("list")
def list_command(
    ctx: typer.Context,
    pattern: Annotated[str, typer.Argument(
        help="A regex that must match the collections to print."
    )] = r'.*',
    progress: Annotated[bool, typer.Option(
        '-p',
        '--progress',
        help="Whether to print a progress bar for iterating over collections."
    )] = False
):
    """Lists all collections in the database."""
    assert not ctx.obj['collection'], "Collection name should not be passed to this command."
    list_database(pattern, progress=progress)


@app.command("clean-db")
def clean_db_command(
    ctx: typer.Context,
    yes: YesAnnotation = False
):
    """Remove orphaned artifacts in the DB from runs which have been deleted.."""
    clean_unreferenced_artifacts(ctx.obj['collection'], yes=yes)


@app.command("configure")
def configure_command(
    ctx: typer.Context,
    all: Annotated[
        bool,
        typer.Option(
            '-a',
            '--all',
            help="Configure all SEML settings",
        ),
    ] = False,
    mongodb: Annotated[
        bool,
        typer.Option(
            help="Configure MongoDB settings",
        ),
    ] = True
):
    """
    Configure SEML (database, argument completion, ...).
    """
    assert not ctx.obj['collection'], "Collection name should not be passed to this command."
    configure(all=all, mongodb=mongodb)


@app.command("start-jupyter")
def start_jupyter_command(
    ctx: typer.Context,
    lab: Annotated[
        bool,
        typer.Option(
            '-l',
            '--lab',
            help="Start a jupyter-lab instance instead of jupyter notebook.",
        ),
    ] = False,
    conda_env: Annotated[
        str,
        typer.Option(
            '-c',
            '--conda-env',
            help="Start the Jupyter instance in a Conda environment.",
        ),
    ] = None,
    sbatch_options: SBatchOptionsAnnotation = None
):
    """
    Start a Jupyter slurm job. Uses SBATCH options defined in settings.py under
    SBATCH_OPTIONS_TEMPLATES.JUPYTER
    """
    assert not ctx.obj['collection'], "Collection name should not be passed to this command."
    start_jupyter_job(lab=lab, conda_env=conda_env, sbatch_options=sbatch_options)


@app.command("cancel")
def cancel_command(
    ctx: typer.Context,
    sacred_id: SacredIdAnnotation = None,
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
    filter_states: FilterStatesAnnotation = [*States.PENDING, *States.RUNNING],
    wait: Annotated[
        bool,
        typer.Option(
            '-w',
            '--wait',
            help="Wait until all jobs are properly cancelled.",
        ),
    ] = False,
    yes: YesAnnotation = False
):
    """
    Cancel the Slurm job/job step corresponding to experiments, filtered by ID or state.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    wait = wait or len([a for a in sys.argv if a in command_names(app)]) > 1
    cancel_experiments(
        ctx.obj['collection'],
        sacred_id=sacred_id,
        filter_dict=filter_dict,
        batch_id=batch_id,
        filter_states=filter_states,
        wait=wait,
        yes=yes
    )


@app.command("add")
def add_command(
    ctx: typer.Context,
    config_files: Annotated[
        List[Path],
        typer.Argument(
            help="Path to the YAML configuration file for the experiment.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    hash: Annotated[
        bool,
        typer.Option(
            help="Use the hash of the config dictionary to filter out duplicates (by comparing all "
                 "dictionary values individually). Only disable this if you have a good reason as it is faster.",
        ),
    ] = True,
    sanity_check: Annotated[
        bool,
        typer.Option(
            help="Check the config for missing/unused arguments. "
                 "Disable this if the check fails unexpectedly when using "
                 "advanced Sacred features or to accelerate adding.",
        ),
    ] = True,
    code_checkpoint: Annotated[
        bool,
        typer.Option(
            help="Save a checkpoint of the code in the database. "
                    "Disable this if you want your experiments to use the current code"
                    "instead of the code at the time of adding.",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            help="Force adding the experiment even if it already exists in the database.",
        ),
    ] = False,
    overwrite_params: Annotated[
        dict,
        typer.Option(
            help="Dictionary (passed as a string, e.g. '{\"epochs\": 100}') to overwrite parameters in the config.",
            metavar='JSON',
            parser=json.loads
        ),
    ] = None,
):
    """
    Add experiments to the database as defined in the configuration.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    add_config_files(
        ctx.obj['collection'],
        config_files,
        force_duplicates=force,
        no_hash=not hash,
        no_sanity_check=not sanity_check,
        no_code_checkpoint=not code_checkpoint,
        overwrite_params=overwrite_params
    )

@app.command("start")
def start_command(
    ctx: typer.Context,
    sacred_id: SacredIdAnnotation = None,
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
    debug: Annotated[
        bool,
        typer.Option(
            '-d',
            '--debug',
            help="Run a single interactive experiment without Sacred observers and with post-mortem debugging. "
                 "Implies `--verbose --num-exps 1 --post-mortem --output-to-console`.",
        ),
    ] = False,
    debug_server: Annotated[
        bool,
        typer.Option(
            '-ds',
            '--debug-server',
            help="Run the experiment with a debug server, to which you can remotely connect with e.g. VS Code. "
                 "Implies `--debug`.",
        ),
    ] = False,
    local: Annotated[
        bool,
        typer.Option(
            '-l',
            '--local',
            help="Run the experiment locally instead of on a Slurm cluster.",
        ),
    ] = False,
    no_worker: Annotated[
        bool,
        typer.Option(
            '-nw',
            '--no-worker',
            help="Do not launch a local worker after setting experiments' state to PENDING.",
        ),
    ] = False,
    num_exps: NumExperimentsAnnotation = 0,
    file_output: FileOutputAnnotation = True,
    steal_slurm: StealSlurmAnnotation = False,
    post_mortem: PostMortemAnnotation = False,
    output_to_console: OutputToConsoleAnnotation = False,
    worker_gpus: WorkerGPUsAnnotation = None,
    worker_cpus: WorkerCPUsAnnotation = None,
    worker_env: WorkerEnvAnnotation = None,
):
    """
    Fetch staged experiments from the database and run them (by default via Slurm).
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    start_experiments(
        ctx.obj['collection'],
        local=local,
        sacred_id=sacred_id,
        batch_id=batch_id,
        filter_dict=filter_dict,
        num_exps=num_exps,
        post_mortem=post_mortem,
        debug=debug,
        debug_server=debug_server,
        output_to_console=output_to_console,
        no_file_output=not file_output,
        steal_slurm=steal_slurm,
        no_worker=no_worker,
        set_to_pending=True,
        worker_gpus=worker_gpus,
        worker_cpus=worker_cpus,
        worker_environment_vars=worker_env,
    )


@app.command("launch-worker")
def launch_worker_command(
    ctx: typer.Context,
    num_exps: NumExperimentsAnnotation = 0,
    file_output: FileOutputAnnotation = True,
    steal_slurm: StealSlurmAnnotation = False,
    post_mortem: PostMortemAnnotation = False,
    output_to_console: OutputToConsoleAnnotation = False,
    worker_gpus: WorkerGPUsAnnotation = None,
    worker_cpus: WorkerCPUsAnnotation = None,
    worker_env: WorkerEnvAnnotation = None,
    sacred_id: SacredIdAnnotation = None,
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
):
    """
    Launch a local worker that runs PENDING jobs.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    start_experiments(
        ctx.obj['collection'],
        local=True,
        sacred_id=sacred_id,
        batch_id=batch_id,
        filter_dict=filter_dict,
        num_exps=num_exps,
        post_mortem=post_mortem,
        output_to_console=output_to_console,
        no_file_output=not file_output,
        steal_slurm=steal_slurm,
        no_worker=False,
        set_to_pending=False,
        worker_gpus=worker_gpus,
        worker_cpus=worker_cpus,
        worker_environment_vars=worker_env,
    )



@app.command("print-fail-trace")
def print_traces_command(
    ctx: typer.Context,
    sacred_id: SacredIdAnnotation = None,
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
    filter_states: FilterStatesAnnotation = [*States.FAILED, *States.KILLED, *States.INTERRUPTED],
    yes: YesAnnotation = False,
):
    """
    Prints fail traces of all failed experiments.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    print_fail_trace(
        ctx.obj['collection'],
        sacred_id=sacred_id,
        filter_states=filter_states,
        batch_id=batch_id,
        filter_dict=filter_dict,
        yes=yes
    )


@app.command("reload-sources")
def reload_sources_command(
    ctx: typer.Context,
    keep_old: Annotated[
        bool,
        typer.Option(
            '-k',
            '-keep-old',
            help="Keep the old source files in the database.",
        ),
    ] = False,
    batch_ids: Annotated[list[int], typer.Option(
        '-b',
        '--batch-ids',
        help="Batch IDs (batch_id in the database collection) of the experiments. "
                "Experiments that were staged together have the same batch_id.",
    )] = None,
    yes: YesAnnotation = False,
):
    """
    Reload stashed source files.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    reload_sources(
        ctx.obj['collection'],
        batch_ids=batch_ids,
        keep_old=keep_old,
        yes=yes
    )


@app.command("print_command")
def print_command_command(
    ctx: typer.Context,
    sacred_id: SacredIdAnnotation = None,
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
    num_exps: NumExperimentsAnnotation = 0,
    worker_gpus: WorkerGPUsAnnotation = None,
    worker_cpus: WorkerCPUsAnnotation = None,
    worker_env: WorkerEnvAnnotation = None,
):
    """
    Print the commands that would be executed by `start`.
    """
    print_command(
        ctx.obj['collection'],
        sacred_id=sacred_id,
        batch_id=batch_id,
        filter_dict=filter_dict,
        num_exps=num_exps,
        worker_gpus=worker_gpus,
        worker_cpus=worker_cpus,
        worker_environment_vars=worker_env,
    )



@app.command("reset")
def reset_command(
    ctx: typer.Context,
    sacred_id: SacredIdAnnotation = None,
    filter_states: FilterStatesAnnotation = [*States.FAILED, *States.KILLED, *States.INTERRUPTED],
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
    yes: YesAnnotation = False,
):
    """
    Reset the state of experiments by setting their state to STAGED and cleaning their database entry.
    Does not cancel Slurm jobs.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    reset_experiments(
        ctx.obj['collection'],
        sacred_id=sacred_id,
        filter_states=filter_states,
        batch_id=batch_id,
        filter_dict=filter_dict,
        yes=yes
    )


@app.command("delete")
def delete_command(
    ctx: typer.Context,
    sacred_id: SacredIdAnnotation = None,
    filter_states: FilterStatesAnnotation = [*States.STAGED, *States.FAILED, *States.KILLED, *States.INTERRUPTED],
    filter_dict: FilterDictAnnotation = None,
    batch_id: BatchIdAnnotation = None,
    yes: YesAnnotation = False,
):
    """
    Delete experiments by ID or state (does not cancel Slurm jobs).
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    delete_experiments(
        ctx.obj['collection'],
        sacred_id=sacred_id,
        filter_states=filter_states,
        batch_id=batch_id,
        filter_dict=filter_dict,
        yes=yes
    )


@app.command("detect-killed")
def detect_killed_command(
    ctx: typer.Context,
):
    """
    Detect experiments where the corresponding Slurm jobs were killed externally.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    detect_killed(ctx.obj['collection'])


@app.command("status")
def status_command(
    ctx: typer.Context,
):
    """
    Report status of experiments in the database collection.
    """
    assert ctx.obj['collection'], "Collection name must be passed to this command."
    report_status(ctx.obj['collection'])


@functools.lru_cache()
def command_names(app: typer.Typer) -> set[str]:
    return {
        cmd.name if cmd.name else cmd.callback.__name__
        for cmd in app.registered_commands
    }


def split_args(commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    if len(split_argv) == 1:
        return split_argv
    shared = split_argv[0]
    chained_commands = split_argv[1:]
    # If none of the shared args contains a collection
    # name, we add the default collection name.
    if all(arg.startswith('-') for arg in shared):
        shared.append('')
    # Combine commands with the shared arguments
    result = []
    for split in chained_commands:
        result.append(shared + split)
    return result


def main():
    # We have to split the arguments manually to get proper chaining.
    # If we were to use typer built-in chaining, lists would end the chain.
    for args in split_args(command_names(app)):
        # The app will typically exit after running once.
        # We want to run it multiple times, so we catch the SystemExit exception.
        try:
            app(args)
        except SystemExit as e:
            if e.code == 0:
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
