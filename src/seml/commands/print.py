from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from io import TextIOWrapper
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, cast

from seml.commands.manage import detect_duplicates, detect_killed, should_check_killed
from seml.database import (
    build_filter_dict,
    get_collection,
    get_database,
    get_mongodb_config,
)
from seml.document import ExperimentDoc
from seml.experiment.command import (
    get_command_from_exp,
    get_config_overrides,
    get_environment_variables,
    get_shell_command,
)
from seml.settings import SETTINGS
from seml.utils import (
    find_jupyter_host,
    flatten,
    get_from_nested,
    resolve_projection_path_conflicts,
    s_if,
    slice_to_str,
    to_hashable,
    to_slices,
)
from seml.utils.slurm import get_cluster_name, get_slurm_jobs

if TYPE_CHECKING:
    import pymongo.database

States = SETTINGS.STATES


def print_fail_trace(
    db_collection_name: str,
    sacred_id: int | None,
    filter_states: list[str] | None,
    batch_id: int | None,
    filter_dict: dict | None,
    projection: list[str] | None = None,
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
    from rich.console import ConsoleRenderable, Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    from seml.console import Table, console

    detect_killed(db_collection_name, print_detected=False)
    collection = get_collection(db_collection_name)
    if projection is None:
        projection = []
    mongo_db_projection = resolve_projection_path_conflicts(
        {
            '_id': 1,
            'status': 1,
            'execution.array_id': 1,
            'execution.task_id': 1,
            'fail_trace': 1,
            'seml.description': 1,
            'batch_id': 1,
            **{key: 1 for key in projection},
        }
    )

    filter_dict = build_filter_dict(filter_states, batch_id, filter_dict, sacred_id)
    exps: list[ExperimentDoc] = list(collection.find(filter_dict, mongo_db_projection))
    for exp in exps:
        exp_id = exp.get('_id')
        status = exp.get('status')
        batch_id = exp.get('batch_id')
        slurm_array_id = exp.get('execution', {}).get('array_id', None)
        slurm_task_id = exp.get('execution', {}).get('task_id', None)
        fail_trace = exp.get('fail_trace', [])
        description = exp.get('seml', {}).get('description', None)
        header = (
            f'Experiment ID {exp_id}, '
            f'Batch ID {batch_id}, '
            f'Status: "{status}", '
            f'Slurm Array-Task id: {slurm_array_id}-{slurm_task_id}'
        )

        renderables: list[ConsoleRenderable | str] = []
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
    projection: list[str] | None = None,
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

    from seml.console import Table, console

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
    # the aggregate function is wrongly typed.
    result = cast(List[Dict[str, Any]], list(result))
    result = sorted(result, key=lambda x: list(States.keys()).index(x['_id']))
    show_descriptions = any(len(row['descriptions']) > 0 for row in result)
    # Unpack the (nested) projections
    # We keep prefixes encoded as ${id} to preserve the order of the projection keys
    result_projection: list[dict[str, set]] = []
    for record in result:
        result_projection.append(defaultdict(set))
        for idx, key in enumerate(projection):
            for values in record[f'projection_{idx}']:
                for x, y in flatten({f'${idx}': values}).items():
                    result_projection[-1][x].add(to_hashable(y))
    projection_columns = sorted({k for record in result_projection for k in record})
    # For the column headers, we replace ${id} with the projection key
    columns = []
    for projection_column in projection_columns:
        match = re.match(r'.*\$([0-9]+)(\..*|$)', projection_column)
        assert match is not None
        projection_key_idx = int(match.groups()[0])
        columns.append(
            projection_column.replace(
                f'${projection_key_idx}', projection[projection_key_idx]
            )
        )
    duplicate_experiment_ids = {
        experiment_id
        for dups in detect_duplicates(db_collection_name)
        for experiment_id in dups
    }

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


def print_collections(
    pattern: str,
    mongodb_config: dict | None = None,
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

    from seml.console import Table, console, track

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
    name_to_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {state: 0 for state in States.keys()}
    )
    name_to_descriptions = defaultdict(str)
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
        # the aggregate function is wrongly typed.
        counts_by_status = cast(List[Dict[str, Any]], list(counts_by_status))
        descriptions = db[collection_name].aggregate(
            [{'$group': {'_id': '$seml.description'}}]
        )
        # the aggregate function is wrongly typed.
        descriptions = cast(List[Dict[str, Any]], list(descriptions))
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

    df = pd.DataFrame.from_dict(name_to_counts, dtype=int).transpose()  # type: ignore
    # Remove empty collections
    if not list_empty:
        df = df[df.sum(axis=1) > 0]
    # sort rows and columns
    df = df.sort_index()[list(States.keys())]
    # add a column with the total
    df['Total'] = df.sum(axis=1)
    # I don't know why but pyright thinks it could be a numpy array.
    df = cast(pd.DataFrame, df)

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
        tab_row = [collection_name, *[str(x) for x in row.to_list()]]
        if show_description:
            tab_row.append(name_to_descriptions[collection_name])
        table.add_row(*tab_row)
    # For some reason the table thinks the terminal is larger than it is
    to_print = Align(table, align='center', width=console.width - max_len + 1)
    console.print(to_print, soft_wrap=True)


def print_duplicates(
    db_collection_name: str,
    filter_states: list[str] | None = None,
    batch_id: int | None = None,
    filter_dict: dict | None = None,
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
    from rich.panel import Panel
    from rich.text import Text

    from seml.console import console

    if should_check_killed(filter_states):
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


def print_experiment(
    db_collection_name: str,
    sacred_id: int | None = None,
    filter_states: list[str] | None = None,
    batch_id: int | None = None,
    filter_dict: dict | None = None,
    projection: list[str] | None = None,
):
    """
    Prints the details of an experiment.

    Parameters
    ----------
    db_collection_name : str
        The collection to print the experiment from
    sacred_id : Optional[int], optional
        The ID of the experiment to print, by default None
    filter_states : Optional[List[str]], optional
        Filter on experiment states, by default None
    batch_id : Optional[int], optional
        Filter on the batch ID of experiments, by default None
    filter_dict : Optional[Dict], optional
        Additional filters, by default None
    projection : Optional[List[str]], optional
        Additional values to print per experiment, by default all are printed
    """
    from rich import print_json

    from seml.console import Heading, console, pause_live_widget

    filter_dict = build_filter_dict(filter_states, batch_id, filter_dict, sacred_id)
    collection = get_collection(db_collection_name)
    if projection is None or len(projection) == 0:
        proj = {}
    else:
        proj = {'_id': 1, 'batch_id': 1, **{p: 1 for p in projection}}
    experiments: list[ExperimentDoc] = list(collection.find(filter_dict, proj))

    if len(experiments) == 0:
        logging.info('No experiment found to print.')
        return

    with pause_live_widget():
        for exp in experiments:
            console.print(Heading(f'Experiment {exp["_id"]} (batch {exp["batch_id"]})'))
            print_json(data=exp, skip_keys=True, default=str)


def print_output(
    db_collection_name: str,
    sacred_id: int | None = None,
    filter_states: list[str] | None = None,
    batch_id: int | None = None,
    filter_dict: dict | None = None,
    slurm: bool = False,
    head: int | None = None,
    tail: int | None = None,
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
    slurm : bool, optional
        Whether to print the Slurm output instead of the experiment output, by default False
    head : Optional[int], optional
        The number of lines to print from the beginning of the output, by default None
    tail : Optional[int], optional
        The number of lines to print from the end of the output, by default None
    """
    from collections import deque

    from seml.console import Heading, console, pause_live_widget

    assert tail is None or head is None, 'Cannot specify both tail and head.'

    filter_dict = build_filter_dict(
        filter_states, batch_id, filter_dict, sacred_id=sacred_id
    )
    collection = get_collection(db_collection_name)
    experiments = collection.find(
        filter_dict,
        {
            'seml.output_file': 1,
            '_id': 1,
            'batch_id': 1,
            'captured_out': 1,
            'execution': 1,
        },
    )

    def print_head(file: TextIOWrapper):
        for i, line in enumerate(file):
            console.print(line[:-1], end=line[-1])
            if head is not None and i >= head:
                break

    def print_tail(file: TextIOWrapper):
        last_rows: deque[str] = deque([], maxlen=tail)
        for line in file:
            last_rows.append(line)

        for line in last_rows:
            console.print(line[:-1], end=line[-1])

    def print_all(file: TextIOWrapper):
        for line in file:
            console.print(line[:-1], end=line[-1])

    count = 0
    for exp in experiments:
        count += 1
        console.print(Heading(f'Experiment {exp["_id"]} (batch {exp["batch_id"]})'))
        with pause_live_widget():
            # Select output file
            out_file = exp['seml'].get('output_file')
            if out_file is None or slurm:
                if not slurm:
                    logging.info(
                        f'No experiment output file found for experiment {exp["_id"]}.'
                        'Using Slurm output instead.'
                    )
                if 'slurm_output_file' in exp['execution']:
                    out_file = exp['execution']['slurm_output_file']
                else:
                    logging.error(
                        f'No Slurm output file found for experiment {exp["_id"]}.'
                    )
                    continue
            # Actually read
            try:
                with open(out_file, newline='', errors='replace') as f:
                    if head:
                        print_head(f)
                    elif tail:
                        print_tail(f)
                    else:
                        print_all(f)
                console.print()  # new line
            except OSError:
                logging.info(f'File {out_file} could not be read.')
                if 'captured_out' in exp and exp['captured_out']:
                    logging.info('Captured output from DB:')
                    console.print(exp['captured_out'])
                else:
                    logging.error('No output available.')

    if count == 0:
        logging.info('No experiments found.')


def generate_queue_table(
    db: pymongo.database.Database,
    job_ids: Sequence[str] | None,
    filter_states: Sequence[str] | None,
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
    from rich.align import Align

    from seml.console import Table

    if job_ids:
        job_infos = get_slurm_jobs(*job_ids)
    else:
        job_infos = get_slurm_jobs()

    # Find the collections
    all_collections = set(db.list_collection_names())

    collection_to_jobs: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
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
    sorted_states = sorted(states)
    table = Table(
        'Collection',
        *map(str.title, sorted_states),
        show_header=True,
    )
    cluster_name = get_cluster_name()

    def format_job(job_info, db_col_name):
        if job_info is None:
            return ''
        nodelist = job_info['NodeList']
        array_id = job_info.get('ArrayJobId', job_info['JobId'])
        task_id = job_info.get('ArrayTaskId', None)
        if task_id:
            if any(x in task_id for x in ',-'):
                task_id = f'[{task_id}]'
            job_id = f'{array_id}_{task_id}'
        else:
            job_id = array_id
        if nodelist:
            # Running
            collection = get_collection(db_col_name)
            experiments = collection.find(
                {
                    'execution.cluster': cluster_name,
                    'execution.array_id': int(array_id),
                    'execution.task_id': int(task_id),
                },
                {'_id': 1},
            )
            ids = [exp['_id'] for exp in experiments]
            if len(ids) > 0:
                return f"{job_id} ({job_info['RunTime']}, {nodelist}, Id{s_if(len(ids))}: {'|'.join(map(str, ids))})"
            else:
                return f"{job_id} ({job_info['RunTime']}, {nodelist})"
        else:
            return f"{job_id} ({job_info.get('Reason', '')})"

    for col in sorted(collections):
        row = [col]
        for state in sorted_states:
            jobs = collection_to_jobs[(col, state)]
            row.append('\n'.join(map(format_job, jobs, [col] * len(jobs))))
        table.add_row(*row)

    if len(collections) == 0:
        return Align('No jobs found.', align='center')
    return Align(table, align='center')


def print_queue(
    job_ids: Sequence[str] | None,
    filter_states: Sequence[str] | None,
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
    from rich.live import Live

    from seml.console import console, pause_live_widget

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


def print_command(
    db_collection_name: str,
    sacred_id: int | None,
    batch_id: int | None,
    filter_states: list[str],
    filter_dict: dict | None,
    num_exps: int,
    worker_gpus: str | None = None,
    worker_cpus: int | None = None,
    worker_environment_vars: dict | None = None,
    unresolved: bool = False,
    resolve_interpolations: bool = True,
):
    import rich

    from seml.console import Heading, console

    collection = get_collection(db_collection_name)

    filter_dict = build_filter_dict(filter_states, batch_id, filter_dict, sacred_id)

    env_dict = get_environment_variables(
        worker_gpus, worker_cpus, worker_environment_vars
    )

    orig_level = logging.root.level
    logging.root.setLevel(logging.NOTSET)

    exps_list = list(collection.find(filter_dict, limit=num_exps))
    if len(exps_list) == 0:
        return

    exp = exps_list[0]
    _, exe, config = get_command_from_exp(
        exp,
        collection.name,
        verbose=logging.root.level <= logging.DEBUG,
        unobserved=True,
        post_mortem=False,
        unresolved=unresolved,
        resolve_interpolations=resolve_interpolations,
    )
    _, exe, vscode_config = get_command_from_exp(
        exp,
        collection.name,
        verbose=logging.root.level <= logging.DEBUG,
        unobserved=True,
        post_mortem=False,
        use_json=True,
        unresolved=unresolved,
        resolve_interpolations=resolve_interpolations,
    )
    env = exp['seml'].get('conda_environment')

    console.print(Heading('First experiment'))
    logging.info(f'Executable: {exe}')
    if env is not None:
        logging.info(f'Anaconda environment: {env}')

    console.print(Heading('Arguments for VS Code debugger'))
    rich.print_json(data=['with', '--debug'] + vscode_config)
    console.print(Heading('Arguments for PyCharm debugger'))
    print('with --debug ' + get_config_overrides(config))

    console.print(Heading('Command for post-mortem debugging'))
    interpreter, exe, config = get_command_from_exp(
        exps_list[0],
        collection.name,
        verbose=logging.root.level <= logging.DEBUG,
        unobserved=True,
        post_mortem=True,
        unresolved=unresolved,
        resolve_interpolations=resolve_interpolations,
    )
    print(get_shell_command(interpreter, exe, config, env=env_dict))

    console.print(Heading('Command for remote debugging'))
    interpreter, exe, config = get_command_from_exp(
        exps_list[0],
        collection.name,
        verbose=logging.root.level <= logging.DEBUG,
        unobserved=True,
        debug_server=True,
        print_info=False,
        unresolved=unresolved,
        resolve_interpolations=resolve_interpolations,
    )
    print(get_shell_command(interpreter, exe, config, env=env_dict))

    console.print(Heading('All raw commands'))
    logging.root.setLevel(orig_level)
    for exp in exps_list:
        interpreter, exe, config = get_command_from_exp(
            exp,
            collection.name,
            verbose=logging.root.level <= logging.DEBUG,
            unresolved=unresolved,
            resolve_interpolations=resolve_interpolations,
        )
        print(get_shell_command(interpreter, exe, config, env=env_dict))
