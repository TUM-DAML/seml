import copy
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from seml.commands.manage import cancel_experiment_by_id, reset_slurm_dict
from seml.database import build_filter_dict, get_collection
from seml.experiment.command import (
    get_command_from_exp,
    get_environment_variables,
    get_shell_command,
)
from seml.experiment.sources import load_sources_from_db
from seml.settings import SETTINGS
from seml.utils import (
    assert_package_installed,
    find_jupyter_host,
    load_text_resource,
    s_if,
)
from seml.utils.errors import ArgumentError, ConfigError
from seml.utils.slurm import (
    get_cluster_name,
    get_current_slurm_array_id,
    get_current_slurm_job_id,
    get_slurm_jobs,
)

if TYPE_CHECKING:
    from pymongo.collection import Collection

States = SETTINGS.STATES
SlurmStates = SETTINGS.SLURM_STATES


def get_output_dir_path(config):
    output_dir = config['seml'].get('output_dir', '.')
    output_dir_path = str(Path(output_dir).expanduser().resolve())
    if not os.path.isdir(output_dir_path):
        raise ConfigError(f"Output directory '{output_dir_path}' does not exist.")
    return output_dir_path


def get_exp_name(exp_config, db_collection_name):
    if 'name' in exp_config['seml']:
        name = exp_config['seml']['name']
    else:
        name = db_collection_name
    return name


def set_slurm_job_name(
    sbatch_options: Dict[str, Any],
    name: str,
    exp: Dict,
    db_collection_name: str,
):
    if 'job-name' in sbatch_options:
        raise ConfigError(
            "Can't set sbatch `job-name` parameter explicitly. "
            'Use `name` parameter instead and SEML will do that for you.'
        )
    job_name = f"{name}_{exp['batch_id']}"
    sbatch_options['job-name'] = job_name
    if sbatch_options.get('comment', db_collection_name) != db_collection_name:
        raise ConfigError(
            "Can't set sbatch `comment` parameter explicitly. "
            'SEML will do that for you and set it to the collection name.'
        )
    sbatch_options['comment'] = db_collection_name


def create_slurm_options_string(slurm_options: dict, srun: bool = False):
    """
    Convert a dictionary with sbatch_options into a string that can be used in a bash script.

    Parameters
    ----------
    slurm_options: Dictionary containing the sbatch options.
    srun: Construct options for an srun command instead of an sbatch script.

    Returns
    -------
    slurm_options_str: sbatch option string.
    """
    if srun:
        option_structure = ' {prepend}{key}={value}'
    else:
        option_structure = '#SBATCH {prepend}{key}={value}\n'

    slurm_options_str = ''
    for key, value_raw in slurm_options.items():
        prepend = '-' if len(key) == 1 else '--'
        if key in ['partition', 'p'] and isinstance(value_raw, list):
            value = ','.join(value_raw)
        else:
            value = value_raw
        slurm_options_str += option_structure.format(
            prepend=prepend, key=key, value=value
        )
    return slurm_options_str


def start_sbatch_job(
    collection: 'Collection',
    exp_array: Sequence[Dict],
    slurm_options_id: int,
    sbatch_options: Dict,
    unobserved: bool = False,
    name: Optional[str] = None,
    output_dir_path: str = '.',
    max_simultaneous_jobs: Optional[int] = None,
    experiments_per_job: int = 1,
    debug_server: bool = False,
):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp_array: List[dict]
        List of chunks of experiments to run. Each chunk is a list of experiments.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    name: str
        Job name, used by Slurm job and output file.
    output_dir_path: str
        Directory (relative to home directory) where to store the slurm output files.
    sbatch_options: dict
        A dictionary that contains options for #SBATCH, e.g. {'mem': 8000} to limit the job's memory to 8,000 MB.
    max_simultaneous_jobs: int
        Maximum number of Slurm jobs running simultaneously.
    debug_server: bool
        Run jobs with a debug server.

    Returns
    -------
    None
    """
    from tempfile import NamedTemporaryFile

    seml_conf = exp_array[0]['seml']

    # Set Slurm job array options
    num_tasks = math.ceil(len(exp_array) / experiments_per_job)
    sbatch_options['array'] = f'0-{num_tasks}'
    if max_simultaneous_jobs is not None:
        sbatch_options['array'] += f'%{max_simultaneous_jobs}'

    # Set Slurm output parameter
    if 'output' in sbatch_options:
        raise ConfigError(
            "Can't set sbatch `output` Parameter explicitly. SEML will do that for you."
        )
    elif output_dir_path == '/dev/null':
        output_file = output_dir_path
    else:
        output_file = f'{output_dir_path}/{name}_%A_%a.out'
        # Ensure that the output path exists
        Path(output_file).parent.mkdir(exist_ok=True)
    sbatch_options['output'] = output_file
    sbatch_options['job-name'] = name

    # Construct sbatch options string
    sbatch_options_str = create_slurm_options_string(sbatch_options, False)

    # Construct list with all experiment IDs
    expid_strings = f"{' '.join([str(exp['_id']) for exp in exp_array])}"

    with_sources = 'source_files' in seml_conf
    use_conda_env = seml_conf.get('conda_environment')
    working_dir = seml_conf.get('working_dir', '${{SLURM_SUBMIT_DIR}}')

    # Build arguments for the prepare_experiment script
    prepare_args = ''
    if with_sources:
        prepare_args += ' --stored-sources-dir $tmpdir'
    if logging.root.level <= logging.VERBOSE:
        prepare_args += ' --verbose'
    if unobserved:
        prepare_args += ' --unobserved'
    if debug_server:
        prepare_args += ' --debug-server'

    variables = {
        'sbatch_options': sbatch_options_str,
        'working_dir': working_dir,
        'use_conda_env': str(use_conda_env is not None).lower(),
        'conda_env': seml_conf['conda_environment'] if use_conda_env else '',
        'exp_ids': expid_strings,
        'with_sources': str(with_sources).lower(),
        'db_collection_name': collection.name,
        'prepare_args': prepare_args,
        'tmp_directory': SETTINGS.TMP_DIRECTORY,
        'experiments_per_job': experiments_per_job,
    }
    variables = {
        **variables,
        'setup_command': SETTINGS.SETUP_COMMAND.format(**variables),
        'end_command': SETTINGS.END_COMMAND.format(**variables),
    }
    # Construct Slurm script
    template = load_text_resource('templates/slurm/slurm_template.sh')
    script = template.format(**variables)

    # Dump the prepared script to a temporary file
    with NamedTemporaryFile('w', dir=SETTINGS.TMP_DIRECTORY) as f:
        f.write(script)
        f.flush()

        # Sbatch the script
        try:
            output = subprocess.run(
                f'sbatch {f.name}', shell=True, check=True, capture_output=True
            ).stdout
        except subprocess.CalledProcessError as e:
            logging.error(
                f"Could not start Slurm job via sbatch. Here's the sbatch error message:\n"
                f"{e.stderr.decode('utf-8')}"
            )
            exit(1)

    # Now we just update the mongodb. So, if we are in unobserved mode, we can stop here.
    if unobserved:
        return

    slurm_array_job_id = int(output.split(b' ')[-1])
    output_file = output_file.replace('%A', str(slurm_array_job_id))
    cluster_name = get_cluster_name()
    collection.update_many(
        {'_id': {'$in': [exp['_id'] for exp in exp_array]}},
        {
            '$set': {
                'status': States.PENDING[0],
                f'slurm.{slurm_options_id}.array_id': slurm_array_job_id,
                f'slurm.{slurm_options_id}.num_tasks': num_tasks,
                f'slurm.{slurm_options_id}.output_files_template': output_file,
                f'slurm.{slurm_options_id}.sbatch_options': sbatch_options,
                'execution.cluster': cluster_name,
            }
        },
    )
    return slurm_array_job_id


def start_srun_job(collection, exp, srun_options=None, seml_arguments=None):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp: dict
        Experiment to run.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    srun_options: dict
        A dictionary that contains arguments for srun, e.g. {'mem': 8000} to limit the job's memory to 8,000 MB.
    seml_arguments: list
        A list that contains arguments for seml, e.g. ['--debug-server']

    Returns
    -------
    None
    """
    from seml.console import pause_live_widget

    with pause_live_widget():
        # Construct srun options string
        # srun will run 2 processes in parallel when ntasks is not specified. Probably because of hyperthreading.
        if 'ntasks' not in srun_options:
            srun_options['ntasks'] = 1
        srun_options_str = create_slurm_options_string(srun_options, True)

        # Set command args for job inside Slurm
        cmd_args = f"--local --sacred-id {exp['_id']} "
        cmd_args += ' '.join(seml_arguments)

        cmd = f'srun{srun_options_str} seml {collection.name} start {cmd_args}'
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(
                f"Could not start Slurm job via srun. Here's the sbatch error message:\n"
                f"{e.stderr.decode('utf-8')}"
            )
            exit(1)


def start_local_job(
    collection,
    exp,
    unobserved=False,
    post_mortem=False,
    output_dir_path='.',
    output_to_console=False,
    debug_server=False,
):
    """Run an experiment locally.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp: dict
        Experiment to run.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    post_mortem: bool
        Activate post-mortem debugging.
    output_dir_path: str
        Write the output to a file in `output_dir` given by the SEML config or in the current directory.
    output_to_console:
        Pipe all output (stdout and stderr) to the console.
    debug_server: bool
        Run job with a debug server.

    Returns
    -------
    True if job was executed successfully; False if it failed; None if job was not started because the database entry
    was not in the PENDING state.
    """
    from seml.console import pause_live_widget

    use_stored_sources = 'source_files' in exp['seml']

    interpreter, exe, config = get_command_from_exp(
        exp,
        collection.name,
        verbose=logging.root.level <= logging.VERBOSE,
        unobserved=unobserved,
        post_mortem=post_mortem,
        debug_server=debug_server,
    )
    cmd = get_shell_command(interpreter, exe, config)

    if not use_stored_sources:
        origin = Path().absolute()
        os.chdir(exp['seml']['working_dir'])

    success = True
    try:
        seml_config = exp['seml']
        slurm_config = exp['slurm']

        if use_stored_sources:
            temp_dir = os.path.join(SETTINGS.TMP_DIRECTORY, str(uuid.uuid4()))
            os.mkdir(temp_dir, mode=0o700)
            load_sources_from_db(exp, collection, to_directory=temp_dir)
            env = {'PYTHONPATH': f'{temp_dir}:$PYTHONPATH'}
            temp_exe = os.path.join(temp_dir, exe)
            # update the command to use the temp dir
            cmd = get_shell_command(interpreter, temp_exe, config, env=env)

        if output_dir_path:
            exp_name = get_exp_name(exp, collection.name)
            output_file = f"{output_dir_path}/{exp_name}_{exp['_id']}.out"
            if not unobserved:
                collection.update_one(
                    {'_id': exp['_id']}, {'$set': {'seml.output_file': output_file}}
                )
            if output_to_console:
                # redirect output to logfile AND output to console. See https://stackoverflow.com/a/34604684.
                # Alternatively, we could go with subprocess.Popen, but this could conflict with pdb.
                cmd = f'{cmd} 2>&1 | tee -a {output_file}'

        if seml_config.get('conda_environment') is not None:
            cmd = (
                f". $(conda info --base)/etc/profile.d/conda.sh "
                f"&& conda activate {seml_config['conda_environment']} "
                f"&& {cmd} "
                f"&& conda deactivate"
            )

        if not unobserved:
            execution = {'cluster': 'local'}
            if 'SLURM_JOBID' in os.environ:
                execution['array_id'] = os.environ['SLURM_JOBID']
                execution['task_id'] = 0
            collection.update_one(
                {'_id': exp['_id']},
                {'$set': {'execution': execution}},
            )

        logging.verbose(f'Running the following command:\n {cmd}')

        if output_dir_path:
            if output_to_console:
                # Let's pause the live widget so we can actually see the output.
                with pause_live_widget():
                    subprocess.run(cmd, shell=True, check=True)
            else:  # redirect output to logfile
                with open(output_file, 'w') as log_file:
                    subprocess.run(
                        cmd, shell=True, stderr=log_file, stdout=log_file, check=True
                    )
        else:
            with pause_live_widget():
                subprocess.run(cmd, shell=True, check=True)

    except subprocess.CalledProcessError:
        success = False
    except IOError:
        logging.error(f'Log file {output_file} could not be written.')
        # Since Sacred is never called in case of I/O error, we need to set the experiment state manually.
        if not unobserved:
            collection.update_one(
                filter={'_id': exp['_id']},
                update={'$set': {'status': States.FAILED[0]}},
            )
        success = False
    finally:
        if use_stored_sources and 'temp_dir' in locals():
            # clean up temp directory
            shutil.rmtree(temp_dir)
        if not use_stored_sources:
            os.chdir(origin)

    return success


def chunk_list(exps):
    """
    Divide experiments by batch id as these will be submitted jointly.
    This assumes constant Slurm settings per batch (which should be the case if MongoDB wasn't edited manually).

    Parameters
    ----------
    exps: list[dict]
        List of dictionaries containing the experiment settings as saved in the MongoDB

    Returns
    -------
    exp_chunks: list
    """
    from collections import defaultdict

    exp_chunks = defaultdict(list)
    for exp in exps:
        exp_chunks[exp['batch_id']].append(exp)
    return list(exp_chunks.values())


def prepare_staged_experiments(
    collection, filter_dict=None, num_exps=0, set_to_pending=True
):
    """
    Load experiments from the input MongoDB collection, and prepare them for running.
    If the filter_dict contains no status or ID, we filter the status by STAGED.
    If set_to_pending is True, we set their status to PENDING.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection with STAGED experiments.
    filter_dict: dict
        Optional dict with custom database filters.
    num_exps: int
        Only set <num_exps> experiments' state to PENDING. If 0, set all STAGED experiments to PENDING.
    set_to_pending: bool
        Whether to update the database entries to status PENDING.

    Returns
    -------
    The filtered list of database entries.
    """
    if filter_dict is None:
        filter_dict = {}

    query_dict = copy.deepcopy(filter_dict)
    if '_id' not in query_dict and 'status' not in query_dict:
        query_dict['status'] = {'$in': States.STAGED}

    experiments = list(collection.find(query_dict, limit=num_exps))

    if set_to_pending:
        update_dict = {'$set': {'status': States.PENDING[0]}}

        if num_exps > 0:
            # Set only those experiments to PENDING which will be run.
            collection.update_many(
                {'_id': {'$in': [e['_id'] for e in experiments]}}, update_dict
            )
        else:
            collection.update_many(query_dict, update_dict)

        nexps_set = len(experiments)
        logging.info(f'Setting {nexps_set} experiment{s_if(nexps_set)} to pending.')

    return experiments


def add_to_slurm_queue(
    collection,
    exps_list,
    unobserved=False,
    post_mortem=False,
    output_to_file=True,
    output_to_console=False,
    srun=False,
    debug_server=False,
):
    """
    Send the input list of experiments to the Slurm system for execution.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exps_list: list of dicts
        The list of database entries corresponding to experiments to be executed.
    unobserved: bool
        Whether to suppress observation by Sacred observers.
    post_mortem: bool
        Activate post-mortem debugging.
    output_to_file: bool
        Whether to capture output in a logfile.
    output_to_console: bool
        Whether to capture output in the console. This is currently not supported for Slurm jobs and will raise an
        error if set to True.
    srun: bool
        Run jobs interactively via srun instead of using sbatch.
    debug_server: bool
        Run jobs with a debug server.

    Returns
    -------
    None
    """

    nexps = len(exps_list)
    exp_arrays = chunk_list(exps_list)
    narrays = 0
    array_ids = []

    for exp_array in exp_arrays:
        slurm_options = exp_array[0]['slurm']
        default_sbatch_options = slurm_options[0]['sbatch_options']
        job_name = get_exp_name(exp_array[0], collection.name)
        if srun:
            set_slurm_job_name(
                default_sbatch_options,
                job_name,
                exp_array[0],
                collection.name,
            )
            assert len(exp_array) == 1
            seml_arguments = []
            seml_arguments.append('--debug')
            if post_mortem:
                seml_arguments.append('--post-mortem')
            if output_to_console:
                seml_arguments.append('--output-to-console')
            if not output_to_file:
                seml_arguments.append('--no-file-output')
            if debug_server:
                seml_arguments.append('--debug-server')
            start_srun_job(
                collection,
                exp_array[0],
                srun_options=default_sbatch_options,
                seml_arguments=seml_arguments,
            )
            narrays += 1
        else:
            if output_to_file:
                output_dir_path = get_output_dir_path(exp_array[0])
            else:
                output_dir_path = '/dev/null'
            assert not post_mortem
            for slurm_options_id, slurm_option in enumerate(slurm_options):
                set_slurm_job_name(
                    slurm_option['sbatch_options'],
                    job_name,
                    exp_array[0],
                    collection.name,
                )
                array_id = start_sbatch_job(
                    collection,
                    exp_array,
                    slurm_options_id,
                    slurm_option['sbatch_options'],
                    unobserved,
                    name=job_name,
                    output_dir_path=output_dir_path,
                    max_simultaneous_jobs=slurm_option.get('max_simultaneous_jobs'),
                    experiments_per_job=slurm_option.get('experiments_per_job', 1),
                    debug_server=debug_server,
                )
                array_ids.append(array_id)
                narrays += 1
    logging.info(
        f'Started {nexps} experiment{s_if(nexps)} in '
        f'{narrays} Slurm job array{s_if(narrays)}: {", ".join(map(str, array_ids))}'
    )


def check_compute_node():
    if os.uname()[1] in SETTINGS.LOGIN_NODE_NAMES:
        raise ArgumentError(
            'Refusing to run a compute experiment on a login node. '
            'Please use Slurm or a compute node.'
        )


def start_local_worker(
    collection,
    num_exps=0,
    filter_dict=None,
    unobserved=False,
    post_mortem=False,
    steal_slurm=False,
    output_to_console=False,
    output_to_file=True,
    gpus=None,
    cpus=None,
    environment_variables=None,
    debug_server=False,
):
    """
    Start a local worker on the current machine that pulls PENDING experiments from the database and executes them.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    num_exps: int
        The maximum number of experiments run by this worker before terminating.
    filter_dict: dict
        Optional dict with custom database filters.
    unobserved: bool
        Whether to suppress observation by Sacred observers.
    post_mortem: bool
        Activate post-mortem debugging.
    steal_slurm: bool
        If True, the local worker will also execute jobs waiting for execution in Slurm.
    output_to_console: bool
        Whether to capture output in the console.
    output_to_file: bool
        Whether to capture output in a logfile.
    gpus: str
        Comma-separated list of GPU IDs to be used by this worker (e.g., "2,3"). Will be passed to CUDA_VISIBLE_DEVICES.
    cpus: int
        Number of CPU cores to be used by this worker. If None, use all cores.
    environment_variables: dict
        Optional dict of additional environment variables to be set.
    debug_server: bool
        Run jobs with a debug server.

    Returns
    -------
    None
    """
    from rich.progress import Progress

    from seml.console import pause_live_widget, prompt

    check_compute_node()

    if 'SLURM_JOBID' in os.environ:
        node_str = subprocess.run(
            'squeue -j ${SLURM_JOBID} -O nodelist:1000',
            shell=True,
            check=True,
            capture_output=True,
        ).stdout
        node_id = node_str.decode('utf-8').split('\n')[1].strip()
        logging.info(f'SLURM assigned me the node(s): {node_id}')

    if num_exps > 0:
        logging.info(
            f'Starting local worker thread that will run up to {num_exps} experiment{s_if(num_exps)}, '
            f'or until no pending experiments remain.'
        )
    else:
        logging.info(
            'Starting local worker thread that will run experiments until no pending experiments remain.'
        )
        num_exps = int(1e30)

    os.environ.update(get_environment_variables(gpus, cpus, environment_variables))

    num_exceptions = 0
    jobs_counter = 0

    exp_query = {}
    if not unobserved:
        exp_query['status'] = {'$in': States.PENDING}
    if not steal_slurm:
        exp_query['slurm'] = {'$elemMatch': {'array_id': {'$exists': False}}}

    exp_query.update(filter_dict)

    with pause_live_widget():
        with Progress(auto_refresh=False) as progress:
            task = progress.add_task('Running experiments...', total=None)
            while collection.count_documents(exp_query) > 0 and jobs_counter < num_exps:
                if unobserved:
                    exp = collection.find_one(exp_query)
                else:
                    exp = collection.find_one_and_update(
                        exp_query,
                        {
                            '$set': {
                                'status': States.RUNNING[0],
                                'execution.cluster': 'local',
                            }
                        },
                    )
                if exp is None:
                    continue
                if 'array_id' in exp.get('execution', {}):
                    # Clean up MongoDB entry
                    slurm_ids = {
                        'array_id': exp['execution']['array_id'],
                        'task_id': exp['execution']['task_id'],
                    }
                    reset_slurm_dict(exp)
                    collection.replace_one({'_id': exp['_id']}, exp, upsert=False)

                    # Cancel Slurm job; after cleaning up to prevent race conditions
                    if prompt(
                        f"SLURM is currently executing experiment {exp['_id']}, do you want to cancel the SLURM job?",
                        type=bool,
                    ):
                        cancel_experiment_by_id(
                            collection,
                            exp['_id'],
                            set_interrupted=False,
                            slurm_dict=slurm_ids,
                        )

                progress.console.print(
                    f"current id : {exp['_id']}, failed={num_exceptions}/{jobs_counter} experiments"
                )

                # Add newline if we need to avoid tqdm's output
                if (
                    debug_server
                    or output_to_console
                    or logging.root.level <= logging.VERBOSE
                ):
                    print(file=sys.stderr)

                if output_to_file:
                    output_dir_path = get_output_dir_path(exp)
                else:
                    output_dir_path = None
                try:
                    success = start_local_job(
                        collection=collection,
                        exp=exp,
                        unobserved=unobserved,
                        post_mortem=post_mortem,
                        output_dir_path=output_dir_path,
                        output_to_console=output_to_console,
                        debug_server=debug_server,
                    )
                    if success is False:
                        num_exceptions += 1
                except KeyboardInterrupt:
                    logging.info('Caught KeyboardInterrupt signal. Aborting.')
                    exit(1)
                jobs_counter += 1
                progress.advance(task)
                # tq.set_postfix(current_id=exp['_id'], failed=f"{num_exceptions}/{jobs_counter} experiments")


def start_experiments(
    db_collection_name,
    local,
    sacred_id,
    batch_id,
    filter_dict,
    num_exps,
    post_mortem,
    debug,
    debug_server,
    output_to_console,
    no_file_output,
    steal_slurm,
    no_worker,
    set_to_pending=True,
    worker_gpus=None,
    worker_cpus=None,
    worker_environment_vars=None,
):
    output_to_file = not no_file_output
    launch_worker = not no_worker

    if debug or debug_server:
        num_exps = 1
        unobserved = True
        post_mortem = True
        output_to_console = True
        srun = True
        logging.root.setLevel(logging.VERBOSE)
    else:
        unobserved = False
        srun = False

    if local:
        check_compute_node()

    if not local:
        local_kwargs = {
            '--no-worker': no_worker,
            '--steal-slurm': steal_slurm,
            '--worker-gpus': worker_gpus,
            '--worker-cpus': worker_cpus,
            '--worker-environment-vars': worker_environment_vars,
        }
        for key, val in local_kwargs.items():
            if val:
                raise ArgumentError(
                    f"The argument '{key}' only works in local mode, not in Slurm mode."
                )
    if not local and not srun:
        non_sbatch_kwargs = {
            '--post-mortem': post_mortem,
            '--output-to-console': output_to_console,
        }
        for key, val in non_sbatch_kwargs.items():
            if val:
                raise ArgumentError(
                    f"The argument '{key}' does not work in regular Slurm mode. "
                    "Remove the argument or use '--debug'."
                )

    if unobserved:
        set_to_pending = False

    filter_dict = build_filter_dict([], batch_id, filter_dict, sacred_id)

    collection = get_collection(db_collection_name)

    staged_experiments = prepare_staged_experiments(
        collection=collection,
        filter_dict=filter_dict,
        num_exps=num_exps,
        set_to_pending=set_to_pending and local,
    )

    if debug_server:
        use_stored_sources = 'source_files' in staged_experiments[0]['seml']
        if use_stored_sources:
            raise ArgumentError(
                'Cannot use a debug server with source code that is loaded from the MongoDB. '
                'Use the `--no-code-checkpoint` option when adding the experiment.'
            )

    if not local:
        add_to_slurm_queue(
            collection=collection,
            exps_list=staged_experiments,
            unobserved=unobserved,
            post_mortem=post_mortem,
            output_to_file=output_to_file,
            output_to_console=output_to_console,
            srun=srun,
            debug_server=debug_server,
        )
    elif launch_worker:
        start_local_worker(
            collection=collection,
            num_exps=num_exps,
            filter_dict=filter_dict,
            unobserved=unobserved,
            post_mortem=post_mortem,
            steal_slurm=steal_slurm,
            output_to_console=output_to_console,
            output_to_file=output_to_file,
            gpus=worker_gpus,
            cpus=worker_cpus,
            environment_variables=worker_environment_vars,
            debug_server=debug_server,
        )


def start_jupyter_job(
    sbatch_options: Optional[dict] = None,
    conda_env: Optional[str] = None,
    lab: bool = False,
):
    if lab:
        assert_package_installed(
            'jupyterlab',
            '`start-jupyter --lab` requires `jupyterlab` (e.g. `pip install jupyterlab`)',
        )
    else:
        assert_package_installed(
            'notebook',
            '`start-jupyter` requires `notebook` (e.g. `pip install notebook`)',
        )

    sbatch_options = sbatch_options if sbatch_options is not None else {}
    sbatch_options_merged = SETTINGS.SLURM_DEFAULT['sbatch_options']
    sbatch_options_merged.update(SETTINGS.SBATCH_OPTIONS_TEMPLATES.JUPYTER)
    sbatch_options_merged.update(sbatch_options)
    # Construct sbatch options string
    sbatch_options_str = create_slurm_options_string(sbatch_options_merged)

    template = load_text_resource('templates/slurm/jupyter_template.sh')

    script = template.format(
        sbatch_options=sbatch_options_str,
        use_conda_env=str(conda_env is not None).lower(),
        conda_env=conda_env,
        notebook_or_lab=' notebook' if not lab else '-lab',
    )

    path = os.path.join(SETTINGS.TMP_DIRECTORY, f'{uuid.uuid4()}.sh')
    with open(path, 'w') as f:
        f.write(script)

    try:
        output = subprocess.run(
            f'sbatch {path}', shell=True, check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Could not start Slurm job via sbatch. Here's the sbatch error message:\n"
            f"{e.stderr.decode('utf-8')}"
        )
        os.remove(path)
        exit(1)
    os.remove(path)

    slurm_array_job_id = int(output.split(b' ')[-1])
    logging.info(f'Queued Jupyter instance in Slurm job with ID {slurm_array_job_id}.')

    job_output = subprocess.run(
        f'scontrol show job {slurm_array_job_id} -o',
        shell=True,
        check=True,
        capture_output=True,
    ).stdout
    job_output_results = job_output.decode('utf-8').split(' ')
    job_info_dict = {
        x.split('=')[0]: x.split('=')[1] for x in job_output_results if '=' in x
    }
    log_file = job_info_dict['StdOut']

    logging.info(f"The job's log-file is '{log_file}'.")
    logging.info(
        'Waiting for start-up to fetch the machine and port of the Jupyter instance... '
        '(ctrl-C to cancel fetching)'
    )

    while job_info_dict['JobState'] in SlurmStates.PENDING:
        job_output = subprocess.run(
            f'scontrol show job {slurm_array_job_id} -o',
            shell=True,
            check=True,
            capture_output=True,
        ).stdout
        job_output_results = job_output.decode('utf-8').split(' ')
        job_info_dict = {
            x.split('=')[0]: x.split('=')[1] for x in job_output_results if '=' in x
        }
        time.sleep(1)
    if job_info_dict['JobState'] not in SlurmStates.RUNNING:
        logging.error(
            f"Slurm job failed. See log-file '{log_file}' for more information."
        )
        exit(1)

    logging.info('Slurm job is running. Jupyter instance is starting up...')

    # Obtain list of hostnames to addresses
    url_str, known_host = find_jupyter_host(log_file, True)
    if known_host is None:
        logging.error(
            f"Could not fetch the host and port of the Jupyter instance. Here's the raw output: \n"
            f'{url_str}'
        )
        exit(1)
    if not known_host:
        logging.warning('Host unknown to SLURM.')
    logging.info(f"Start-up completed. The Jupyter instance is running at '{url_str}'.")
    logging.info(f"To stop the job, run 'scancel {slurm_array_job_id}'.")


def get_experiment_to_prepare(
    collection: 'Collection',
    exp_id: int,
    unobserved: bool,
):
    """
    Retrieves the experiment the pending experiment with the given ID from the database and sets its state to RUNNING.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp_id: int
        The ID of the experiment to retrieve.
    unobserved: bool
        Whether to suppress observation by Sacred observers.

    Returns
    -------
    The experiment document if it was found and set to RUNNING, None otherwise.
    """
    if unobserved:
        # If the experiment has no observer, we just pull the configuration but never update the database.
        return collection.find_one({'_id': exp_id})
    # This returns the document as it was BEFORE the update. So we first have to check whether its state was
    # PENDING. This is to avoid race conditions, since find_one_and_update is an atomic operation.
    slurm_array_id, slurm_task_id = get_current_slurm_array_id()
    if slurm_array_id is not None and slurm_task_id is not None:
        # We're running in SLURM.
        # Check if the job executing is this one.
        cluster_name = get_cluster_name()
        job_filter = {
            '_id': exp_id,
            'execution.array_id': int(slurm_array_id),
            'execution.task_id': int(slurm_task_id),
            'execution.cluster': cluster_name,
        }
        # Either take the experiment if it is pending or if it is the one being executed.
        # The latter case is important for multi-node jobs.
        return collection.find_one(job_filter)
    # Steal slurm case
    return collection.find_one({'_id': exp_id})


def claim_experiment(db_collection_name: str, exp_ids: Sequence[int]):
    """
    Claim an experiment for execution by setting its state to RUNNING.

    Parameters
    ----------
    db_collection_name: str
        The name of the MongoDB collection containing the experiments.
    exp_ids: Sequence[int]
        The IDs of the experiments to claim.

    Exit Codes
    ----------
    0: Experiment claimed successfully
    3: Experiment not in the database

    Stdout
    -------
    The ID of the claimed experiment.
    """
    collection = get_collection(db_collection_name)
    array_id, task_id = get_current_slurm_array_id()
    if array_id is not None and task_id is not None:
        # We are running in slurm
        array_id, task_id = int(array_id), int(task_id)
        cluster_name = get_cluster_name()
        update = {
            'execution.array_id': array_id,
            'execution.task_id': task_id,
            'execution.cluster': cluster_name,
        }
        exp = collection.find_one_and_update(
            {'_id': {'$in': list(exp_ids)}, 'status': {'$in': States.PENDING}},
            {'$set': {'status': States.RUNNING[0], **update}},
            {'_id': 1, 'slurm': 1},
        )
        # Set slurm output file
        for s_conf in exp['slurm']:
            if s_conf['array_id'] == array_id:
                output_file = s_conf['output_files_template']
                output_file = output_file.replace('%a', str(task_id))
                collection.update_one(
                    {'_id': exp['_id']},
                    {'$set': {'execution.slurm_output_file': output_file}},
                )
    else:
        # Steal slurm
        exp = collection.find_one_and_update(
            {'_id': {'$in': list(exp_ids)}, 'status': {'$in': States.PENDING}},
            {'$set': {'status': States.RUNNING[0], 'execution.cluster': 'local'}},
            {'_id': 1},
        )
    if exp is None:
        exit(3)
    print(exp['_id'])
    exit(0)


def prepare_experiment(
    db_collection_name: str,
    exp_id: int,
    verbose: bool,
    unobserved: bool,
    post_mortem: bool,
    stored_sources_dir: Optional[str],
    debug_server: bool,
):
    """
    Prepare an experiment for execution by printing the command that should be executed.
    If stored_sources_dir is set, the source files are loaded from the database and stored in the directory.

    Parameters
    ----------
    db_collection_name: str
        The name of the MongoDB collection containing the experiments.
    exp_id: int
        The ID of the experiment to prepare.
    verbose: bool
        Whether to print the command verbosely.
    unobserved: bool
        Whether to suppress observation by Sacred observers.
    post_mortem: bool
        Activate post-mortem debugging.
    stored_sources_dir: str
        The directory where the source files are stored.
    debug_server: bool
        Run job with a debug server.

    Exit Codes
    ----------
    0: Preparation successful
    3: Experiment is not in the database
    4: Experiment is in the database but not in the PENDING state

    Returns
    -------
    None
    """
    from sacred.randomness import get_seed

    from seml.utils.multi_process import (
        is_local_main_process,
        is_main_process,
        is_running_in_multi_process,
    )

    # This process should only be executed once per node, so if we are not the main
    # process per node, we directly exit.
    if not is_local_main_process():
        exit(0)

    collection = get_collection(db_collection_name)
    exp = get_experiment_to_prepare(collection, exp_id, unobserved)

    if exp is None:
        # These exit codes will be handled in the bash script
        if collection.count_documents({'_id': exp_id}) == 0:
            exit(4)
        else:
            exit(3)

    if stored_sources_dir:
        os.makedirs(stored_sources_dir, exist_ok=True)
        if not os.listdir(stored_sources_dir):
            assert (
                'source_files' in exp['seml']
            ), '--stored-sources-dir is set but no source files are stored in the database.'
            load_sources_from_db(exp, collection, to_directory=stored_sources_dir)

    # The remaining part (updateing MongoDB & printing the python command) is only executed by the main process.
    if not is_main_process():
        exit(0)

    # If we run in a multi task environment, we want to make sure that the seed is fixed once and
    # all tasks start with the same seed. Otherwise, one could not reproduce the experiment as the
    # seed would change on the child nodes. It is up to the user to distribute seeds if needed.
    if is_running_in_multi_process():
        if SETTINGS.CONFIG_KEY_SEED not in exp['config']:
            exp['config'][SETTINGS.CONFIG_KEY_SEED] = get_seed()

    # Let's generate a output file
    output_dir = get_output_dir_path(exp)
    try:
        job_info = get_slurm_jobs(get_current_slurm_job_id())[0]
        name = job_info['JobName']
        array_id, task_id = get_current_slurm_array_id()
        name = f'{name}_{array_id}_{task_id}'
    except Exception:
        name = str(uuid.uuid4())
    output_file = f'{output_dir}/{name}_{exp["_id"]}.out'

    interpreter, exe, config = get_command_from_exp(
        exp,
        db_collection_name,
        verbose=verbose,
        unobserved=unobserved,
        post_mortem=post_mortem,
        debug_server=debug_server,
    )
    cmd = get_shell_command(interpreter, exe, config)
    cmd_unresolved = get_shell_command(
        *get_command_from_exp(
            exp,
            db_collection_name,
            verbose=verbose,
            unobserved=unobserved,
            post_mortem=post_mortem,
            debug_server=debug_server,
            unresolved=True,
        )
    )
    updates = {
        'seml.command': cmd,
        'seml.command_unresolved': cmd_unresolved,
        'seml.output_file': output_file,
    }

    if stored_sources_dir:
        temp_dir = stored_sources_dir
        # Store the temp dir for debugging purposes
        updates['seml.temp_dir'] = temp_dir
        cmd = get_shell_command(interpreter, os.path.join(temp_dir, exe), config)

    if not unobserved:
        collection.update_one({'_id': exp['_id']}, {'$set': updates})

    # Print the command to be ran.
    print(f'{cmd} > {output_file} 2>&1')
    # We exit with 0 to signal that the preparation was successful.
    exit(0)
