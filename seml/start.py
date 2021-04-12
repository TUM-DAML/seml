import os
import sys
import subprocess
import logging
import numpy as np
import shutil
import pkg_resources
from pathlib import Path
from tqdm.autonotebook import tqdm
import time

from seml.database import get_collection, build_filter_dict
from seml.sources import load_sources_from_db
from seml.utils import s_if
from seml.network import find_free_port
from seml.settings import SETTINGS
from seml.manage import cancel_experiment_by_id, reset_slurm_dict
from seml.errors import ConfigError, ArgumentError, MongoDBError

States = SETTINGS.STATES


def get_command_from_exp(exp, db_collection_name, verbose=False, unobserved=False,
                         post_mortem=False, debug=False, debug_server=False):
    if 'executable' not in exp['seml']:
        raise MongoDBError(f"No executable found for experiment {exp['_id']}. Aborting.")
    exe = exp['seml']['executable']
    if 'executable_relative' in exp['seml']:  # backwards compatibility
        exe = exp['seml']['executable_relative']

    config = exp['config']
    config['db_collection'] = db_collection_name
    if not unobserved:
        config['overwrite'] = exp['_id']
    config_strings = [f'{key}="{val}"' if type(val) != str else f'{key}="\'{val}\'"' for key, val in config.items()]
    if not verbose:
        config_strings.append("--force")
    if unobserved:
        config_strings.append("--unobserved")
    if post_mortem:
        config_strings.append("--pdb")
    if debug:
        config_strings.append("--debug")

    if debug_server:
        ip_address, port = find_free_port()
        logging.info(f"Starting debug server with IP {ip_address} and port {port}. "
                     f"Experiment will wait for a debug client to attach.")
        interpreter = f"python -m debugpy --listen {ip_address}:{port} --wait-for-client"
    else:
        interpreter = "python"

    return interpreter, exe, config_strings


def get_output_dir_path(config):
    if 'output_dir' in config['slurm']:
        logging.warning("'output_dir' has moved from 'slurm' to 'seml'. Please adapt your YAML accordingly"
                        "by moving the 'output_dir' parameter from 'slurm' to 'seml'.")
        output_dir = config['slurm']['output_dir']
    elif 'output_dir' in config['seml']:
        output_dir = config['seml']['output_dir']
    else:
        output_dir = '.'
    output_dir_path = str(Path(output_dir).expanduser().resolve())
    if not os.path.isdir(output_dir_path):
        raise ConfigError(f"Output directory '{output_dir_path}' does not exist.")
    return output_dir_path


def get_exp_name(exp_config, db_collection_name):
    if 'name' in exp_config['slurm']:
        logging.warning("'name' has moved from 'slurm' to 'seml'. Please adapt your YAML accordingly"
                        "by moving the 'name' parameter from 'slurm' to 'seml'.")
        name = exp_config['slurm']['name']
    elif 'name' in exp_config['seml']:
        name = exp_config['seml']['name']
    else:
        name = db_collection_name
    return name


def set_slurm_job_name(sbatch_options, name, exp):
    if 'job-name' in sbatch_options:
        raise ConfigError("Can't set sbatch `job-name` parameter explicitly. "
                          "Use `name` parameter instead and SEML will do that for you.")
    job_name = f"{name}_{exp['batch_id']}"
    sbatch_options['job-name'] = job_name


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
        option_structure = " {prepend}{key}={value}"
    else:
        option_structure = "#SBATCH {prepend}{key}={value}\n"

    slurm_options_str = ""
    for key, value_raw in slurm_options.items():
        prepend = '-' if len(key) == 1 else '--'
        if key in ['partition', 'p'] and isinstance(value_raw, list):
            value = ','.join(value_raw)
        else:
            value = value_raw
        slurm_options_str += option_structure.format(prepend=prepend, key=key, value=value)
    return slurm_options_str


def start_sbatch_job(collection, exp_array, unobserved=False, name=None,
                     output_dir_path=".", sbatch_options=None, max_jobs_per_batch=None,
                     debug_server=False):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp_array: List[List[dict]]
        List of chunks of experiments to run. Each chunk is a list of experiments.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    name: str
        Job name, used by Slurm job and output file.
    output_dir_path: str
        Directory (relative to home directory) where to store the slurm output files.
    sbatch_options: dict
        A dictionary that contains options for #SBATCH, e.g. {'mem': 8000} to limit the job's memory to 8,000 MB.
    max_jobs_per_batch: int
        Maximum number of Slurm jobs running per experiment batch.
    debug_server: bool
        Run jobs with a debug server.

    Returns
    -------
    None
    """

    # Set Slurm job array options
    sbatch_options['array'] = f"0-{len(exp_array) - 1}"
    if max_jobs_per_batch is not None:
        sbatch_options['array'] += f"%{max_jobs_per_batch}"

    # Set Slurm output parameter
    if 'output' in sbatch_options:
        raise ConfigError(f"Can't set sbatch `output` Parameter explicitly. SEML will do that for you.")
    elif output_dir_path == "/dev/null":
        output_file = output_dir_path
    else:
        output_file = f'{output_dir_path}/{name}_%A_%a.out'
    sbatch_options['output'] = output_file

    # Construct sbatch options string
    sbatch_options_str = create_slurm_options_string(sbatch_options, False)

    # Construct chunked list with all experiment IDs
    expid_strings = [('"' + ';'.join([str(exp['_id']) for exp in chunk]) + '"') for chunk in exp_array]

    with_sources = ('source_files' in exp_array[0][0]['seml'])
    use_conda_env = ('conda_environment' in exp_array[0][0]['seml']
                     and exp_array[0][0]['seml']['conda_environment'] is not None)

    # Construct Slurm script
    template = pkg_resources.resource_string(__name__, "slurm_template.sh").decode("utf-8")
    prepare_experiment_script = pkg_resources.resource_string(__name__, "prepare_experiment.py").decode("utf-8")
    prepare_experiment_script = prepare_experiment_script.replace("'", "'\\''")
    if 'working_dir' in exp_array[0][0]['seml']:
        working_dir = exp_array[0][0]['seml']['working_dir']
    else:
        working_dir = "${{SLURM_SUBMIT_DIR}}"

    script = template.format(
            sbatch_options=sbatch_options_str,
            working_dir=working_dir,
            use_conda_env=str(use_conda_env).lower(),
            conda_env=exp_array[0][0]['seml']['conda_environment'] if use_conda_env else "",
            exp_ids=' '.join(expid_strings),
            with_sources=str(with_sources).lower(),
            prepare_experiment_script=prepare_experiment_script,
            db_collection_name=collection.name,
            sources_argument="--stored-sources-dir $tmpdir" if with_sources else "",
            verbose=logging.root.level <= logging.VERBOSE,
            unobserved=unobserved,
            debug_server=debug_server,
    )

    random_int = np.random.randint(0, 999999)
    path = f"/tmp/{random_int}.sh"
    while os.path.exists(path):
        random_int = np.random.randint(0, 999999)
        path = f"/tmp/{random_int}.sh"
    with open(path, "w") as f:
        f.write(script)

    output = subprocess.run(f'sbatch {path}', shell=True, check=True, capture_output=True).stdout

    slurm_array_job_id = int(output.split(b' ')[-1])
    for task_id, chunk in enumerate(exp_array):
        for exp in chunk:
            if not unobserved:
                collection.update_one(
                        {'_id': exp['_id']},
                        {'$set': {
                            'status': States.PENDING[0],
                            'slurm.array_id': slurm_array_job_id,
                            'slurm.task_id': task_id,
                            'slurm.sbatch_options': sbatch_options,
                            'seml.output_file': f"{output_dir_path}/{name}_{slurm_array_job_id}_{task_id}.out"}})
            logging.verbose(f"Started experiment with array job ID {slurm_array_job_id}, task ID {task_id}.")
    os.remove(path)


def start_srun_job(collection, exp, unobserved=False,
                   srun_options=None, seml_arguments=None):
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

    # Construct srun options string
    # srun will run 2 processes in parallel when ntasks is not specified. Probably because of hyperthreading.
    if 'ntasks' not in srun_options:
        srun_options['ntasks'] = 1
    srun_options_str = create_slurm_options_string(srun_options, True)

    if not unobserved:
        collection.update_one(
                {'_id': exp['_id']},
                {'$set': {'slurm.sbatch_options': srun_options}})

    # Set command args for job inside Slurm
    cmd_args = f"--local --sacred-id {exp['_id']} "
    cmd_args += ' '.join(seml_arguments)

    cmd = (f"srun{srun_options_str} seml {collection.name} start {cmd_args}")
    subprocess.run(cmd, shell=True, check=True)


def start_local_job(collection, exp, unobserved=False, post_mortem=False,
                    output_dir_path='.', output_to_console=False, debug_server=False):
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

    use_stored_sources = ('source_files' in exp['seml'])

    interpreter, exe, config = get_command_from_exp(exp, collection.name,
                                                    verbose=logging.root.level <= logging.VERBOSE,
                                                    unobserved=unobserved, post_mortem=post_mortem,
                                                    debug_server=debug_server)
    if not use_stored_sources:
        os.chdir(exp['seml']['working_dir'])

    cmd = f"{interpreter} {exe} with {' '.join(config)}"

    success = True
    try:
        seml_config = exp['seml']
        slurm_config = exp['slurm']

        if use_stored_sources:
            random_int = np.random.randint(0, 999999)
            temp_dir = f"/tmp/{random_int}"
            while os.path.exists(temp_dir):
                random_int = np.random.randint(0, 999999)
                temp_dir = f"/tmp/{random_int}"
            os.mkdir(temp_dir, mode=0o700)
            load_sources_from_db(exp, collection, to_directory=temp_dir)
            # update the command to use the temp dir
            cmd = f'PYTHONPATH="{temp_dir}:$PYTHONPATH" {interpreter} {temp_dir}/{exe} with {" ".join(config)}'

        if output_dir_path:
            exp_name = get_exp_name(exp, collection.name)
            output_file = f"{output_dir_path}/{exp_name}_{exp['_id']}.out"
            if not unobserved:
                collection.update_one({'_id': exp['_id']}, {"$set": {"seml.output_file": output_file}})
            if output_to_console:
                # redirect output to logfile AND output to console. See https://stackoverflow.com/a/34604684.
                # Alternatively, we could go with subprocess.Popen, but this could conflict with pdb.
                cmd = f"{cmd} 2>&1 | tee -a {output_file}"

        if 'conda_environment' in seml_config and seml_config['conda_environment'] is not None:
            cmd = (f". $(conda info --base)/etc/profile.d/conda.sh "
                   f"&& conda activate {seml_config['conda_environment']} "
                   f"&& {cmd} "
                   f"&& conda deactivate")

        if 'SLURM_JOBID' in os.environ and not unobserved:
            collection.update_one(
                    {'_id': exp['_id']},
                    {'$set': {
                        'slurm.array_id': os.environ['SLURM_JOBID'],
                        'slurm.task_id': 0}})

        logging.verbose(f'Running the following command:\n {cmd}')

        if output_dir_path:
            if output_to_console:
                subprocess.run(cmd, shell=True, check=True)
            else:  # redirect output to logfile
                with open(output_file, "w") as log_file:
                    subprocess.run(cmd, shell=True, stderr=log_file, stdout=log_file, check=True)
        else:
            subprocess.run(cmd, shell=True, check=True)

    except subprocess.CalledProcessError:
        success = False
    except IOError:
        logging.error(f"Log file {output_file} could not be written.")
        # Since Sacred is never called in case of I/O error, we need to set the experiment state manually.
        if not unobserved:
            collection.update_one(filter={'_id': exp['_id']},
                                  update={'$set': {'status': States.FAILED[0]}})
        success = False
    finally:
        if use_stored_sources and 'temp_dir' in locals():
            # clean up temp directory
            shutil.rmtree(temp_dir)
    return success


def chunk_list(exps):
    """
    Divide experiments into chunks of `experiments_per_job` that will be run in parallel in one job.
    This assumes constant Slurm settings per batch (which should be the case if MongoDB wasn't edited manually).

    Parameters
    ----------
    exps: list[dict]
        List of dictionaries containing the experiment settings as saved in the MongoDB

    Returns
    -------
    exp_chunks: list
    """
    batch_idx = [exp['batch_id'] for exp in exps]
    unique_batch_idx = np.unique(batch_idx)
    exp_chunks = []
    for batch in unique_batch_idx:
        idx = [i for i, batch_id in enumerate(batch_idx)
               if batch_id == batch]
        size = exps[idx[0]]['slurm']['experiments_per_job']
        exp_chunks.extend(([exps[i] for i in idx[pos:pos + size]] for pos in range(0, len(idx), size)))
    return exp_chunks


def batch_chunks(exp_chunks):
    """
    Divide chunks of experiments into Slurm job arrays with one experiment batch per array.
    Each array is started together.
    This assumes constant Slurm settings per batch (which should be the case if MongoDB wasn't edited manually).

    Parameters
    ----------
    exp_chunks: list[list[dict]]
        List of list of dictionaries containing the experiment settings as saved in the MongoDB

    Returns
    -------
    exp_arrays: list[list[list[dict]]]
    """
    batch_idx = np.array([chunk[0]['batch_id'] for chunk in exp_chunks])
    unique_batch_idx = np.unique(batch_idx)
    ids_per_array = [np.where(batch_idx == array_bidx)[0] for array_bidx in unique_batch_idx]
    exp_arrays = [[exp_chunks[idx] for idx in chunk_ids] for chunk_ids in ids_per_array]
    return exp_arrays


def prepare_staged_experiments(collection, filter_dict=None, num_exps=0,
                               slurm=True, set_to_pending=True, print_pending=False):
    """
    Load experiments with state STAGED from the input MongoDB collection. If set_to_pending is True, we also set their
    status to PENDING.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection with STAGED experiments.
    filter_dict: dict
        Optional dict with custom database filters.
    num_exps: int
        Only set <num_exps> experiments' state to PENDING. If 0, set all STAGED experiments to PENDING.
    slurm: bool
        If True, we also set 'slurm.array_id' in order to prevent these jobs from being executed by local workers.
    set_to_pending: bool
        Whether to update the database entries to status PENDING.
    print_pending: bool
        Print the number of experiments set to PENDING.

    Returns
    -------
    The filtered list of database entries with status STAGED.
    """
    if filter_dict is None:
        filter_dict = {}

    query_dict = {'status': {"$in": States.STAGED}}
    query_dict.update(filter_dict)

    staged_experiments = list(collection.find(query_dict, limit=num_exps))
    if set_to_pending:
        update_dict = {"$set": {"status": States.PENDING[0]}}
        if slurm:
            # set slurm.array_id so that local workers don't start these jobs.
            update_dict['$set']['slurm.array_id'] = None

        if num_exps > 0:
            # only set the experiments to PENDING which will be run.
            collection.update_many({'_id': {'$in': [e['_id'] for e in staged_experiments
                                                    if e['status'] in States.STAGED]}},
                                   update_dict)
            nexps_set = min(num_exps, len(staged_experiments))
        else:
            collection.update_many(query_dict, update_dict)
            nexps_set = len(staged_experiments)
        if print_pending:
            logging.info(f"Setting {nexps_set} experiment{s_if(nexps_set)} to pending.")

    return staged_experiments


def set_environment_variables(gpus=None, cpus=None, environment_variables=None):
    if environment_variables is None:
        environment_variables = {}

    if gpus is not None:
        if isinstance(gpus, list):
            raise ArgumentError('Received an input of type list to set CUDA_VISIBLE_DEVICES.'
                                'Please pass a string for input "gpus", e.g. "1,2" if you want to use GPUs with IDs 1 and 2.')
        environment_variables['CUDA_VISIBLE_DEVICES'] = str(gpus)
    if cpus is not None:
        environment_variables['OMP_NUM_THREADS'] = str(cpus)
    os.environ.update(environment_variables)


def add_to_slurm_queue(collection, exps_list, unobserved=False, post_mortem=False,
                       output_to_file=True, output_to_console=False, srun=False,
                       debug_server=False):
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
    exp_chunks = chunk_list(exps_list)
    exp_arrays = batch_chunks(exp_chunks)
    njobs = len(exp_chunks)
    narrays = len(exp_arrays)

    logging.info(f"Starting {nexps} experiment{s_if(nexps)} in "
                 f"{njobs} Slurm job{s_if(njobs)} in {narrays} Slurm job array{s_if(narrays)}.")

    for exp_array in exp_arrays:
        sbatch_options = exp_array[0][0]['slurm']['sbatch_options']
        job_name = get_exp_name(exp_array[0][0], collection.name)
        set_slurm_job_name(sbatch_options, job_name, exp_array[0][0])
        if srun:
            assert len(exp_array) == 1
            assert len(exp_array[0]) == 1
            seml_arguments = []
            seml_arguments.append("--debug")
            if post_mortem:
                seml_arguments.append("--post-mortem")
            if output_to_console:
                seml_arguments.append("--output-to-console")
            if not output_to_file:
                seml_arguments.append("--no-file-output")
            if debug_server:
                seml_arguments.append("--debug-server")
            start_srun_job(collection, exp_array[0][0], unobserved,
                           srun_options=sbatch_options,
                           seml_arguments=seml_arguments)
        else:
            if output_to_file:
                output_dir_path = get_output_dir_path(exp_array[0][0])
            else:
                output_dir_path = "/dev/null"
            assert not post_mortem
            start_sbatch_job(collection, exp_array, unobserved,
                             name=job_name, output_dir_path=output_dir_path,
                             sbatch_options=sbatch_options,
                             debug_server=debug_server)


def start_local_worker(collection, num_exps=0, filter_dict=None, unobserved=False, post_mortem=False,
                       steal_slurm=False, output_to_console=False, output_to_file=True,
                       gpus=None, cpus=None, environment_variables=None, debug_server=False):
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
    login_node_name = 'fs'
    if login_node_name in os.uname()[1]:
        raise ArgumentError("Refusing to run a compute experiment on a login node. "
                            "Please use Slurm or a compute node.")

    if 'SLURM_JOBID' in os.environ:
        node_str = subprocess.run("squeue -j ${SLURM_JOBID} -O nodelist:1000",
                                  shell=True, check=True, capture_output=True).stdout
        node_id = node_str.decode("utf-8").split('\n')[1].strip()
        logging.info(f"SLURM assigned me the node(s): {node_id}")

    if num_exps > 0:
        logging.info(f'Starting local worker thread that will run up to {num_exps} experiment{s_if(num_exps)}, '
                     f'or until no pending experiments remain.')
    else:
        logging.info(f'Starting local worker thread that will run experiments until no pending experiments remain.')
        num_exps = int(1e30)

    set_environment_variables(gpus, cpus, environment_variables)

    num_exceptions = 0
    jobs_counter = 0

    exp_query = {}
    if not unobserved:
        exp_query['status'] = {"$in": States.PENDING}
    if not steal_slurm:
        exp_query['slurm.array_id'] = {'$exists': False}
        exp_query['slurm.id'] = {'$exists': False}

    exp_query.update(filter_dict)

    tq = tqdm()
    while collection.count_documents(exp_query) > 0 and jobs_counter < num_exps:
        if unobserved:
            exp = collection.find_one(exp_query)
        else:
            exp = collection.find_one_and_update(exp_query, {"$set": {"status": States.RUNNING[0]}})
        if exp is None:
            continue
        if 'array_id' in exp['slurm']:
            # Clean up MongoDB entry
            slurm_ids = {'array_id': exp['slurm']['array_id'],
                         'task_id': exp['slurm']['task_id']}
            reset_slurm_dict(exp)
            collection.replace_one({'_id': exp['_id']}, exp, upsert=False)

            # Cancel Slurm job; after cleaning up to prevent race conditions
            cancel_experiment_by_id(collection, exp['_id'], set_interrupted=False, slurm_dict=slurm_ids)

        tq.set_postfix(current_id=exp['_id'], failed=f"{num_exceptions}/{jobs_counter} experiments")

        # Add newline if we need to avoid tqdm's output
        if debug_server or output_to_console or logging.root.level <= logging.VERBOSE:
            print(file=sys.stderr)

        if output_to_file:
            output_dir_path = get_output_dir_path(exp)
        else:
            output_dir_path = None
        try:
            success = start_local_job(collection=collection, exp=exp, unobserved=unobserved, post_mortem=post_mortem,
                                      output_dir_path=output_dir_path, output_to_console=output_to_console,
                                      debug_server=debug_server)
            if success is False:
                num_exceptions += 1
        except KeyboardInterrupt:
            logging.info("Caught KeyboardInterrupt signal. Aborting.")
            exit(1)
        jobs_counter += 1
        tq.update()
        tq.set_postfix(current_id=exp['_id'], failed=f"{num_exceptions}/{jobs_counter} experiments")


def print_commands(collection, unobserved, post_mortem, debug_server, num_exps, filter_dict):
    orig_level = logging.root.level
    logging.root.setLevel(logging.VERBOSE)
    exps_list = prepare_staged_experiments(collection=collection, filter_dict=filter_dict, num_exps=num_exps,
                                           set_to_pending=False)
    if len(exps_list) == 0:
        return

    exp = exps_list[0]
    _, exe, config = get_command_from_exp(exp, collection.name,
                                          verbose=logging.root.level <= logging.VERBOSE,
                                          unobserved=unobserved, post_mortem=False)
    env = exp['seml']['conda_environment'] if 'conda_environment' in exp['seml'] else None

    logging.info("********** First experiment **********")
    logging.info(f"Executable: {exe}")
    if env is not None:
        logging.info(f"Anaconda environment: {env}")
    config.insert(0, 'with')
    config.append('--debug')

    # Remove double quotes, change single quotes to escaped double quotes
    config_vscode = [c.replace('"', '') for c in config]
    config_vscode = [c.replace("'", '\\"') for c in config_vscode]

    logging.info("\nArguments for VS Code debugger:")
    logging.info('["' + '", "'.join(config_vscode) + '"]')
    logging.info("Arguments for PyCharm debugger:")
    logging.info(" ".join(config))

    logging.info("\nCommand for running locally with post-mortem debugging:")
    interpreter, exe, config = get_command_from_exp(exps_list[0], collection.name,
                                                    verbose=logging.root.level <= logging.VERBOSE,
                                                    unobserved=unobserved, post_mortem=True)
    logging.info(f"{interpreter} {exe} with {' '.join(config)}")

    logging.info("\n********** All raw commands **********")
    logging.root.setLevel(orig_level)
    start_experiments(collection, local=True,
                      unobserved=unobserved, post_mortem=post_mortem,
                      num_exps=num_exps, filter_dict=filter_dict, print_command=True)
    for exp in exps_list:
        interpreter, exe, config = get_command_from_exp(
                exp, collection.name, unobserved=unobserved,
                post_mortem=post_mortem, debug_server=debug_server)
        logging.info(f"{interpreter} {exe} with {' '.join(config)}")


def start_experiments(db_collection_name, local, sacred_id, batch_id, filter_dict,
                      num_exps, post_mortem, debug, debug_server, print_command,
                      output_to_console, no_file_output, steal_slurm,
                      no_worker, set_to_pending=True,
                      worker_gpus=None, worker_cpus=None, worker_environment_vars=None):

    use_slurm = not local
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

    if not local:
        local_kwargs = {
                "--no-worker": no_worker,
                "--steal-slurm": steal_slurm,
                "--worker-gpus": worker_gpus,
                "--worker-cpus": worker_cpus,
                "--worker-environment-vars": worker_environment_vars}
        for key, val in local_kwargs.items():
            if val:
                raise ArgumentError(f"The argument '{key}' only works in local mode, not in Slurm mode.")
    if not local and not srun:
        non_sbatch_kwargs = {
                "--post-mortem": post_mortem,
                "--output-to-console": output_to_console}
        for key, val in non_sbatch_kwargs.items():
            if val:
                raise ArgumentError(f"The argument '{key}' does not work in regular Slurm mode. "
                                    "Remove the argument or use '--debug'.")

    if filter_dict is None:
        filter_dict = {}

    if unobserved:
        set_to_pending = False

    if worker_environment_vars is None:
        worker_environment_vars = {}

    if sacred_id is None:
        filter_dict = build_filter_dict([], batch_id, filter_dict)
    else:
        # if we have a specific sacred ID, we ignore the state of the experiment and run it in any case.
        all_states = (States.PENDING + States.STAGED + States.RUNNING + States.FAILED + States.INTERRUPTED
                      + States.KILLED + States.COMPLETED)
        filter_dict.update({'_id': sacred_id, "status": {"$in": all_states}})

    collection = get_collection(db_collection_name)

    if print_command:
        print_commands(collection, unobserved=unobserved,
                       post_mortem=post_mortem, debug_server=debug_server,
                       num_exps=num_exps, filter_dict=filter_dict)
        return

    staged_experiments = prepare_staged_experiments(
            collection=collection, filter_dict=filter_dict, num_exps=num_exps,
            slurm=use_slurm, set_to_pending=set_to_pending, print_pending=not use_slurm)

    if use_slurm:
        add_to_slurm_queue(collection=collection, exps_list=staged_experiments, unobserved=unobserved,
                           post_mortem=post_mortem, output_to_file=output_to_file,
                           output_to_console=output_to_console, srun=srun,
                           debug_server=debug_server)
    elif launch_worker:
        start_local_worker(collection=collection, num_exps=num_exps, filter_dict=filter_dict, unobserved=unobserved,
                           post_mortem=post_mortem, steal_slurm=steal_slurm,
                           output_to_console=output_to_console, output_to_file=output_to_file,
                           gpus=worker_gpus, cpus=worker_cpus, environment_variables=worker_environment_vars,
                           debug_server=debug_server)


def start_jupyter_job(sbatch_options: dict = None, conda_env: str = None, lab: bool = False):

    sbatch_options = sbatch_options if sbatch_options is not None else {}
    default_sbatch = SETTINGS.SBATCH_OPTIONS_TEMPLATES.JUPYTER_JOB
    default_sbatch.update(sbatch_options)
    # Construct sbatch options string
    sbatch_options_str = create_slurm_options_string(default_sbatch)

    template = pkg_resources.resource_string(__name__, "jupyter_template.sh").decode("utf-8")

    script = template.format(
            sbatch_options=sbatch_options_str,
            use_conda_env=str(conda_env is not None).lower(),
            conda_env=conda_env,
            notebook_or_lab=" notebook" if lab is False else "-lab",
    )

    random_int = np.random.randint(0, 999999)
    path = f"/tmp/{random_int}.sh"
    while os.path.exists(path):
        random_int = np.random.randint(0, 999999)
        path = f"/tmp/{random_int}.sh"
    with open(path, "w") as f:
        f.write(script)

    output = subprocess.run(f'sbatch {path}', shell=True, check=True, capture_output=True).stdout
    os.remove(path)

    slurm_array_job_id = int(output.split(b' ')[-1])
    logging.info(f"Started Jupyter job in Slurm job with ID {slurm_array_job_id}.")

    job_output = subprocess.run(f'scontrol show job {slurm_array_job_id} -o',
                                shell=True, check=True, capture_output=True).stdout
    job_output_results = job_output.decode("utf-8").split(" ")
    job_info_dict = {x.split("=")[0]: x.split("=")[1] for x in job_output_results}
    log_file = job_info_dict['StdOut']

    logging.info(f"The logfile of the job is {log_file}.")
    logging.info(f"Waiting to fetch the machine and port of the Jupyter instance once the job is running... "
                 f"(ctrl-C to cancel fetching).")

    while job_info_dict['JobState'] in States.PENDING:
        job_output = subprocess.run(f'scontrol show job {slurm_array_job_id} -o',
                                    shell=True, check=True, capture_output=True).stdout
        job_output_results = job_output.decode("utf-8").split(" ")
        job_info_dict = {x.split("=")[0]: x.split("=")[1] for x in job_output_results}
        time.sleep(1)
    is_starting_up = True
    time.sleep(1)
    if job_info_dict['JobState'] not in States.RUNNING:
        logging.error(f"Job failed. See log file at {log_file} for debug information.")
        exit(1)

    logging.info("Jupyter instance is starting up...")
    while is_starting_up:
        with open(log_file, "r") as f:
            log_file_contents = f.read()
        if " is running at" in log_file_contents:
            is_starting_up = False
        else:
            time.sleep(0.5)
    log_file_split = log_file_contents.split("\n")
    url_line = [x for x in log_file_split if "http" in x]
    if len(url_line) == 1:
        url = url_line[0].split(" ")[3]
        url = url.replace("https://", "")
        url = url.replace("http://", "")
        url = url.replace("/", "")
    logging.info(f"Startup completed. The Jupyter instance is running at '{url}'.")
    logging.info(f"To stop the job, run 'scancel {slurm_array_job_id}'.")
