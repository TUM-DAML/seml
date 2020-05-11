import os
import sys
import subprocess
import logging
import numpy as np
import shutil
import pkg_resources
try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable

from seml.database import get_collection, build_filter_dict
from seml.sources import load_sources_from_db
from seml.utils import s_if


def get_command_from_exp(exp, db_collection_name, verbose=False, unobserved=False,
                         post_mortem=False, debug=False):
    if 'executable' not in exp['seml']:
        raise ValueError(f"No executable found for experiment {exp['_id']}. Aborting.")
    exe = exp['seml']['executable']
    if 'executable_relative' in exp['seml']:  # backwards compatibility
        exe = exp['seml']['executable_relative']

    config = exp['config']
    config['db_collection'] = db_collection_name
    if not unobserved:
        config['overwrite'] = exp['_id']
    config_strings = [f'{key}="{val}"' for key, val in config.items()]
    if not verbose:
        config_strings.append("--force")
    if unobserved:
        config_strings.append("--unobserved")
    if post_mortem:
        config_strings.append("--pdb")
    if debug:
        config_strings.append("--debug")

    return exe, config_strings


def get_output_dir_path(config):
    if 'output_dir' in config['slurm']:
        logging.warning("'output_dir' has moved from 'slurm' to 'seml'. Please adapt your YAML accordingly"
                        "by moving the 'output_dir' parameter from 'slurm' to 'seml'.")
        output_dir = config['slurm']['output_dir']
    elif 'output_dir' in config['seml']:
        output_dir = config['seml']['output_dir']
    else:
        output_dir = '.'
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.isdir(output_dir_path):
        logging.error(f"Output directory '{output_dir_path}' does not exist.")
        sys.exit(1)
    return output_dir_path


def get_exp_name(config, db_collection_name):
    if 'name' in config['slurm']:
        logging.warning("'name' has moved from 'slurm' to 'seml'. Please adapt your YAML accordingly"
                        "by moving the 'name' parameter from 'slurm' to 'seml'.")
        name = config['slurm']['name']
    elif 'name' in config['seml']:
        name = config['seml']['name']
    else:
        name = db_collection_name
    return name


def start_slurm_job(collection, exp_array, unobserved=False, post_mortem=False, name=None,
                    output_dir_path=".", sbatch_options=None, max_jobs_per_batch=None):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp_array: List[List[dict]]
        List of chunks of experiments to run. Each chunk is a list of experiments.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    post_mortem: bool
        Activate post-mortem debugging.
    name: str
        Job name, used by Slurm job and output file.
    output_dir_path: str
        Directory (relative to home directory) where to store the slurm output files.
    sbatch_options: dict
        A dictionary that contains options for #SBATCH, e.g., {'mem': 8000} to limit the job's memory to 8,000 MB.
    max_jobs_per_batch: int
        Maximum number of Slurm jobs running per experiment batch.

    Returns
    -------
    None
    """

    # Set Slurm job-name parameter
    if 'job-name' in sbatch_options:
        logging.error("Can't set sbatch `job-name` Parameter explicitly. "
                      "Use `name` parameter instead and SEML will do that for you.")
        sys.exit(1)
    job_name = f"{name}_{exp_array[0][0]['batch_id']}"
    sbatch_options['job-name'] = job_name

    # Set Slurm job array options
    sbatch_options['array'] = f"0-{len(exp_array) - 1}"
    if max_jobs_per_batch is not None:
        sbatch_options['array'] += f"%{max_jobs_per_batch}"

    # Set Slurm output parameter
    if 'output' in sbatch_options:
        logging.error(f"Can't set sbatch `output` Parameter explicitly. SEML will do that for you.")
        sys.exit(1)
    sbatch_options['output'] = f'{output_dir_path}/{name}_%A_%a.out'

    # Construct sbatch options string
    sbatch_options_str = ""
    for key, value in sbatch_options.items():
        prepend = '-' if len(key) == 1 else '--'
        if key in ['partition', 'p'] and isinstance(value, list):
            sbatch_options_str += f"#SBATCH {prepend}{key}={','.join(value)}\n"
        else:
            sbatch_options_str += f"#SBATCH {prepend}{key}={value}\n"

    # Construct chunked list with all experiment IDs
    expid_strings = [('"' + ';'.join([str(exp['_id']) for exp in chunk]) + '"') for chunk in exp_array]

    with_sources = ('source_files' in exp_array[0][0]['seml'])
    use_conda_env = ('conda_environment' in exp_array[0][0]['seml']
                     and exp_array[0][0]['seml']['conda_environment'] is not None)

    # Construct Slurm script
    template = pkg_resources.resource_string(__name__, "slurm_template.sh").decode("utf-8")
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
            get_cmd_fname=f"{os.path.dirname(__file__)}/prepare_experiment.py",
            db_collection_name=collection.name,
            sources_argument="--stored-sources-dir $tmpdir" if with_sources else "",
            verbose=logging.root.level <= logging.VERBOSE,
            unobserved=unobserved,
            post_mortem=post_mortem,
    )

    random_int = np.random.randint(0, 999999)
    path = f"/tmp/{random_int}.sh"
    while os.path.exists(path):
        random_int = np.random.randint(0, 999999)
        path = f"/tmp/{random_int}.sh"
    with open(path, "w") as f:
        f.write(script)
    output = subprocess.check_output(f'sbatch {path}', shell=True)
    slurm_array_job_id = int(output.split(b' ')[-1])
    for task_id, chunk in enumerate(exp_array):
        for exp in chunk:
            if not unobserved:
                collection.update_one(
                        {'_id': exp['_id']},
                        {'$set': {
                            'status': 'PENDING',
                            'slurm.array_id': slurm_array_job_id,
                            'slurm.task_id': task_id,
                            'slurm.sbatch_options': sbatch_options,
                            'seml.output_file': f"{output_dir_path}/{name}_{slurm_array_job_id}_{task_id}.out"}})
            logging.verbose(f"Started experiment with array job ID {slurm_array_job_id}, task ID {task_id}.")
    os.remove(path)


def start_local_job(collection, exp, unobserved=False, post_mortem=False, output_dir_path='.'):
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

    Returns
    -------
    None
    """

    use_stored_sources = ('source_files' in exp['seml'])

    exe, config = get_command_from_exp(exp, collection.name,
                                       verbose=logging.root.level <= logging.VERBOSE,
                                       unobserved=unobserved, post_mortem=post_mortem)
    if not use_stored_sources:
        os.chdir(exp['seml']['working_dir'])

    cmd = f"python {exe} with {' '.join(config)}"
    if not unobserved:
        # check also whether PENDING experiments have their Slurm ID set, in this case they are waiting
        # for Slurm execution and we don't start them locally.
        db_entry = collection.find_one_and_update(filter={'_id': exp['_id'], 'status': 'PENDING',
                                                          'slurm.array_id': {'$exists': False}},
                                                  update={'$set': {'seml.command': cmd,
                                                                   'status': 'RUNNING'}},
                                                  upsert=False)
        if db_entry is None:
            # another worker already set this entry to PENDING (or at least, it's no longer QUEUED)
            # so we ignore it.
            return None

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
            cmd = f'PYTHONPATH="{temp_dir}:$PYTHONPATH" python {temp_dir}/{exe} with {" ".join(config)}'

        if 'conda_environment' in seml_config and seml_config['conda_environment'] is not None:
            cmd = (f". $(conda info --base)/etc/profile.d/conda.sh "
                   f"&& conda activate {seml_config['conda_environment']} "
                   f"&& {cmd} "
                   f"&& conda deactivate")

        logging.verbose(f'Running the following command:\n {cmd}')

        if output_dir_path:
            exp_name = get_exp_name(exp, collection.name)
            output_file = f"{output_dir_path}/{exp_name}_{exp['_id']}.out"
            collection.find_and_modify({'_id': exp['_id']}, {"$set": {"seml.output_file": output_file}})
            with open(output_file, "w") as log_file:
                # pdb works with check_call but not with check_output. Maybe because of stdout/stdin.
                subprocess.check_call(cmd, shell=True, stderr=log_file, stdout=log_file)
        else:
            subprocess.check_call(cmd, shell=True)

    except subprocess.CalledProcessError:
        success = False
    except IOError:
        logging.error(f"Log file {output_file} could not be written.")
        # Since Sacred is never called in case of I/O error, we need to set the experiment state manually.
        collection.find_one_and_update(filter={'_id': exp['_id']},
                                       update={'$set': {'status': 'FAILED'}},
                                       upsert=False)
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


def start_jobs(db_collection_name, slurm=True, unobserved=False,
               post_mortem=False, num_exps=-1, filter_dict={}, dry_run=False,
               output_to_file=True):
    """Pull queued experiments from the database and run them.

    Parameters
    ----------
    db_collection_name: str
        Name of the collection in the MongoDB.
    slurm: bool
        Use the Slurm cluster.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    post_mortem: bool
        Activate post-mortem debugging.
    num_exps: int, default: -1
        If >0, will only submit the specified number of experiments to the cluster.
        This is useful when you only want to test your setup.
    filter_dict: dict
        Dictionary for filtering the entries in the collection.
    dry_run: bool
        Just return the executables and configurations instead of running them.
    output_to_file: bool
        Pipe all output (stdout and stderr) to an output file.
        Can only be False if slurm is False.

    Returns
    -------
    None
    """

    collection = get_collection(db_collection_name)

    if unobserved and not slurm and '_id' in filter_dict:
        query_dict = {}
    else:
        query_dict = {'status': {"$in": ['QUEUED']}}
    query_dict.update(filter_dict)

    if collection.count_documents(query_dict) <= 0:
        logging.error("No queued experiments.")
        return

    exps_full = list(collection.find(query_dict))

    nexps = num_exps if num_exps > 0 else len(exps_full)
    exps_list = exps_full[:nexps]

    if dry_run:
        configs = []
        for exp in exps_list:
            exe, config = get_command_from_exp(exp, db_collection_name,
                                               verbose=logging.root.level <= logging.VERBOSE,
                                               unobserved=unobserved, post_mortem=post_mortem)
            if 'conda_environment' in exp['seml']:
                configs.append((exe, exp['seml']['conda_environment'], config))
            else:
                configs.append((exe, None, config))
        return configs
    elif slurm:
        if not output_to_file:
            logging.error("Output cannot be written to stdout in Slurm mode. "
                          "Remove the '--output-to-console' argument.")
            sys.exit(1)
        exp_chunks = chunk_list(exps_list)
        exp_arrays = batch_chunks(exp_chunks)
        njobs = len(exp_chunks)
        narrays = len(exp_arrays)

        logging.info(f"Starting {nexps} experiment{s_if(nexps)} in "
                     f"{njobs} Slurm job{s_if(njobs)} in {narrays} Slurm job array{s_if(narrays)}.")

        for exp_array in exp_arrays:
            job_name = get_exp_name(exp_array[0][0], collection.name)
            output_dir_path = get_output_dir_path(exp_array[0][0])
            slurm_config = exp_array[0][0]['slurm']
            del slurm_config['experiments_per_job']
            start_slurm_job(collection, exp_array, unobserved, post_mortem,
                            name=job_name, output_dir_path=output_dir_path, **slurm_config)
    else:
        login_node_name = 'fs'
        if login_node_name in os.uname()[1]:
            logging.error("Refusing to run a compute experiment on a login node. "
                          "Please use Slurm or a compute node.")
            sys.exit(1)
        [get_output_dir_path(exp) for exp in exps_list]  # Check if output dir exists
        logging.info(f'Starting local worker thread that will run up to {nexps} experiment{s_if(nexps)}, '
                     f'until no queued experiments remain.')
        if not unobserved:
            collection.update_many({'_id': {'$in': [e['_id'] for e in exps_list]}}, {"$set": {"status": "PENDING"}})
        num_exceptions = 0
        tq = tqdm(enumerate(exps_list))
        for i_exp, exp in tq:
            if output_to_file:
                output_dir_path = get_output_dir_path(exp)
            else:
                output_dir_path = None
            success = start_local_job(collection, exp, unobserved, post_mortem, output_dir_path)
            if success is False:
                num_exceptions += 1
            tq.set_postfix(failed=f"{num_exceptions}/{i_exp} experiments")


def print_commands(db_collection_name, unobserved, post_mortem, num_exps, filter_dict):
    orig_level = logging.root.level
    logging.root.setLevel(logging.VERBOSE)
    configs = start_jobs(db_collection_name, slurm=False,
                         unobserved=True, post_mortem=False,
                         num_exps=1, filter_dict=filter_dict, dry_run=True)
    if configs is None:
        return
    logging.info("********** First experiment **********")
    exe, env, config = configs[0]
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
    configs = start_jobs(db_collection_name, slurm=False,
                         unobserved=True, post_mortem=True,
                         num_exps=1, filter_dict=filter_dict, dry_run=True)
    exe, _, config = configs[0]
    logging.info(f"python {exe} with {' '.join(config)}")

    logging.info("\n********** All raw commands **********")
    logging.root.setLevel(orig_level)
    configs = start_jobs(db_collection_name, slurm=False,
                         unobserved=unobserved, post_mortem=post_mortem,
                         num_exps=num_exps, filter_dict=filter_dict, dry_run=True)
    for (exe, _, config) in configs:
        logging.info(f"python {exe} with {' '.join(config)}")


def start_experiments(db_collection_name, local, sacred_id, batch_id, filter_dict,
                      num_exps, unobserved, post_mortem, debug, dry_run,
                      output_to_console):
    use_slurm = not local
    output_to_file = not output_to_console

    if debug:
        num_exps = 1
        use_slurm = False
        unobserved = True
        post_mortem = True
        output_to_file = False
        logging.root.setLevel(logging.VERBOSE)

    if sacred_id is None:
        filter_dict = build_filter_dict([], batch_id, filter_dict)
    else:
        filter_dict = {'_id': sacred_id}

    if dry_run:
        print_commands(db_collection_name, unobserved=unobserved, post_mortem=post_mortem,
                       num_exps=num_exps, filter_dict=filter_dict)
    else:
        start_jobs(db_collection_name, slurm=use_slurm,
                   unobserved=unobserved, post_mortem=post_mortem,
                   num_exps=num_exps, filter_dict=filter_dict, dry_run=dry_run,
                   output_to_file=output_to_file)
