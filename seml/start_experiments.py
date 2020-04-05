import os
import subprocess
import numpy as np

from seml.misc import get_config_from_exp, s_if
from seml import database_utils as db_utils
from seml import get_experiment_command
import warnings

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


def start_slurm_job(collection, exp_array, log_verbose, unobserved=False, post_mortem=False, name=None,
                    output_dir=".", sbatch_options=None, max_jobs_per_batch=None):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exp_array: List[List[dict]]
        List of chunks of experiments to run. Each chunk is a list of experiments.
    log_verbose: bool
        Print all the Python syscalls before running them.
    unobserved: bool
        Disable all Sacred observers (nothing written to MongoDB).
    post_mortem: bool
        Activate post-mortem debugging.
    name: str
        Job name, used by Slurm job and output file.
    output_dir: str
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
        raise ValueError(
            f"Can't set sbatch `job-name` Parameter explicitly. "
             "Use `name` parameter instead and SEML will do that for you.")
    name = name if name is not None else exp_array[0][0]['seml']['db_collection']
    job_name = f"{name}_{exp_array[0][0]['batch_id']}"
    sbatch_options['job-name'] = job_name

    # Set Slurm job array options
    sbatch_options['array'] = f"0-{len(exp_array) - 1}"
    if max_jobs_per_batch is not None:
        sbatch_options['array'] += f"%{max_jobs_per_batch}"

    # Set Slurm output parameter
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.isdir(output_dir_path):
        raise ValueError(
            f"Slurm output directory '{output_dir_path}' does not exist.")
    if 'output' in sbatch_options:
        raise ValueError(
            f"Can't set sbatch `output` Parameter explicitly. SEML will do that for you.")
    sbatch_options['output'] = f'{output_dir_path}/{name}_%A_%a.out'

    script = "#!/bin/bash\n"

    for key, value in sbatch_options.items():
        prepend = '-' if len(key) == 1 else '--'
        if key in ['partition', 'p'] and isinstance(value, list):
            script += f"#SBATCH {prepend}{key}={','.join(value)}\n"
        else:
            script += f"#SBATCH {prepend}{key}={value}\n"

    script += "\n"
    script += "cd ${SLURM_SUBMIT_DIR} \n"
    script += "echo Starting job ${SLURM_JOBID} \n"
    script += "echo \"SLURM assigned me the node(s): $(squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2)\"\n"

    if 'conda_environment' in exp_array[0][0]['seml']:
        script += "CONDA_BASE=$(conda info --base)\n"
        script += "source $CONDA_BASE/etc/profile.d/conda.sh\n"
        script += f"conda activate {exp_array[0][0]['seml']['conda_environment']}\n"

    get_config_file = get_experiment_command.__file__
    script += "process_ids=() \n"

    # Construct chunked list with all experiment IDs
    expid_strings = [('"' + ';'.join([str(exp['_id']) for exp in chunk]) + '"') for chunk in exp_array]
    script += f"all_exp_ids=({' '.join(expid_strings)}) \n"

    # Get experiment IDs for this Slurm task
    script += 'exp_ids_str="${all_exp_ids[$SLURM_ARRAY_TASK_ID]}"\n'
    script += 'IFS=";" read -r -a exp_ids <<< "$exp_ids_str"\n'

    collection_str = exp_array[0][0]['seml']['db_collection']

    script += "for exp_id in \"${exp_ids[@]}\"\n"
    script += "do\n"
    script += (f"cmd=$(python {get_config_file} "
               f"--experiment_id ${{exp_id}} --database_collection {collection_str} "
               f"--log-verbose {log_verbose} --unobserved {unobserved} --post-mortem {post_mortem})\n")
    script += "ret=$?\n"
    script += "if [ $ret -eq 0 ]\n"
    script += "then\n"
    script += "    eval $cmd &\n"
    script += "    process_ids+=($!)\n"

    script += "elif [ $ret -eq 1 ]\n"
    script += "then\n"
    script += "    echo WARNING: Experiment with ID ${exp_id} does not have status PENDING and will not be run. \n"
    script += "elif [ $ret -eq 2 ]\n"
    script += "then\n"
    script += "    (>&2 echo ERROR: Experiment with id ${exp_id} not found in the database.)\n"
    script += "fi\n"
    script += "done\n"

    script += "echo Experiments are running under the following process IDs:\n"
    script += "num_it=${#process_ids[@]}\n"
    script += "for ((i=0; i<$num_it; i++))\n"
    script += "do\n"
    script += "    echo \"Experiment ID: ${exp_ids[$i]}\tProcess ID: ${process_ids[$i]}\"\n"
    script += "done\n"
    script += "echo\n"
    script += "wait\n"

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
            if log_verbose:
                print(f"Started experiment with array job ID {slurm_array_job_id}, task ID {task_id}.")
    os.remove(path)


def do_work(collection_name, log_verbose, slurm=True, unobserved=False,
            post_mortem=False, num_exps=-1, filter_dict={}, dry_run=False,
            output_to_file=True):
    """Pull queued experiments from the database and run them.

    Parameters
    ----------
    collection_name: str
        Name of the collection in the MongoDB.
    log_verbose: bool
        Print all the Python syscalls before running them.
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

    collection = db_utils.get_collection(collection_name)

    if unobserved and not slurm and '_id' in filter_dict.keys():
        query_dict = {}
    else:
        query_dict = {'status': {"$in": ['QUEUED']}}
    query_dict.update(filter_dict)

    if collection.count_documents(query_dict) <= 0:
        print("No queued experiments.")
        return

    exps_full = list(collection.find(query_dict))

    nexps = num_exps if num_exps > 0 else len(exps_full)
    exps_list = exps_full[:nexps]

    if dry_run:
        configs = []
        for exp in exps_list:
            configs.append(get_config_from_exp(exp, log_verbose=log_verbose,
                                               unobserved=unobserved, post_mortem=post_mortem))
        return configs
    elif slurm:
        assert output_to_file is True, "Output cannot be written to stdout in Slurm mode."
        exp_chunks = db_utils.chunk_list(exps_list)
        exp_arrays = db_utils.batch_chunks(exp_chunks)
        njobs = len(exp_chunks)
        narrays = len(exp_arrays)

        print(f"Starting {nexps} experiment{s_if(nexps)} in "
              f"{njobs} Slurm job{s_if(njobs)} in {narrays} Slurm job array{s_if(narrays)}.")

        for exp_array in exp_arrays:
            seml_config = exp_array[0][0]['seml']
            slurm_config = exp_array[0][0]['slurm']
            if 'output_dir' in slurm_config:
                warnings.warn("'output_dir' has moved from 'slurm' to 'seml'. Please adapt your YAML accordingly"
                              "by moving the 'output_dir' parameter from 'slurm' to 'seml'.")
            elif 'output_dir' in seml_config:
                slurm_config['output_dir'] = seml_config['output_dir']
            del slurm_config['experiments_per_job']
            start_slurm_job(collection, exp_array, log_verbose, unobserved, post_mortem, **slurm_config)
    else:
        login_node_name = 'fs'
        if login_node_name in os.uname()[1]:
            raise ValueError("Refusing to run a compute experiment on a login node. "
                             "Please use Slurm or a compute node.")

        print(f'Starting local worker thread that will run up to {nexps} experiments, '
              f'until no queued experiments remain.')
        if not unobserved:
            collection.update_many({'_id': {'$in': [e['_id'] for e in exps_list]}}, {"$set": {"status": "PENDING"}})
        num_exceptions = 0
        i_exp = 0

        tq = tqdm(exps_list)
        for exp in tq:
            exe, config = get_config_from_exp(exp, log_verbose=log_verbose,
                                              unobserved=unobserved, post_mortem=post_mortem)

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
                    continue

            if log_verbose:
                print(f'Running the following command:\n {cmd}')
            try:
                output_dir = "."
                seml_config = exp['seml']
                slurm_config = exp['slurm']
                if 'output_dir' in slurm_config:
                    warnings.warn(
                        "'output_dir' has moved from 'slurm' to 'seml'. Please adapt your YAML accordingly"
                        "by moving the 'output_dir' parameter from 'slurm' to 'seml'.")
                    output_dir = slurm_config['output_dir']
                if 'output_dir' in seml_config:
                    output_dir = seml_config['output_dir']
                output_dir_path = os.path.abspath(os.path.expanduser(output_dir))
                exp_name = slurm_config['name']

                output_file = f"{output_dir_path}/{exp_name}_{exp['_id']}.out"
                collection.find_and_modify({'_id': exp['_id']}, {"$set": {"seml.output_file": output_file}})

                if 'conda_environment' in seml_config:
                    cmd = (f". $(conda info --base)/etc/profile.d/conda.sh "
                            f"&& conda activate {seml_config['conda_environment']} "
                            f"&& {cmd} "
                            f"&& conda deactivate")

                if output_to_file:
                    with open(output_file, "w") as log_file:
                        # pdb works with check_call but not with check_output. Maybe because of stdout/stdin.
                        subprocess.check_call(cmd, shell=True, stderr=log_file,
                                              stdout=log_file)
                else:
                    subprocess.check_call(cmd, shell=True)

            except subprocess.CalledProcessError:
                num_exceptions += 1
            except IOError:
                print(f"Log file {output_file} could not be written.")
                # Since Sacred is never called in case of I/O error, we need to set the experiment state manually.
                collection.find_one_and_update(filter={'_id': exp['_id']},
                                               update={'$set': {'status': 'FAILED'}},
                                               upsert=False)
            finally:
                i_exp += 1
                tq.set_postfix(failed=f"{num_exceptions}/{i_exp} experiments")


def print_commands(db_collection_name, log_verbose, unobserved, post_mortem, num_exps, filter_dict):
    configs = do_work(db_collection_name, log_verbose=True, slurm=False,
                      unobserved=True, post_mortem=False,
                      num_exps=1, filter_dict=filter_dict, dry_run=True)
    if configs is None:
        return
    print("********** First experiment **********")
    exe, config = configs[0]
    print(f"Executable: {exe}")
    config.insert(0, 'with')
    config.append('--debug')

    # Remove double quotes, change single quotes to escaped double quotes
    config_vscode = [c.replace('"', '') for c in config]
    config_vscode = [c.replace("'", '\\"') for c in config_vscode]

    print("Arguments for VS Code debugger:")
    print('["' + '", "'.join(config_vscode) + '"]')
    print("Arguments for PyCharm debugger:")
    print(" ".join(config))

    print("\nCommand for running locally with post-mortem debugging:")
    configs = do_work(db_collection_name, log_verbose=True, slurm=False,
                      unobserved=True, post_mortem=True,
                      num_exps=1, filter_dict=filter_dict, dry_run=True)
    exe, config = configs[0]
    print(f"python {exe} with {' '.join(config)}")

    print()
    print("********** All raw commands **********")
    configs = do_work(db_collection_name, log_verbose=log_verbose, slurm=False,
                      unobserved=unobserved, post_mortem=post_mortem,
                      num_exps=num_exps, filter_dict=filter_dict, dry_run=True)
    for (exe, config) in configs:
        print(f"python {exe} with {' '.join(config)}")


def start_experiments(config_file, local, sacred_id, batch_id, filter_dict,
                      test, unobserved, post_mortem, debug, verbose, dry_run,
                      output_to_console):
    use_slurm = not local
    output_to_file = not output_to_console

    db_collection_name = db_utils.read_config(config_file)[0]['db_collection']

    if debug:
        test = 1
        use_slurm = False
        unobserved = True
        post_mortem = True

    if test != -1:
        verbose = True

    if test != -1 and not use_slurm:
        output_to_file = False

    if sacred_id is None:
        filter_dict = db_utils.build_filter_dict([], batch_id, filter_dict)
    else:
        filter_dict = {'_id': sacred_id}

    if dry_run:
        print_commands(db_collection_name, log_verbose=verbose,
                       unobserved=unobserved, post_mortem=post_mortem,
                       num_exps=test, filter_dict=filter_dict)
    else:
        do_work(db_collection_name, log_verbose=verbose, slurm=use_slurm,
                unobserved=unobserved, post_mortem=post_mortem,
                num_exps=test, filter_dict=filter_dict, dry_run=dry_run,
                output_to_file=output_to_file)
