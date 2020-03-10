import os
import subprocess
import numpy as np

from seml.misc import get_config_from_exp, s_if
from seml import database_utils as db_utils
from seml import check_cancelled

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


def start_slurm_job(collection, exps, log_verbose, unobserved=False, post_mortem=False, name=None,
                    output_dir=".", sbatch_options=None):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    exps: List[dict]
        List of experiments to run.
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

    Returns
    -------
    None
    """

    # Set Slurm job-name parameter
    if 'job-name' in sbatch_options.keys():
        raise ValueError(
            f"Can't set sbatch `job-name` Parameter explicitly. "
             "Use `name` parameter instead and SEML will do that for you.")
    name = name if name is not None else exps[0]['seml']['db_collection']
    id_strs = [str(exp['_id']) for exp in exps]
    job_name = f"{name}_{','.join(id_strs)}"
    sbatch_options['job-name'] = job_name

    # Set Slurm output parameter
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.isdir(output_dir_path):
        raise ValueError(
            f"Slurm output directory '{output_dir_path}' does not exist.")
    if 'output' in sbatch_options.keys():
        raise ValueError(
            f"Can't set sbatch `output` Parameter explicitly. SEML will do that for you.")
    sbatch_options['output'] = f'{output_dir_path}/{name}-%j.out'

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

    if "conda_environment" in exps[0]['seml']:
        script += "CONDA_BASE=$(conda info --base)\n"
        script += "source $CONDA_BASE/etc/profile.d/conda.sh\n"
        script += f"conda activate {exps[0]['seml']['conda_environment']}\n"

    check_file = check_cancelled.__file__
    script += "process_ids=() \n"
    script += f"exp_ids=({' '.join([str(e['_id']) for e in exps])}) \n"
    commands = []
    for ix, exp in enumerate(exps):
        exe, config = get_config_from_exp(exp, log_verbose=log_verbose,
                                          unobserved=unobserved, post_mortem=post_mortem)
        cmd = f"python {exe} with {' '.join(config)}"
        commands.append(cmd)
        collection_str = exp['seml']['db_collection']
        script += f"python {check_file} --experiment_id {exp['_id']} --database_collection {collection_str}\n"
        script += "ret=$?\n"
        script += "if [ $ret -eq 0 ]\n"
        script += "then\n"
        script += f"    {cmd}  & \n"
        script += f"    process_ids[{ix}]=$!\n"

        script += "elif [ $ret -eq 1 ]\n"
        script += "then\n"
        script += f"    echo WARNING: Experiment with ID {exp['_id']} has status INTERRUPTED and will not be run. \n"
        script += "elif [ $ret -eq 2 ]\n"
        script += "then\n"
        script += f"    (>&2 echo ERROR: Experiment with id {exp['_id']} not found in the database.)\n"
        script += "fi\n"

        if log_verbose:
            print(f'Running the following command:\n {cmd}')

    script += f"echo Experiments are running under the following process IDs:\n"
    script += f"num_it=${{#process_ids[@]}}\n"
    script += f"for ((i=0; i<$num_it; i++))\n"
    script += f"do\n"
    script += f"    echo \"Experiment ID: ${{exp_ids[$i]}}\tProcess ID: ${{process_ids[$i]}}\"\n"
    script += f"done\n"
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
    slurm_job_id = int(output.split(b' ')[-1])
    for ix, exp in enumerate(exps):
        if not unobserved:
            collection.update_one(
                    {'_id': exp['_id']},
                    {'$set': {
                        'status': 'PENDING',
                        'slurm.id': slurm_job_id,
                        'slurm.step_id': ix,
                        'slurm.sbatch_options': sbatch_options,
                        'slurm.command': commands[ix],
                        'slurm.output_file': f"{output_dir_path}/{name}-{slurm_job_id}.out"}})
        if log_verbose:
            print(f"Started experiment with ID {slurm_job_id}")
    os.remove(path)


def do_work(collection_name, log_verbose, slurm=True, unobserved=False,
            post_mortem=False, num_exps=-1, filter_dict={}, dry_run=False):
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

    Returns
    -------
    None
    """

    collection = db_utils.get_collection(collection_name)

    query_dict = {'status': {"$in": ['QUEUED']}}
    query_dict.update(filter_dict)

    if collection.count_documents(query_dict) <= 0:
        print("No queued experiments.")
        return

    exps_list = list(collection.find(query_dict))

    nexps = num_exps if num_exps > 0 else len(exps_list)
    exp_chunks = db_utils.chunk_list(exps_list[:nexps])
    njobs = len(exp_chunks)

    if dry_run:
        configs = []
        for exps in exp_chunks:
            for exp in exps:
                configs.append(get_config_from_exp(exp, log_verbose=log_verbose,
                                                   unobserved=unobserved, post_mortem=post_mortem))
        return configs
    elif slurm:
        print(f"Starting {nexps} experiment{s_if(nexps)} in "
              f"{njobs} Slurm job{s_if(njobs)}.")

        for exps in tqdm(exp_chunks):
            slurm_config = exps[0]['slurm']
            del slurm_config['experiments_per_job']
            start_slurm_job(collection, exps, log_verbose, unobserved, post_mortem, **slurm_config)
    else:
        login_node_name = 'fs'
        if login_node_name in os.uname()[1]:
            raise ValueError("Refusing to run a compute experiment on a login node. "
                             "Please use Slurm or a compute node.")

        print(f'Starting local worker thread that will run up to {nexps} experiments, '
              f'until no queued experiments remain.')

        for exps in tqdm(exp_chunks):
            for exp in exps:
                exe, config = get_config_from_exp(exp, log_verbose=log_verbose,
                                                  unobserved=unobserved, post_mortem=post_mortem)

                cmd = f"python {exe} with {' '.join(config)}"

                if not unobserved:
                    db_entry = collection.find_one_and_update(filter={'_id': exp['_id'], 'status': 'QUEUED'},
                                                              update={'$set': {'status': 'PENDING',
                                                                               'seml.command': cmd}},
                                                              upsert=False)
                    if db_entry is None:
                        # another worker already set this entry to PENDING (or at least, it's no longer QUEUED)
                        # so we ignore it.
                        continue

                if log_verbose:
                    print(f'Running the following command:\n {cmd}')
                # pdb works with check_call but not with check_output. Maybe because of stdout/stdin.
                try:
                    subprocess.check_call(cmd, shell=True)
                except subprocess.CalledProcessError as e:
                    output = e.output
                    print(output)


def print_commands(db_collection_name, log_verbose, unobserved, post_mortem, num_exps, filter_dict):
    print("********** First experiment **********")
    configs = do_work(db_collection_name, log_verbose=True, slurm=False,
                      unobserved=True, post_mortem=False,
                      num_exps=1, filter_dict=filter_dict, dry_run=True)
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
                      test, unobserved, post_mortem, debug, verbose, dry_run):
    use_slurm = not local

    db_collection_name = db_utils.read_config(config_file)[0]['db_collection']

    if debug:
        test = 1
        use_slurm = False
        unobserved = True
        post_mortem = True

    if test != -1:
        verbose = True

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
                num_exps=test, filter_dict=filter_dict, dry_run=dry_run)
