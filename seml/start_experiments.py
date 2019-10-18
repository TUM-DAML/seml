import os
import math
import subprocess
import numpy as np

from seml.misc import get_cmd_from_exp_dict, s_if
from seml import database_utils as db_utils
from seml import check_cancelled

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total: x


def start_slurm_job(exps, log_verbose, name=None, output_dir=".",
                    sbatch_options=None):
    """Run a list of experiments as a job on the Slurm cluster.

    Parameters
    ----------
    exps: List[dict]
        List of experiments to run.
    log_verbose: bool
        Print all the Python syscalls before running them.
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
    script += "echo SLURM assigned me these nodes:\n"
    script += "squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2\n"

    collection = db_utils.get_collection(exps[0]['seml']['db_collection'])

    if "conda_environment" in exps[0]['seml']:
        script += "CONDA_BASE=$(conda info --base)\n"
        script += "source $CONDA_BASE/etc/profile.d/conda.sh\n"
        script += f"conda activate {exps[0]['seml']['conda_environment']}\n"

    check_file = check_cancelled.__file__
    script += "process_ids=() \n"
    script += f"exp_ids=({' '.join([str(e['_id']) for e in exps])}) \n"
    for ix, exp in enumerate(exps):
        cmd = get_cmd_from_exp_dict(exp)
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

        collection.update_one(
                {'_id': exp['_id']},
                {'$set': {'status': 'PENDING'}})
        collection.update_one(
                {'_id': exp['_id']},
                {'$set': {
                    'slurm.sbatch_options': sbatch_options,
                    'slurm.step_id': ix}})

        if log_verbose:
            print(f'Running the following command:\n {cmd}')

    script += f"echo Experiments are running under the following process IDs:\n"
    script += f"num_it=${{#process_ids[@]}}\n"
    script += f"for ((i=0; i<$num_it; i++))\n"
    script += f"do\n"
    script += f"    echo \"Experiment ID: ${{exp_ids[$i]}}\tProcess ID: ${{process_ids[$i]}}\"\n"
    script += f"done\n"
    script += f"wait \n"

    random_int = np.random.randint(0, 999999)
    path = f"/tmp/{random_int}.sh"
    while os.path.exists(path):
        random_int = np.random.randint(0, 999999)
        path = f"/tmp/{random_int}.sh"
    with open(path, "w") as f:
        f.write(script)

    output = subprocess.check_output(f'sbatch {path}', shell=True)
    os.remove(path)
    slurm_job_id = int(output.split(b' ')[-1])
    for exp in exps:
        collection.update_one(
                {'_id': exp['_id']},
                {'$set': {'slurm.id': slurm_job_id,
                          'slurm.output_file': f"{output_dir_path}/slurm-{slurm_job_id}.out"}})
        if log_verbose:
            print(f"Started experiment with ID {slurm_job_id}")


def do_work(collection_name, log_verbose, slurm=True, num_exps=-1, filter_dict={}):
    """Pull queued experiments from the database and run them.

    Parameters
    ----------
    collection_name: str
        Name of the collection in the MongoDB.
    log_verbose: bool
        Print all the Python syscalls before running them.
    slurm: bool
        Use the Slurm cluster.
    num_exps: int, default: -1
        If >0, will only submit the specified number of experiments to the cluster.
        This is useful when you only want to test your setup.
    filter_dict: dict
        Dictionary for filtering the entries in the collection.

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

    if slurm:
        print(f"Starting {nexps} experiment{s_if(nexps)} in "
              f"{njobs} Slurm job{s_if(njobs)}.")

        for exps in tqdm(exp_chunks):
            slurm_config = exps[0]['slurm']
            del slurm_config['experiments_per_job']
            start_slurm_job(exps, log_verbose, **slurm_config)
    else:
        login_node_name = 'fs'
        if login_node_name in os.uname()[1]:
            raise ValueError("Refusing to run a compute experiment on a login node. "
                             "Please use Slurm or a compute node.")

        print(f"Starting {nexps} experiment{s_if(nexps)} locally.")
        for exp in exps_list[:nexps]:
            collection.update_one(
                    {'_id': exp['_id']},
                    {'$set': {'status': 'PENDING'}})

        for exps in tqdm(exp_chunks):
            for exp in exps:
                cmd = get_cmd_from_exp_dict(exp)
                if log_verbose:
                    print(f'Running the following command:\n {cmd}')
                # pdb works with check_call but not with check_output. Maybe because of stdout/stdin.
                subprocess.check_call(cmd, shell=True)


def start_experiments(config_file, local, sacred_id, batch_id, filter_dict, test, verbose):
    use_slurm = not local

    db_collection_name = db_utils.read_config(config_file)[0]['db_collection']

    if test != -1:
        verbose = True

    if sacred_id is None:
        filter_dict = db_utils.build_filter_dict([], batch_id, filter_dict)
    else:
        filter_dict = {'_id': sacred_id}

    do_work(db_collection_name, verbose, slurm=use_slurm,
            num_exps=test, filter_dict=filter_dict)
