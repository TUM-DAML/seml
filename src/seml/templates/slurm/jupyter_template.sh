#!/bin/bash
{sbatch_options}

# Move either to project root dir or the config file path.
cd ${{SLURM_SUBMIT_DIR}}

# Print job information
echo "Starting job ${{SLURM_JOBID}}"
echo "SLURM assigned me the node(s): $(squeue -j ${{SLURM_JOBID}} -O nodelist:1000 | tail -n +2 | sed -e 's/[[:space:]]*$//')"

# Activate Anaconda environment
if {use_conda_env}; then
    CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate {conda_env}
fi

# Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318
export XDG_RUNTIME_DIR=""
jupyter{notebook_or_lab} --no-browser --ip="*"
