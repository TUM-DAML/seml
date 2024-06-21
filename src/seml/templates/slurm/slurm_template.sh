#!/bin/bash
{sbatch_options}

# Execute optional bash commands
{setup_command}

# Move either to project root dir or the config file path.
cd {working_dir}

# Print job information
echo "Starting job ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "SLURM assigned me the node(s): ${{SLURM_NODELIST}}"

# Activate Anaconda environment
if {use_conda_env}; then
    CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate {conda_env}
fi

# List with all experiment IDs
all_exp_ids=({exp_ids})

process_ids=() # list of all process ids
exp_ids=() # list of all sacred ids
tmp_dirs=() # list of temporary directories
# Start experiments in separate processes
for i in $(seq 1 {experiments_per_job}); do
    # Claim an experiment, this is separate from the experiment preparation
    # to avoid race conditions and handle multi-process experiments well.
    exp_id=$(seml {db_collection_name} claim-experiment ${{all_exp_ids[@]}})
    if [ $? -eq 3 ]; then
        echo "WARNING: No more experiments to run."
        break
    fi
    exp_ids+=($exp_id)

    # Create directory for the source files in MongoDB
    if {with_sources}; then
        tmpdir="{tmp_directory}/$(uuidgen)"  # unique temp dir based on UUID
        # Prepend the temp dir to $PYTHONPATH so it will be used by python.
        exp_pypath="$tmpdir:$PYTHONPATH"
        tmp_dirs+=($tmpdir)
    fi

    # Prepare the epxeriment
    cmd=$(srun seml {db_collection_name} prepare-experiment -id ${{exp_id}}{prepare_args})

    # Check if the preparation was successful
    ret=$?
    if [ $ret -eq 0 ]; then
        # This experiment works and will be started.
        PYTHONPATH=$exp_pypath srun bash -c "$cmd" &
        process_ids+=($!)
    elif [ $ret -eq 3 ]; then
        echo "ERROR: Experiment with id ${{exp_id}} got claimed by this job but is not associated correctly."
    elif [ $ret -eq 4 ]; then
        (>&2 echo "ERROR: Experiment with id ${{exp_id}} not found in the database.")
    fi
done

# Kill unnecessary jobs
seml {db_collection_name} clean-jobs ${{all_exp_ids[@]}}

# Print process information
echo "Experiments are running under the following process IDs:"
num_it=${{#process_ids[@]}}
for ((i=0; i<$num_it; i++)); do
    echo "Experiment ID: ${{exp_ids[$i]}}	Process ID: ${{process_ids[$i]}}"
done
echo

# Wait for all experiments to finish
wait

# Delete temporary source files
if {with_sources}; then
    for tmpdir in ${{tmp_dirs[@]}}; do
        srun rm -rf $tmpdir
    done
fi


# Execute optional bash commands
{end_command}
