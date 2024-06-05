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

# Chunked list with all experiment IDs
all_exp_ids=({exp_ids})

# Get experiment IDs for this Slurm task
exp_ids_str="${{all_exp_ids[$SLURM_ARRAY_TASK_ID]}}"
IFS=";" read -r -a exp_ids <<< "$exp_ids_str"

# Create directory for the source files in MongoDB
if {with_sources}; then
    tmpdir="{tmp_directory}/$(uuidgen)"  # unique temp dir based on UUID
    # Prepend the temp dir to $PYTHONPATH so it will be used by python.
    export PYTHONPATH="$tmpdir:$PYTHONPATH"
fi

# Start experiments in separate processes
process_ids=()
for exp_id in "${{exp_ids[@]}}"; do
    cmd=$(srun seml {db_collection_name} prepare-experiment -id ${{exp_id}} {prepare_args})

    ret=$?
    if [ $ret -eq 0 ]; then
        srun bash -c "$cmd" &
        process_ids+=($!)
    elif [ $ret -eq 3 ]; then
        echo "WARNING: Experiment with ID ${{exp_id}} does not have status PENDING and will not be run."
    elif [ $ret -eq 4 ]; then
        (>&2 echo "ERROR: Experiment with id ${{exp_id}} not found in the database.")
    fi
done

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
    srun rm -rf $tmpdir
fi


# Execute optional bash commands
{end_command}
