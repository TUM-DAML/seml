# `seml`

SEML - Slurm Experiment Management Library.

**Usage**:

```console
$ seml [OPTIONS] COLLECTION COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...
```

**Arguments**:

* `COLLECTION`: The name of the database collection to use.  [required]

**Options**:

* `-v, --verbose`: Whether to print debug messages.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `add`: Add experiments to the database as defined...
* `cancel`: Cancel the Slurm job/job step...
* `clean-db`: Remove orphaned artifacts in the DB from...
* `configure`: Configure SEML (database, argument...
* `delete`: Delete experiments by ID or state (does...
* `detect-killed`: Detect experiments where the corresponding...
* `launch-worker`: Launch a local worker that runs PENDING jobs.
* `list`: Lists all collections in the database.
* `print-fail-trace`: Prints fail traces of all failed experiments.
* `print_command`: Print the commands that would be executed...
* `reload-sources`: Reload stashed source files.
* `reset`: Reset the state of experiments by setting...
* `start`: Fetch staged experiments from the database...
* `start-jupyter`: Start a Jupyter slurm job.
* `status`: Report status of experiments in the...

## `seml add`

Add experiments to the database as defined in the configuration.

**Usage**:

```console
$ seml add [OPTIONS] CONFIG_FILES...
```

**Arguments**:

* `CONFIG_FILES...`: Path to the YAML configuration file for the experiment.  [required]

**Options**:

* `-nh, --no-hash`: By default, we use the hash of the config dictionary to filter out duplicates (by comparing all dictionary values individually). Only disable this if you have a good reason as it is faster.
* `-ncs, --no-sanity-check`: Disable this if the check fails unexpectedly when using advanced Sacred features or to accelerate adding.
* `-ncc, --no-code-checkpoint`: Disable this if you want your experiments to use the current codeinstead of the code at the time of adding.
* `-f, --force`: Force adding the experiment even if it already exists in the database.
* `-o, --overwrite-params JSON`: Dictionary (passed as a string, e.g. '{"epochs": 100}') to overwrite parameters in the config.
* `--help`: Show this message and exit.

## `seml cancel`

Cancel the Slurm job/job step corresponding to experiments, filtered by ID or state.

**Usage**:

```console
$ seml cancel [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: PENDING, RUNNING]
* `-w, --wait`: Wait until all jobs are properly cancelled.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml clean-db`

Remove orphaned artifacts in the DB from runs which have been deleted..

**Usage**:

```console
$ seml clean-db [OPTIONS]
```

**Options**:

* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml configure`

Configure SEML (database, argument completion, ...).

**Usage**:

```console
$ seml configure [OPTIONS]
```

**Options**:

* `-a, --all`: Configure all SEML settings
* `--mongodb / --no-mongodb`: Configure MongoDB settings  [default: mongodb]
* `--help`: Show this message and exit.

## `seml delete`

Delete experiments by ID or state (does not cancel Slurm jobs).

**Usage**:

```console
$ seml delete [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: STAGED, QUEUED, FAILED, KILLED, INTERRUPTED]
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml detect-killed`

Detect experiments where the corresponding Slurm jobs were killed externally.

**Usage**:

```console
$ seml detect-killed [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `seml launch-worker`

Launch a local worker that runs PENDING jobs.

**Usage**:

```console
$ seml launch-worker [OPTIONS]
```

**Options**:

* `-n, --num-experiments INTEGER`: Number of experiments to start. 0: all (staged) experiments   [default: 0]
* `-nf, --no-file-output`: Do not write the experiment's output to a file.
* `-ss, --steal-slurm`: Local jobs 'steal' from the Slurm queue, i.e. also execute experiments waiting for execution via Slurm.
* `-pm, --post-mortem`: Activate post-mortem debugging with pdb.
* `-o, --output-to-console`: Write the experiment's output to the console.
* `-wg, --worker-gpus TEXT`: The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.
* `-wc, --worker-cpus INTEGER`: The number of CPUs used by the local worker. Will be directly passed to OMP_NUM_THREADS.
* `-we, --worker-env JSON`: Further environment variables to be set for the local worker.
* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `--help`: Show this message and exit.

## `seml list`

Lists all collections in the database.

**Usage**:

```console
$ seml list [OPTIONS] [PATTERN]
```

**Arguments**:

* `[PATTERN]`: A regex that must match the collections to print.  [default: .*]

**Options**:

* `-p, --progress`: Whether to print a progress bar for iterating over collections.
* `--help`: Show this message and exit.

## `seml print-fail-trace`

Prints fail traces of all failed experiments.

**Usage**:

```console
$ seml print-fail-trace [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: FAILED, KILLED, INTERRUPTED]
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml print_command`

Print the commands that would be executed by `start`.

**Usage**:

```console
$ seml print_command [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-n, --num-experiments INTEGER`: Number of experiments to start. 0: all (staged) experiments   [default: 0]
* `-wg, --worker-gpus TEXT`: The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.
* `-wc, --worker-cpus INTEGER`: The number of CPUs used by the local worker. Will be directly passed to OMP_NUM_THREADS.
* `-we, --worker-env JSON`: Further environment variables to be set for the local worker.
* `--help`: Show this message and exit.

## `seml reload-sources`

Reload stashed source files.

**Usage**:

```console
$ seml reload-sources [OPTIONS]
```

**Options**:

* `-k, -keep-old`: Keep the old source files in the database.
* `-b, --batch-ids INTEGER`: Batch IDs (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml reset`

Reset the state of experiments by setting their state to STAGED and cleaning their database entry.
Does not cancel Slurm jobs.

**Usage**:

```console
$ seml reset [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: FAILED, KILLED, INTERRUPTED]
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml start`

Fetch staged experiments from the database and run them (by default via Slurm).

**Usage**:

```console
$ seml start [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-d, --debug`: Run a single interactive experiment without Sacred observers and with post-mortem debugging. Implies `--verbose --num-exps 1 --post-mortem --output-to-console`.
* `-ds, --debug-server`: Run the experiment with a debug server, to which you can remotely connect with e.g. VS Code. Implies `--debug`.
* `-l, --local`: Run the experiment locally instead of on a Slurm cluster.
* `-nw, --no-worker`: Do not launch a local worker after setting experiments' state to PENDING.
* `-n, --num-experiments INTEGER`: Number of experiments to start. 0: all (staged) experiments   [default: 0]
* `-nf, --no-file-output`: Do not write the experiment's output to a file.
* `-ss, --steal-slurm`: Local jobs 'steal' from the Slurm queue, i.e. also execute experiments waiting for execution via Slurm.
* `-pm, --post-mortem`: Activate post-mortem debugging with pdb.
* `-o, --output-to-console`: Write the experiment's output to the console.
* `-wg, --worker-gpus TEXT`: The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.
* `-wc, --worker-cpus INTEGER`: The number of CPUs used by the local worker. Will be directly passed to OMP_NUM_THREADS.
* `-we, --worker-env JSON`: Further environment variables to be set for the local worker.
* `--help`: Show this message and exit.

## `seml start-jupyter`

Start a Jupyter slurm job. Uses SBATCH options defined in settings.py under
SBATCH_OPTIONS_TEMPLATES.JUPYTER

**Usage**:

```console
$ seml start-jupyter [OPTIONS]
```

**Options**:

* `-l, --lab`: Start a jupyter-lab instance instead of jupyter notebook.
* `-c, --conda-env TEXT`: Start the Jupyter instance in a Conda environment.
* `-sb, --sbatch-options JSON`: Dictionary (passed as a string, e.g. '{"gres": "gpu:2"}') to request two GPUs.
* `--help`: Show this message and exit.

## `seml status`

Report status of experiments in the database collection.

**Usage**:

```console
$ seml status [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
