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
* `-V, --version`: Print the version number.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `add`: Add experiments to the database as defined...
* `cancel`: Cancel the Slurm job/job step...
* `clean-db`: Remove orphaned artifacts in the DB from...
* `configure`: Configure SEML (database, argument...
* `delete`: Delete experiments by ID or state (cancels...
* `description`: Manage descriptions of the experiments in...
* `detect-duplicates`: Prints duplicate experiment configurations.
* `detect-killed`: Detect experiments where the corresponding...
* `drop`: Drop collections from the database.
* `hold`: Hold queued experiments via SLURM.
* `launch-worker`: Launch a local worker that runs PENDING jobs.
* `list`: Lists all collections in the database.
* `print-command`: Print the commands that would be executed...
* `print-fail-trace`: Prints fail traces of all failed experiments.
* `print-output`: Print the output of experiments.
* `project`: Setting up new projects.
* `release`: Release holded experiments via SLURM.
* `reload-sources`: Reload stashed source files.
* `reset`: Reset the state of experiments by setting...
* `restore-sources`: Restore source files from the database to...
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
* `-d, --description TEXT`: A description for the experiment.
* `--no-resolve-descriptions`: Whether to prevent using omegaconf to resolve experiment descriptions
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

* `-sf, --ssh-forward`: Configure SSH forwarding settings for MongoDB.
* `--help`: Show this message and exit.

## `seml delete`

Delete experiments by ID or state (cancels Slurm jobs first if not --no-cancel).

**Usage**:

```console
$ seml delete [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: STAGED, QUEUED, FAILED, KILLED, INTERRUPTED]
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-nc, --no-cancel`: Do not cancel the experiments before deleting them.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml description`

Manage descriptions of the experiments in a collection.

**Usage**:

```console
$ seml description [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `delete`: Deletes the description of experiment(s).
* `list`: Lists the descriptions of all experiments.
* `set`: Sets the description of experiment(s).

### `seml description delete`

Deletes the description of experiment(s).

**Usage**:

```console
$ seml description delete [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

### `seml description list`

Lists the descriptions of all experiments.

**Usage**:

```console
$ seml description list [OPTIONS]
```

**Options**:

* `-u, --update-status`: Whether to update the status of experiments in the database. This can take a while for large collections. Use only if necessary.
* `--help`: Show this message and exit.

### `seml description set`

Sets the description of experiment(s).

**Usage**:

```console
$ seml description set [OPTIONS] DESCRIPTION
```

**Arguments**:

* `DESCRIPTION`: The description to set.  [required]

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--no-resolve-descriptions`: Whether to prevent using omegaconf to resolve experiment descriptions
* `--help`: Show this message and exit.

## `seml detect-duplicates`

Prints duplicate experiment configurations.

**Usage**:

```console
$ seml detect-duplicates [OPTIONS]
```

**Options**:

* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: STAGED, QUEUED, FAILED, KILLED, INTERRUPTED]
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `--help`: Show this message and exit.

## `seml detect-killed`

Detect experiments where the corresponding Slurm jobs were killed externally.

**Usage**:

```console
$ seml detect-killed [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `seml drop`

Drop collections from the database.

Note: This is a dangerous operation and should only be used if you know what you are doing.

**Usage**:

```console
$ seml drop [OPTIONS] [PATTERN]
```

**Arguments**:

* `[PATTERN]`: A regex that must match the collections to print.  [default: .*]

**Options**:

* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

## `seml hold`

Hold queued experiments via SLURM.

**Usage**:

```console
$ seml hold [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
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
* `-d, --debug`: Run a single interactive experiment without Sacred observers and with post-mortem debugging. Implies `--verbose --num-exps 1 --post-mortem --output-to-console`.
* `-ds, --debug-server`: Run the experiment with a debug server, to which you can remotely connect with e.g. VS Code. Implies `--debug`.
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
* `-u, --update-status`: Whether to update the status of experiments in the database. This can take a while for large collections. Use only if necessary.
* `-fd, --full-descriptions`: Whether to print full descriptions (possibly with line breaks).
* `--help`: Show this message and exit.

## `seml print-command`

Print the commands that would be executed by `start`.

**Usage**:

```console
$ seml print-command [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: STAGED, QUEUED]
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-n, --num-experiments INTEGER`: Number of experiments to start. 0: all (staged) experiments   [default: 0]
* `-wg, --worker-gpus TEXT`: The IDs of the GPUs used by the local worker. Will be directly passed to CUDA_VISIBLE_DEVICES.
* `-wc, --worker-cpus INTEGER`: The number of CPUs used by the local worker. Will be directly passed to OMP_NUM_THREADS.
* `-we, --worker-env JSON`: Further environment variables to be set for the local worker.
* `--unresolved`: Whether to print the unresolved command.
* `--no-interpolation`: Whether disable variable interpolation. Only compatible with --unresolved.
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
* `-p, --projection KEY`: List of configuration keys, e.g., `config.model`, to additionally print.
* `--help`: Show this message and exit.

## `seml print-output`

Print the output of experiments.

**Usage**:

```console
$ seml print-output [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.  [default: RUNNING, FAILED, KILLED, INTERRUPTED, COMPLETED]
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `--help`: Show this message and exit.

## `seml project`

Setting up new projects.

**Usage**:

```console
$ seml project [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `init`: Initialize a new project in the given...
* `list-templates`: List available project templates.

### `seml project init`

Initialize a new project in the given directory.

**Usage**:

```console
$ seml project init [OPTIONS] [DIRECTORY]
```

**Arguments**:

* `[DIRECTORY]`: The directory in which to initialize the project.  [default: .]

**Options**:

* `-t, --template TEXT`: The template to use for the project. To view available templates use `seml project list-templates`.  [default: default]
* `-n, --name TEXT`: The name of the project. (By default inferred from the directory name.)
* `-u, --username TEXT`: The author name to use for the project. (By default inferred from $USER)
* `-m, --usermail TEXT`: The author email to use for the project. (By default empty.)
* `-r, --git-remote TEXT`: The git remote to use for the project. (By default SETTINGS.TEMPLATE_REMOTE.)
* `-c, --git-commit TEXT`: The exact git commit to use. May also be a tag or branch (By default latest)
* `-y, --yes`: Automatically confirm all dialogues with yes.
* `--help`: Show this message and exit.

### `seml project list-templates`

List available project templates.

**Usage**:

```console
$ seml project list-templates [OPTIONS]
```

**Options**:

* `-r, --git-remote TEXT`: The git remote to use for the project. (By default SETTINGS.TEMPLATE_REMOTE.)
* `-c, --git-commit TEXT`: The exact git commit to use. May also be a tag or branch (By default latest)
* `--help`: Show this message and exit.

## `seml release`

Release holded experiments via SLURM.

**Usage**:

```console
$ seml release [OPTIONS]
```

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
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

## `seml restore-sources`

Restore source files from the database to the provided path.

**Usage**:

```console
$ seml restore-sources [OPTIONS] TARGET_DIRECTORY
```

**Arguments**:

* `TARGET_DIRECTORY`: The directory where the source files should be restored.  [required]

**Options**:

* `-id, --sacred-id INTEGER`: Sacred ID (_id in the database collection) of the experiment. Takes precedence over other filters.
* `-s, --filter-states [STAGED|QUEUED|PENDING|RUNNING|FAILED|KILLED|INTERRUPTED|COMPLETED]`: List of states to filter the experiments by. If empty (""), all states are considered.
* `-f, --filter-dict JSON`: Dictionary (passed as a string, e.g. '{"config.dataset": "cora_ml"}') to filter the experiments by.
* `-b, --batch-id INTEGER`: Batch ID (batch_id in the database collection) of the experiments. Experiments that were staged together have the same batch_id.
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

* `-u, --update-status`: Whether to update the status of experiments in the database. This can take a while for large collections. Use only if necessary.  [default: True]
* `-p, --projection KEY`: List of configuration keys, e.g., `config.model`, to additionally print.
* `--help`: Show this message and exit.
