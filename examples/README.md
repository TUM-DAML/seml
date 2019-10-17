# Experiment tracking example
This example will show you how to track your experiments using Sacred, how to perform hyperparameter search and how to perform the experiments in a distributed manner on our Slurm cluster.


## MongoDB configuration
Before starting, please make sure you have your MongoDB credentials stored in `$HOME/.config/seml/mongodb.config`, e.g.:

```
username: <your_username>
password: <your_password>
host: mongodb-host.example.com
port: 27017
database: <database_name>
```
You can simply clone the example config in the root directory of this repo to get started:
```bash
mkdirs ~/.config/seml
cp mongodb.config.example ~/.config/seml/mongodb.config
```

## Experiment configuration

In `example_config.yaml` we define the parameter configurations that will be run.
<details><summary><b>Example config file</b></summary>
  
```yaml
seml:
  name: 'example_experiment'
  db_collection: 'example_experiment'
  executable: 'example_experiment.py'

slurm:
  output_dir: '.'
  experiments_per_job: 1
  sbatch_options:
    gres: 'gpu:1'     # num GPUs
    mem: 16000        # memory
    cpus-per-task: 1  # num cores
    time: '0-08:00'   # max time, D-HH:MM

###### BEGIN PARAMETER CONFIGURATION ######

fixed:
  reg_scale: 0.0
  keep_prob: 0.5
  max_epochs: 500
  patience: 10
  display_step: 25

grid:
  learning_rate:
    type: "loguniform"
    min: 1e-5
    max: 1e-1
    num: 5

random:
  samples: 3
  seed: 821

  max_epochs:
    type: "randint"
    min: 200
    max: 1000
    seed: 222

small_datasets:

  grid:
    dataset:
      type: "choice"
      options:
        - "small_dataset_1"
        - "small_dataset_2"

    hidden_sizes:
      type: "choice"
      options:
        - [16]
        - [32, 16]

  random:
    samples: 3
    seed: 2223

    reg_scale:
      type: "loguniform"
      min: 1e-9
      max: 1e-1

    keep_prob:
      type: "uniform"
      min: 0.3
      max: 1

    patience:
      type: "choice"
      options:
        - 10
        - 50

large_datasets:

  fixed:
    max_epochs: 1000

  grid:
    learning_rate:
      type: 'choice'
      options:
        - 0.001

    dataset:
      type: 'choice'
      options:
        - "large_dataset_1"
        - "large_dataset_2"

    hidden_sizes:
      type: 'choice'
      options:
        - [64]
        - [64, 32]
```
</details>
There are two special blocks for meta-configuration: `seml` and `slurm`.

### `seml` block
The `seml` block is required for every experiment. It has to contain the following values:
   - `db_collection`: Name of the MongoDB collection to save the experiment information to
   - `executable`: Name of the Python script containing the experiment
Optionally, it can contain
   - `conda_environment`: name of the Anaconda virtual environment; will be loaded before the experiment is executed.
### `slurm` block
The special 'slurm' block contains the slurm parameters. This block and all values are optional. Possible values are:
   - `name`: Job name used by Slurm and file name of Slurm output. Default: Collection name
   - `output_dir`: Directory to store the Slurm log files in. Default: Current directory
   - `experiments_per_job`: Number of parallel experiments to run in each Slurm job. Default: 1
   - `sbatch_options`: dictionary that contains custom values that will be passed to `sbatch`, specifying e.g. the
   memory and the number of GPUs to be allocated. See [here](https://slurm.schedmd.com/sbatch.html) for possible parameters of `sbatch`.

### Parameter blocks
In the `small_datasets` and `large_datasets` (names are of course only examples; you can name sub-configs as you like) we have specified different sets of parameters to try.
They will be combined with the parameters in `grid` in the root of the document.

If a specific configuration (e.g. `large_datasets`) defines the same parameters, they will override the ones defined in the root, e.g. the learning rate in the example above.
This means that for all configurations in the `large_datasets` the learning rate will be `0.001` and not `0.01` or `0.05` as defined in the root of the document.

This can be nested arbitrarily deeply (be aware of combinatorial explosion of the parameter space, though!)

### Grid parameters
In an experiment config, under `grid` you can define parameters that should be sampled from a regular grid. Currently supported
are:
   - `choice`: List the different values you want to evaluate under `options`.
   - `range`: Specify the `min`, `max`, and `step`. Parameter values will be generated using `np.arange(min, max, step)`.
   - `uniform`: Specify the `min`, `max`, and `num`. Parameter values will be generated using
              `np.linspace(min, max, num, endpoint=True)`
   - `loguniform`: Specify `min`, `max`, and `num`. Parameter values will be uniformly generated in log space (base 10).

### Random parameters
Under 'random' you can specify parameters for which you want to try several random values. Specify the number
of samples per parameter with the `samples` value as in the examples below. Supported parameter types are:
  - `choice`: Randomly samples `<samples>` entries (with replacement) from the list in `options`
  - `uniform`: Uniformly samples between `min` and `max` as specified in the parameter dict.
  - `loguniform`:  Uniformly samples in log space between `min` and `max` as specified in the parameter dict.
  - `randint`: Randomly samples integers between `min` (included) and `max` (excluded).

## Add experiments to queue

To insert the experiments the queue in the database, open a terminal on a machine with access to the `Slurm` system. In this directory and run

```
python /path/to/seml/main.py -c example_config.yaml queue
```

If you open your MongoDB (e.g. with the software `robo3t`), you should now find a collection `seml_example` with the queued experiments.
Note that the config file is specified _before_ the operation (`queue`).
Since all commands are run via this Python script, you might want to add `alias seml="python /path/to/seml/main.py"` to your bashrc.

To see what the option `--force-duplicates` does, run the above command again. The output should now read something like:

```
72 of 72 experiments were already found in the database. They were not added again.
```

That is, the script checks whether experiments with the same configuration are already in the database collection.
In this case, they are not added to the queue to avoid redundant computations. In order to force add duplicates to the database, use the `--force-duplicates` argument.

All experiments are now already in the database collection specified in `example_config.yaml` and in the QUEUED state.

## Run experiments using Slurm
To run the queued experiments on the Slurm cluster, run:
```bash
python /path/to/seml/main.py -c example_config.yaml start
```
The config file is only needed to specify the MongoDB collection. All other parameters will be loaded from the MongoDB.

### Running multiple experiments per Slurm job
Often, a single experiment requires much less GPU RAM than is available on a GPU. Thus, we can
often run multiple experiments per Slurm job (which commonly uses a single GPU) to increase the throughput of our experiments.
This can be done by setting the `experiments_per_job` argument in the `slurm` block of the config file.

Note that this will only run your own experiments in parallel on a GPU. It will never run
your experiments on a GPU that is reserved by another user's job.

## Check the status of your Slurm jobs

You can check the status of your Slurm jobs by running `squeue` or `python ~/path/to/seml/main.py -c example_config.yaml status`
in the terminal. To check the console output of a experiment, open the corresponding logfile, e.g. `cat slurm-564.out`.

To check whether some experiments may have failed due to errors, you can run:
```bash
python /path/to/seml/main.py -c example_config.yaml status
```

You can delete all queued, failed, killed, or interrupted experiments with
```bash
python /path/to/seml/main.py -c example_config.yaml delete
```

You can reset all failed, killed, or interrupted experiments to QUEUED with
```bash
python /path/to/seml/main.py -c example_config.yaml reset
```

You can cancel (interrupt) all pending and running experiments with
```bash
python /path/to/seml/main.py -c example_config.yaml cancel
```

These three commands also support passing a specific Sacred ID and a custom list of states.

Moreover, you can specifically cancel/reset/delete experiments that match a custom dictionary, e.g.
```bash
python /path/to/seml/main.py -c example_config.yaml cancel --filter-dict '{"config.dataset":"cora_ml", "config.hidden_sizes": [16]}'
```

Finally, you can detect experiments whose corresponding Slurm jobs were killed unexpectedly with
```bash
python /path/to/seml/main.py -c example_config.yaml detect-killed
```

### Batches
`seml` assigns each experiment a batch ID, where all experiments that were queued together get the same batch ID. 
You can use this to cancel all the experiments from the last configuration that you've started, e.g. if you find a bug. 
Use
```bash
python /path/to/seml/main.py -c example_config.yaml cancel --batch-id i
```
or equivalently
 ```bash
python /path/to/seml/main.py -c example_config.yaml cancel --filter-dict '{"batch_id": i}'
```
to cancel all jobs from batch `i`.

## Retrieve and evaluate results
See the [example notebook](notebooks/experiment_results.ipynb) for an example of how to retrieve and evaluate our toy experiment's results.
