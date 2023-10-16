# Start a Jupyter job
To start a Jupyter instance, you can use the convenience function `seml jupyter`. This requires having Jupyter Notebook or Jupyter Lab installed in the current (or specified) environment.

To modify the default Slurm `SBATCH`
options, see `seml/settings.py`. The easiest way of changing these is via a file in `$HOME/.config/seml/settings.py`.
This file must contain a `SETTINGS` dictionary, structured in the same way as the one in `seml/settings.py`.

After the Jupyter instance has successfully started, `seml` will provide useful information such as the hostname and 
port of the instance, e.g.:
```
Started Jupyter job in Slurm job with ID 12345.
The logfile of the job is /nfs/homedirs/zuegnerd/libraries/seml/slurm-6322311.out.
Trying to fetch the machine and port of the Jupyter instance once the job is running... (ctrl-C to cancel).
Jupyter instance is starting up...
Startup completed. The Jupyter instance is running at 'gpuxx.kdd.in.tum.de:8889'.
To stop the job, run 'scancel 12345'.
```
# Experiment tracking example
This example will show you how to track your experiments using Sacred, how to perform hyperparameter search and how to perform the experiments in a distributed manner on our Slurm cluster.


## MongoDB configuration
Before starting, please make sure you have your MongoDB credentials stored in `$HOME/.config/seml/mongodb.config`. The easiest way to do so is to run `seml configure`, which will store your credentials in the correct format in the right place.


## Experiment configuration

In `example_config.yaml` we define the parameter configurations that will be run. 
For a more advanced example with modular structure using 
[Sacred prefixes](https://sacred.readthedocs.io/en/stable/configuration.html#prefix), 
see the [advanced example configuration](advanced_example_config.yaml) and the corresponding 
[experiment](advanced_example_experiment.py).
<details><summary><b>Example config file</b></summary>
  
```yaml
seml:
  executable: examples/example_experiment.py
  name: example_experiment
  output_dir: examples/logs
  project_root_dir: ..

slurm:
  experiments_per_job: 1
  sbatch_options:
    gres: gpu:1       # num GPUs
    mem: 16G          # memory
    cpus-per-task: 2  # num cores
    time: 0-08:00     # max time, D-HH:MM

###### BEGIN PARAMETER CONFIGURATION ######

fixed:
  max_epochs: 500

grid:

  learning_rate:
    type: loguniform
    min: 1e-5
    max: 1e-1
    num: 5

random:
  samples: 3
  seed: 821

  # SEML supports dot-notation for nested dictionaries.
  regularization_params.dropout:
    type: uniform
    min: 0.0
    max: 0.7
    seed: 222

small_datasets:

  grid:
    dataset:
      type: choice
      options:
        - small_dataset_1
        - small_dataset_2

    hidden_sizes:
      type: choice
      options:
        - [16]
        - [32, 16]  # this will be parsed into a Python list.

  random:
    samples: 3
    seed: 2223

    max_epochs:
       type: randint
       min: 200
       max: 1000

large_datasets:

  fixed:
    max_epochs: 1000

  grid:
    learning_rate:
      type: choice
      options:
        - 0.001

    dataset:
      type: choice
      options:
        - large_dataset_1
        - large_dataset_2

    hidden_sizes:
      type: choice
      options:
        - [64]
        - [64, 32]
```
</details>
There are two special blocks for meta-configuration: `seml` and `slurm`.

### `seml` block
The `seml` block is required for every experiment. It has to contain the following values:
   - `executable`: Name of the Python script containing the experiment. The path should be relative to the `project_root_dir`.
                   For backward compatibility SEML also supports paths relative to the location of the config file.
                   In case there are files present both relative to the project root and the config file, the former takes precedence.
Optionally, it can contain
   - `name`: Prefix for output file and Slurm job name. Default: Collection name
   - `output_dir`: Directory to store log files in. Default: Current directory
   - `conda_environment`: Specifies which Anaconda virtual environment will be activated before the experiment is executed. 
                          Default: The environment used when queuing.
   - `project_root_dir`: (Relative or absolute) path to the root of the project. seml will then upload all the source
                         files imported by the experiment to the MongoDB. Moreover, the uploaded source files will be
                         downloaded before starting an experiment, so any changes to the source files in the project
                         between staging and starting the experiment will have no effect.
### `slurm` block
The special 'slurm' block contains the slurm parameters. This block and all values are optional. Possible values are:
   - `experiments_per_job`: Number of parallel experiments to run in each Slurm job. Note that only experiments from the same batch share a job. Default: 1
   - `max_simultaneous_jobs`: Maximum number of simultaneously running Slurm jobs per job array. Default: No restriction
   - `sbatch_options_template`: Name of a custom template of `SBATCH` options. Define your own templates in `settings.py`
     under `SBATCH_OPTIONS_TEMPLATES`, e.g. for long-running jobs, CPU-only jobs, etc.
   - `sbatch_options`: dictionary that contains custom values that will be passed to `sbatch`, specifying e.g. the
                       memory and the number of GPUs to be allocated. See [here](https://slurm.schedmd.com/sbatch.html)
                       for possible parameters of `sbatch` (prepended dashes are not required). Values provided here 
                       overwrite any values defined in a `SBATCH` options template.

### Sub-configurations
In the `small_datasets` and `large_datasets` (names are of course only examples; you can name sub-configs as you like) we have specified different sets of parameters to try.
They will be combined with the parameters in `grid` in the root of the document.

If a specific configuration (e.g. `large_datasets`) defines the same parameters as a higher-level configuration (e.g., the "root" configuration),
 they will override the ones defined before, e.g. the learning rate in the example above.
This means that for all configurations in the `large_datasets` the learning rate will be `0.001` and not `0.01` or 
`0.05` as defined in the root of the document.
This can be nested arbitrarily deeply (be aware of combinatorial explosion of the parameter space, though).

If a parameter is defined in (at least) two **different blocks** in `[grid, random, fixed]` on the same level, `seml` will throw an error to avoid ambiguity.
If a parameter is re-defined in a sub-configuration, the redefinition overrides any previous definitions of that parameter.

### Grid parameters
In an experiment config, under `grid` you can define parameters that should be sampled from a regular grid. Currently supported
are:
   - `choice`: List the different values you want to evaluate under `options`.
   - `range`: Specify the `min`, `max`, and `step`. Parameter values will be generated using `np.arange(min, max, step)`.
   - `uniform`: Specify the `min`, `max`, and `num`. Parameter values will be generated using
                `np.linspace(min, max, num, endpoint=True)`
   - `loguniform`: Specify `min`, `max`, and `num`. Parameter values will be uniformly generated in log space (base 10).

Additionally, `grid` parameters might be coupled by setting the `zip_id` property. All parameters with the same `zip_id` are treated as a single dimension when constructing the cartesian product of parameters. This ensures that zipped parameters only change jointly.

### Random parameters
Under 'random' you can specify parameters for which you want to try several random values. Specify the number
of samples per parameter with the `samples` value and optionally the random seed with `seed` as in the examples below. Supported parameter types are:
  - `choice`: Randomly samples `<samples>` entries (with replacement) from the list in `options`
  - `uniform`: Uniformly samples between `min` and `max` as specified in the parameter dict.
  - `loguniform`:  Uniformly samples in log space between `min` and `max` as specified in the parameter dict.
  - `randint`: Randomly samples integers between `min` (included) and `max` (excluded).

### Named Configurations
`sacred`, the on which experiments are based on, allows to define subgroups of configurations via its [named configurations](https://sacred.readthedocs.io/en/stable/configuration.html#named-configurations) feature. These can either be defined in external files (yaml, json, ...) or in functions decorated with `experiment.named_config`. SEML also supports this functionality by defining parameter groups that have the prefix `'+'`. Two config values can be defined for such parameter groups:
- `name`: The name of the named config, i.e. the name of the python function or the path to the file to load
- `priority`: Defines in which order the named configs will be loaded. Configs with lower priority will be listed first and thus resolved first. Therefore, the highest priority item will have the highest precedence. If no priority is given, this will be treated as `infinity`. Ties are broken based on the name of the named config.


## Add experiments to database

All SEML commands follow the pattern
```
seml [database_collection_name] [command] [command_options]
```

To insert the experiments to the database, open a terminal on a machine with access to the `Slurm` system. Move to this directory and run

```
seml seml_example add example_config.yaml
```

If you open your MongoDB (e.g. with the software `robo3t`), you should now find a collection `seml_example` with the staged experiments.
Note that the collection name is specified _before_ the operation (`add`).

To see what the option `--force-duplicates` does, run the above command again. The output should now read something like:

```
72 of 72 experiments were already found in the database. They were not added again.
```

That is, the script checks whether experiments with the same configuration are already in the database collection.
In this case, they are not added to the database to avoid redundant computations. In order to force add duplicates to the database, use the `--force-duplicates` argument.

All experiments are now already in the database collection you specified and in the STAGED state.

## Run experiments using Slurm
To run the staged experiments on the Slurm cluster, run:
```bash
seml seml_example start
```
This will start all experiments in the MongoDB collection `seml_example` that currently are in the STAGED state.

## Run experiments locally
You can also run your experiments locally without Slurm. For this, add the `--local` option:
```bash
seml seml_example start --local
```
You can even have multiple local workers running jobs in parallel. To add a local worker, run
```bash
seml seml_example launch-worker --worker-gpus="1" --worker-cpus=8
```
In this example, the worker will use the GPU with ID 1 (i.e., set `CUDA_VISIBLE_DEVICES="1"`) and can use 8 CPU cores.

The `--steal-slurm` option allows local workers to pop experiments from the Slurm queue. Since SEML checks the
database state of each experiment before actually executing it via Slurm, there is no risk of running duplicate 
experiments.

## Debugging experiments

To run an interactive debug session on Slurm (or locally) you can start an experiment with the `--debug` option.

For even more convenience you can also use VS Code for a remote debug session. First make sure that your experiments were added to the database with the `--no-code-checkpoint` option:

```
seml seml_example add example_config.yaml -ncc
```

This will prevent the caching of your code in the MongoDB and allow you to directly run the code that is in your working directory, set breakpoints and interactively step through your code in VS Code.

To start a remote debug server run:

```
seml seml_example start --debug-server
```

This will add your experiment to the queue, wait for the necessary resources to be assigned, spawn a debug process on the server and print the debug server's IP address and port number. The experiment will only start running once the VS Code client is attached.

To attach to the debug server you need to add the printed IP address and port number to the `.vscode/launch.json` config:
```
{
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "YOUR_DEBUG_SERVER_IP",
                "port": YOUR_DEBUG_SERVER_PORT
            }
        }
    ]
}
```
The IP address and port number of the debug server might change at every start, so make sure to update the `host` and `port` launch config. 
Note: The "restart" operation of the VS Code Debugger is not supported. 

## Running multiple experiments per Slurm job
Often a single experiment does not fully utilize the GPU and requires much less GPU RAM than available. Thus, we can often
run multiple experiments per Slurm job (which commonly uses a single GPU) to increase the throughput of our experiments.
This can be done by setting the `experiments_per_job` argument in the `slurm` block of the config file.

Note that this will only run your own experiments in parallel on a GPU. It will never run
your experiments on a GPU that is reserved by another user's job.
Furthermore, only experiments from the same batch share jobs.

## Check the status of your Slurm jobs

You can check the status of your Slurm jobs by running `squeue` or `seml seml_example status`
in the terminal. To check the console output of a experiment, open the corresponding logfile, e.g. `cat slurm-564.out`.

To check whether some experiments may have failed due to errors, you can run:
```bash
seml seml_example status
```

You can cancel (interrupt) all pending and running experiments with
```bash
seml seml_example cancel
```

You can reset all failed, killed, or interrupted experiments to STAGED with
```bash
seml seml_example reset
```

You can delete all staged, failed, killed, or interrupted experiments with
```bash
seml seml_example delete
```

These three commands also support passing a specific Sacred ID and a custom list of states.

Moreover, you can specifically cancel/reset/delete experiments that match a custom dictionary, e.g.
```bash
seml seml_example cancel --filter-dict '{"config.dataset":"cora_ml", "config.hidden_sizes": [16]}'
```

Finally, you can manually detect experiments whose corresponding Slurm jobs were killed unexpectedly with
```bash
seml seml_example detect-killed
```
(Detection is run automatically when executing the `status`, `delete`, `reset`, and `cancel` commands and therefore rarely necessary to do manually.)

### Batches
`seml` assigns each experiment a batch ID, where all experiments that were staged together get the same batch ID. 
You can use this to cancel all the experiments from the last configuration that you've started, e.g. if you find a bug. 
Use
```bash
seml seml_example cancel --batch-id i
```
or equivalently
 ```bash
seml seml_example cancel --filter-dict '{"batch_id": i}'
```
to cancel all jobs from batch `i`.

## Retrieve and evaluate results
See the [example notebook](notebooks/experiment_results.ipynb) for an example of how to retrieve and evaluate our toy experiment's results.


## Command chaining
`seml` also supports command chaining to execute multiple `seml` commands sequentially, i.e.,
```bash
seml seml_example add advanced_example_config.yaml start
```
to add a config file and start it immediately after or 
```
seml seml_example cancel -y reset -y reload-sources start
```
to cancel experiments, reset them, reload their source files and restarting them.
