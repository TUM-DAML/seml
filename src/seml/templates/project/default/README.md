# Project Template

This template combines three libraries to give you some basic training infrastructure:

- [seml](https://github.com/TUM-DAML/seml/) to load configuration files and run jobs


## Installation (Quick Guide)
Install your project editably with dependencies
```sh
pip install -e .
```
*Optionally*: Install pre-commit hooks via
```sh
pre-commit install
``` 

## Developement

**IDE**

We recommend [VS Code](https://code.visualstudio.com) for development. Select the conda environment you created earlier as your default python interpreter. *Optionally*, use static typecheckers and linters like [ruff](https://github.com/astral-sh/ruff).

**Sacred**

`seml` is based on [Sacred](https://sacred.readthedocs.io/en/stable/index.html). Familiarize yourself with the rough concept behind this framework. Importantly, understand how [experiments](https://sacred.readthedocs.io/en/stable/experiment.html) work and how they can be [configured](https://sacred.readthedocs.io/en/stable/experiment.html#configuration) using config overrides and `named configs`.

**MongoDB**

`seml` will log your experiments on our local `MongoDB` server after you set it up according to the [installation guide]((https://github.com/TUM-DAML/seml/)). Familiarize yourself with the core functionality of `seml` experiments from the example configurations.


## Running experiments locally

To start a training locally, call `main.py` with the your settings, for example

```sh
./main.py with config/data/small.yaml config/model/big.yaml
```

You can use this for debugging, e.g. in an interactive slurm session or on your own machine.

## Running experiments on the cluster

Use `seml` to run experiments on the cluster. Pick a collection name, e.g. `example_experiment`. Each experiment should be referred to with an configuration file in `experiments/`. Use the `seml.description` field to keep track of your experiments. Add experiments using:

```bash
seml {your-collection-name} add config/seml/grid.yaml
```

Run them on the cluster using:

```bash
seml {your-collection-name} start
```

You can monitor the experiment using:

```bash
seml {your-collection-name} status
```

More advanced usage of seml can be found in the [documentation](https://github.com/TUM-DAML/seml/tree/master/examples).


## Analyzing results

You can analyze the results by inspecting output files your code generates or values you log in the MongoDB. For reference, see `notebooks/visualize_results.ipynb`.
