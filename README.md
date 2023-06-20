![Github Actions](https://github.com/TUM-DAML/seml/workflows/Test/badge.svg)

# `SEML`: Slurm Experiment Management Library
**`SEML`** is the missing link between the open-source workload scheduling system `Slurm`, the experiment management tool `sacred`, and a `MongoDB` experiment database. It is lightweight, hackable, written in pure Python, and scales to thousands of experiments.

Keeping track of computational experiments can be annoying and failure to do so can lead to lost results, duplicate running of the same experiments, and lots of headaches.
While workload scheduling systems such as [`Slurm`](https://slurm.schedmd.com/overview.html) make it easy to run many experiments in parallel on a cluster, it can be hard to keep track of which parameter configurations are running, failed, or completed.
[`sacred`](https://github.com/IDSIA/sacred) is a great tool to collect and manage experiments and their results, especially when used with a [`MongoDB`](https://www.mongodb.com/). However, it is lacking integration with workload schedulers.

**`SEML`** enables you to 
* very easily define hyperparameter search spaces using YAML files,
* run these hyperparameter configurations on a compute cluster using `Slurm`,
* and to track the experimental results using `sacred` and `MongoDB`.


In addition, **`SEML`** offers many more features to make your life easier, such as
* automatically saving and loading your source code for reproducibility,
* easy debugging on Slurm or locally,
* automatically checking your experiment configurations,
* extending Slurm with local workers,
* and keeping track of resource usage (experiment runtime, RAM, etc.).

## Get started
To get started, install **`SEML`** either via `pip`:
```bash
pip install seml
```
or `conda`:
```bash
conda install -c conda-forge seml
```
Then configure your MongoDB via:
```bash
seml configure  --mongodb # provide your MongoDB credentials
```

## Example
See our simple [example](examples) to get familiar with how **`SEML`** works.

## CLI completion
SEML supports (rudimentary) command line completion. To install this feature run:
```bash
seml configure --argcomplete
```
There are three options available, we recommend to install the completion as `user`:
- `user`: Creates `~/.bash_completion`, which will be automatically sourced when a shell is opened
- `global`: Creates a similar file at a [suitable location](https://pypi.org/project/argcomplete/) depending on the shell being used. You need sudo rights to install argument completion globally.
- `path`: Queries for a specific path to which to install the argument completion script. This script needs to be sourced in every shell session in which argument completion is to be used.
In the background, this feature relies on the [argcomplete](https://pypi.org/project/argcomplete/) module. To disable and purge the feature you should uninstall this module and delete (the relevant parts of) the file created by `seml configure --argcomplete` (which depends on the installation mode you chose).


## Slurm version

SEML should work with Slurm 18.08 and above out of the box. Version 17.11 and earlier do not have a SIGNALING job state, which you have to remove from the SLURM_STATES defined in SEML's settings (`seml/settings.py`). Earlier versions have not been tested and might have other issues.

## Contact
Contact us at zuegnerd@in.tum.de, johannes.gasteiger@tum.de, or n.gao@tum.de for any questions.

## Cite
When you use SEML in your own work, please cite the software along the lines of the following bibtex:

```
@software{seml_2022,
  author = {Z{\"u}gner, Daniel and Gasteiger, Johannes and Gao, Nicholas},
  title = {{SEML: Slurm Experiment Management Library}},
  url = {https://github.com/TUM-DAML/seml},
  version = {0.3.7},
  year = {2022}
}
```


Copyright (C) 2023
Daniel Zügner, Johannes Gasteiger, and Nicholas Gao  
Technical University of Munich
