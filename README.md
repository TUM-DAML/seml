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
or `conda` (the conda version may be outdated, we highly recommend the pypi version!):
```bash
conda install -c conda-forge seml
```
Then configure your MongoDB via:
```bash
seml configure
```
For convenience, you may create your first **`SEML`** project via:
```bash
# initialize the default template in the 'new_project' folder.
seml project init -t default new_project
```
in an empty directoy. **`SEML`** will automatically create a python package for you.


### SSH Port Forwarding
If your MongoDB is only accessible via an SSH port forward, **`SEML`** allows you to directly configure this as well if you install the `ssh_forward` dependencies via:
```bash
pip install seml[ssh_forward]
```
It remains to configure the SSH settings:
```bash
seml configure --ssh_forward
```

### Development
If you want to develop `seml` please clone the repository and install it via
```bash
pip install -e .[dev]
```
and install pre-commit hooks via
```bash
pre-commit install
```

## Documentation
Documentation is available in our [docs.md](docs.md) or via the CLI:
```python
seml --help
```

## Example
See our simple [example](examples) to get familiar with how **`SEML`** works.

## CLI completion
SEML supports command line completion. To install this feature run:
```bash
seml --install-completion {shell}
```

If you are using the zsh shell, you might have to append `compinit -D` to the `~/.zshrc` file (see this [issue](https://github.com/tiangolo/typer/issues/180#issuecomment-812620805)).

## Slurm version

SEML should work with Slurm 18.08 and above out of the box. Version 17.11 and earlier do not have a SIGNALING job state, which you have to remove from the SLURM_STATES defined in SEML's settings (`seml/settings.py`). Earlier versions have not been tested and might have other issues.

## Contact
Contact us at zuegnerd@in.tum.de, johannes.gasteiger@tum.de, or n.gao@tum.de for any questions.

## Cite
When you use SEML in your own work, please cite the software along the lines of the following bibtex:

```
@software{seml_2023,
  author = {Z{\"u}gner, Daniel and Gasteiger, Johannes and Gao, Nicholas and Dominik Fuchsgruber},
  title = {{SEML: Slurm Experiment Management Library}},
  url = {https://github.com/TUM-DAML/seml},
  version = {0.4.0},
  year = {2023}
}
```


Copyright (C) 2023
Daniel Zügner, Johannes Gasteiger, Nicholas Gao, Dominik Fuchsgruber
Technical University of Munich
