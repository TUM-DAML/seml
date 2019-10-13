# `SEML`: Slurm Experiment Management Library

Keeping track of computational experiments can be annoying and failure to do so can lead to lost results, duplicate running of the same experiments, and lots of headaches. While workload scheduling systems such as [`Slurm`](https://slurm.schedmd.com/overview.html) make it easy to run many experiments in parallel on a cluster, it can be hard to keep track of which parameter configurations are running, failed, or completed.

[`sacred`](https://github.com/IDSIA/sacred) is a great tool to collect and manage experiments and their result, but is lacking integration with workload schedulers.

**`SEML`** is the missing link between the open-source workload scheduling system `Slurm` and the experiment management tool `sacred`.

**`SEML`** enables you to 
* very easily define hyperparameter search spaces using YAML files,
* run these hyperparameter configurations on a compute cluster using `Slurm`,
* and to track the experimental results using `sacred` and [`MongoDB`](https://www.mongodb.com/).

## Get started
To get started, install **`SEML`** using the following commands:
```bash
git clone https://github.com/TUM-KDD/seml.git
cd seml
pip install -r requirements.txt
python setup.py develop
cp mongodb.config.example mongodb.config
# modify mongodb config to reflect your setup:
vim mongodb.config
```
## Example
See our simple [example](examples) to get familiar with how **`SEML`** works.

## Contact
Contact us at zuegnerd@in.tum.de or klicpera@in.tum.de for any questions.

Copyright (C) 2019  
Daniel Zügner and Johannes Klicpera  
Technical University of Munich
