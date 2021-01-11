import logging
from sacred import Experiment
import numpy as np
import seml


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ModelVariant1:
    def __init__(self, hidden_sizes, dropout):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout


class ModelVariant2:
    def __init__(self, hidden_sizes, dropout):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout


class ExperimentWrapper:

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset):
        if dataset == "large_dataset_1":
            self.data = "load_dataset_here"
        elif dataset == "large_dataset_2":
            self.data = "and so on"
        # ...
        else:
            self.data = "..."

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, model_params: dict):
        if model_type == "variant_1":
            # Here we can pass the "model_params" dict to the constructor directly.
            self.model = ModelVariant1(**model_params)
        elif model_type == "variant_2":
            self.model = ModelVariant2(**model_params)

    @ex.capture(prefix="optimization")
    def init_optimizer(self, regularization: dict, optimizer_type: str):
        weight_decay = regularization['weight_decay']
        self.optimizer = optimizer_type  # initialize optimizer

    def init_all(self):
        self.init_dataset()
        self.init_model()
        self.init_optimizer()

    @ex.capture(prefix="training")
    def train(self, patience, num_epochs):
        # everything is set up
        for e in range(num_epochs):
            continue
        results = {
            'test_acc': 0.5 + 0.3 * np.random.randn(),
            'test_loss': np.random.uniform(0, 10),
            # ...
        }
        return results


@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()