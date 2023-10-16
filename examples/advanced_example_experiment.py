"""
This is an advanced experiment example, which makes use of sacred's captured functions with prefixes.
We wrap all the experiment-specific functionality inside the "ExperimentWrapper" class, and define methods with sacred's
@ex.capture decorator. This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data")
are parsed by a specific method. This avoids having one large "main" function which takes all parameters as input.
"""

from sacred import Experiment
import numpy as np
import seml


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.named_config
def preprocessing_none():
    """ A named configuration that can be enabled in the configuration yaml file """
    preprocessing = {
        'mean' : 0.0,
        'std' : 1.0,
    }
    
@ex.named_config
def preprocessing_normalize():
    """ A named configuration that can be enabled in the configuration yaml file """
    preprocessing = {
        'mean' : 0.377,
        'std' : 0.23,
    }
    
@ex.named_config
def batchnorm():
    """ A named configuration that can be enabled in the configuration yaml file """
    model = {'batchnorm' : True}

@ex.named_config
def no_batchnorm():
    """ A named configuration that can be enabled in the configuration yaml file """
    model = {'batchnorm' : False, 'residual' : False}


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))
    name = '${model.model_type}_${data.dataset}'


class ModelVariant1:
    """
    A dummy model variant 1, which could, e.g., be a certain model or baseline in practice.
    """
    def __init__(self, hidden_sizes, dropout, batchnorm, residual):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual

class ModelVariant2:
    """
    A dummy model variant 2, which could, e.g., be a certain model or baseline in practice.
    """
    def __init__(self, hidden_sizes, dropout, batchnorm, residual):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        if dataset == "large_dataset_1":
            self.data = "load_dataset_here"
        elif dataset == "large_dataset_2":
            self.data = "and so on"
        # ...
        else:
            self.data = "..."

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, model_params: dict, batchnorm: bool, residual: bool=True):
        if model_type == "variant_1":
            # Here we can pass the "model_params" dict to the constructor directly, which can be very useful in
            # practice, since we don't have to do any model-specific processing of the config dictionary.
            self.model = ModelVariant1(**model_params, batchnorm=batchnorm, residual=residual)
        elif model_type == "variant_2":
            self.model = ModelVariant2(**model_params, batchnorm=batchnorm, residual=residual)

    @ex.capture(prefix="optimization")
    def init_optimizer(self, regularization: dict, optimizer_type: str):
        weight_decay = regularization['weight_decay']
        self.optimizer = optimizer_type  # initialize optimizer

    @ex.capture(prefix='preprocessing')
    def init_preprocessing(self, mean: float, std: float):
        self.preprocessing_parameters = (mean, std)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_preprocessing()

    @ex.capture(prefix="training")
    def train(self, patience, num_epochs):
        # everything is set up
        for e in range(num_epochs):
            # simulate training
            continue
        results = {
            'test_acc': 0.5 + 0.3 * np.random.randn(),
            'test_loss': np.random.uniform(0, 10),
            # ...
        }
        return results


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()
