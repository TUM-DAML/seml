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


@ex.automain
def run(dataset: str, hidden_sizes: list, learning_rate: float, max_epochs: int,
        regularization_params: dict):
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:')
    logging.info(f'Dataset: {dataset}, hidden sizes: {hidden_sizes}, learning_rate: {learning_rate}, '
                 f'max_epochs: {max_epochs}, regularization: {regularization_params}')

    #  do your processing here

    results = {
        'test_acc': 0.5 + 0.3 * np.random.randn(),
        'test_loss': np.random.uniform(0, 10),
        # ...
    }
    # the returned result will be written into the database
    return results
