import logging

import numpy as np

from seml import Experiment

ex = Experiment()


@ex.automain
def run(
    dataset: str,
    hidden_sizes: list,
    learning_rate: float,
    max_epochs: int,
    regularization_params: dict,
):
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:')
    logging.info(
        f'Dataset: {dataset}, hidden sizes: {hidden_sizes}, learning_rate: {learning_rate}, '
        f'max_epochs: {max_epochs}, regularization: {regularization_params}'
    )

    #  do your processing here

    results = {
        'test_acc': 0.5 + 0.3 * np.random.randn(),
        'test_loss': np.random.uniform(0, 10),
        # ...
    }
    # the returned result will be written into the database
    return results
