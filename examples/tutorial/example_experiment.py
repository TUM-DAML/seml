import logging
from sacred import Experiment
import numpy as np
import seml
import time

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(hidden_sizes: list, learning_rate: float, max_epochs: int):
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info("Received the following configuration:")
    logging.info(
        f"Hidden sizes: {hidden_sizes}, learning_rate: {learning_rate}, "
        f"max_epochs: {max_epochs}"
    )
    # res = hidden_sizes / 2
    #  do your processing here
    time.sleep(60)
    results = {
        "test_acc": learning_rate * np.sqrt(np.arange(1, 1001, 1))
        + np.random.uniform(0, 5),
        # ...
    }
    # the returned result will be written into the database
    return results
