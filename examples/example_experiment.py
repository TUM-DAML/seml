import logging
from sacred import Experiment
import numpy as np
from seml import database_utils as db_utils
from seml import misc


ex = Experiment()
misc.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(dataset, hidden_sizes, learning_rate, reg_scale, keep_prob, max_epochs, patience, display_step, regularization_params):

    logging.info('Received the following configuration:')
    logging.info(f'Dataset: {dataset}, hidden sizes: {hidden_sizes}, learning_rate: {learning_rate},'
                 f'reg_scale: {reg_scale}, keep_prob:{keep_prob}, max_epochs: {max_epochs}, patience:{patience}, '
                 f'display_step: {display_step}')

    #  do your processing here

    results = {
        'test_acc': 0.5 + 0.3 * np.random.randn(),
        'test_loss': np.random.uniform(0, 10),
        # ...
    }
    # the returned result will be written into the database
    return results


misc.collect_exp_stats(ex)
