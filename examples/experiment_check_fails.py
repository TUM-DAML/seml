import logging
from sacred import Experiment
import numpy as np
from seml import database_utils as db_utils
from seml import misc


ex = Experiment()
misc.setup_logger(ex)


@ex.post_run_hook
def collect_stats():
    misc.collect_exp_stats(ex)


@ex.config
def config():
    observe_slack = False
    observe_neptune = False
    if observe_slack:
        pass
    if observe_neptune:
        pass
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.capture(prefix="model")
def init_model(layer, activation):
    print("init_model")


@ex.capture(prefix="optimizer")
def init_optimizer(opt, lr):
    print("init_optimizer")


@ex.automain
def run(dataset):
    print("automain")
    init_model()
    init_optimizer()
    print(dataset)
    return True
