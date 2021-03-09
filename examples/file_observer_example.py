import logging
from sacred import Experiment
import numpy as np
import seml
import pickle
import gzip
import os


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    use_file_observer = False
    mongodb_observer = None

    if db_collection is not None:
        mongodb_observer = seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        ex.observers.append(mongodb_observer)
    if use_file_observer:
        file_observer_base_dir = None
        runs_folder_name = db_collection
        if file_observer_base_dir is not None:
            if file_observer_base_dir.startswith("/"):  # use absolute path provided in configuration
                file_observer_base_dir = file_observer_base_dir
            else:  # relative path
                if mongodb_observer is not None and overwrite is not None:
                    # use the project root dir provided in the seml config.
                    project_root_dir = mongodb_observer.overwrite['seml']['working_dir']
                    file_observer_base_dir = f"{project_root_dir}/{file_observer_base_dir}"
                    del project_root_dir
        ex.observers.append(seml.create_file_storage_observer(runs_folder_name=runs_folder_name,
                                                              basedir=file_observer_base_dir))
    del mongodb_observer


@ex.automain
def run(dataset: str, hidden_sizes: list, learning_rate: float, max_epochs: int,
        regularization_params: dict):
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:')
    logging.info(f'Dataset: {dataset}, hidden sizes: {hidden_sizes}, learning_rate: {learning_rate}, '
                 f'max_epochs: {max_epochs}, regularization: {regularization_params}')

    #  do your processing here

    # simulate saving a model snapshot
    model = {'params': np.random.uniform(0, 5, size=[64,32])}
    path = "model_checkpoint"
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)
    # this copies the local file to the file observer directory, and deletes the local copy afterwards.
    seml.add_to_file_storage_observer(path, ex, delete_local_file=True)
    # verify that the local file has been cleaned up
    assert not os.path.exists(path)

    results = {
        'test_acc': 0.5 + 0.3 * np.random.randn(),
        'test_loss': np.random.uniform(0, 10),
        # ...
    }
    # the returned result will be written into the database
    return results
