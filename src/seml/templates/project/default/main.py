from seml.experiment import Experiment


experiment = Experiment()


@experiment.config
def default_config():
    # Define your default configuration here
    dataset = 'small'
    model = dict(n_layers=2, hidden_dim=32)


@experiment.automain
def main(
    # Define your configuration parameters here
    dataset: str,
    model: dict,
    seed: int,  # seml automatically assigns a random seed
):
    # Define your main function here
    pass
    # The result will be stored in the MongoDB
    return {
        'dataset': dataset,
        'model': model,
        'seed': seed,
    }
