import logging
from time import sleep

from seml import Experiment

ex = Experiment()


@ex.reschedule_hook
def reschedule(step: int):
    logging.info(f'Reschedule triggered at step {step}.')
    return {'checkpoint': step}


@ex.automain
def run(
    n_steps: int,
    checkpoint: int | None = None,
):
    logging.info('Starting experiment with the following parameters:')
    logging.info(f'n_steps: {n_steps}, checkpoint: {checkpoint}')

    if checkpoint is not None:
        logging.info(f'Resuming from checkpoint: {checkpoint}')
        # Load your model/state from the checkpoint here

    # Simulate some processing
    for step in range(checkpoint or 0, n_steps):
        reschedule(step)
        logging.info(f'Processing step {step + 1}/{n_steps}')
        sleep(1.0)

    logging.info('Experiment completed successfully.')
    return
