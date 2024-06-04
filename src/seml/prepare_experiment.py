import argparse
import os
from sacred.randomness import get_seed

from seml.database import get_collection
from seml.experiment import (
    is_local_main_process,
    is_main_process,
    is_running_in_multi_process,
)
from seml.settings import SETTINGS
from seml.sources import load_sources_from_db
from seml.start import get_command_from_exp, get_shell_command

States = SETTINGS.STATES

if __name__ == '__main__':
    # This process should only be executed once per node, so if we are not the main
    # process per node, we directly exit.
    if not is_local_main_process():
        exit(0)

    parser = argparse.ArgumentParser(
        description='Get the config and executable of the experiment with given ID and '
        'check whether it has been cancelled before its start.',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--experiment_id', type=int, help='The experiment ID.')
    parser.add_argument(
        '--db_collection_name', type=str, help='The collection in the database to use.'
    )
    parser.add_argument(
        '--verbose',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help='Display more log messages.',
    )
    parser.add_argument(
        '--unobserved',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help='Run the experiments without Sacred observers.',
    )
    parser.add_argument(
        '--post-mortem',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help='Activate post-mortem debugging with pdb.',
    )
    parser.add_argument(
        '--stored-sources-dir',
        default=None,
        type=str,
        help='Load source files into this directory before starting.',
    )
    parser.add_argument(
        '--debug-server',
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help='Run the experiment with a debug server.',
    )
    args = parser.parse_args()

    exp_id = args.experiment_id
    db_collection_name = args.db_collection_name
    collection = get_collection(db_collection_name)

    # This returns the document as it was BEFORE the update. So we first have to check whether its state was
    # PENDING. This is to avoid race conditions, since find_one_and_update is an atomic operation.
    if args.unobserved:
        exp = collection.find_one({'_id': exp_id})
    else:
        slurm_array_id = os.environ.get('SLURM_ARRAY_JOB_ID', None)
        slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
        if slurm_array_id is not None and slurm_task_id is not None:
            # We're running in SLURM.
            # Check if the job executing is this one.
            job_filter = {
                'slurm.array_id': int(slurm_array_id),
                'slurm.task_id': int(slurm_task_id),
            }
            # Either take the experiment if it is pending or if it is the one being executed.
            # The latter case is important for multi-node jobs.
            exp = collection.find_one_and_update(
                {
                    '$and': [
                        {'_id': exp_id},
                        {
                            '$or': [
                                {'status': {'$in': States.PENDING}},
                                job_filter,
                            ]
                        },
                    ]
                },
                {'$set': {'status': States.RUNNING[0]}},
            )
        else:
            exp = collection.find_one_and_update(
                {'_id': exp_id, 'status': {'$in': States.PENDING}},
                {
                    '$set': {'status': States.RUNNING[0]},
                    '$unset': {'slurm.array_id': '', 'slurm.task_id': ''},
                },
            )

    if exp is None:
        # check whether experiment is actually missing from the DB or has the wrong state
        if collection.count_documents({'_id': exp_id}) == 0:
            exit(4)
        else:
            exit(3)

    use_stored_sources = args.stored_sources_dir is not None
    if use_stored_sources:
        os.makedirs(args.stored_sources_dir, exist_ok=True)
        if not os.listdir(args.stored_sources_dir):
            assert (
                'source_files' in exp['seml']
            ), '--stored-sources-dir was supplied but staged experiment does not contain stored source files.'
            load_sources_from_db(exp, collection, to_directory=args.stored_sources_dir)

    # The remaining part (updateing MongoDB & printing the python command) is only executed by the main process.
    if not is_main_process():
        exit(0)

    # If we run in a multi task environment, we want to make sure that the seed is fixed once and
    # all tasks start with the same seed. Otherwise, one could not reproduce the experiment as the
    # seed would change on the child nodes. It is up to the user to distribute seeds if needed.
    if is_running_in_multi_process():
        if SETTINGS.CONFIG_KEY_SEED not in exp['config']:
            exp['config'][SETTINGS.CONFIG_KEY_SEED] = get_seed()

    interpreter, exe, config = get_command_from_exp(
        exp,
        db_collection_name,
        verbose=args.verbose,
        unobserved=args.unobserved,
        post_mortem=args.post_mortem,
        debug_server=args.debug_server,
    )
    cmd = get_shell_command(interpreter, exe, config)
    cmd_unresolved = get_shell_command(
        *get_command_from_exp(
            exp,
            db_collection_name,
            verbose=args.verbose,
            unobserved=args.unobserved,
            post_mortem=args.post_mortem,
            debug_server=args.debug_server,
            unresolved=True,
        )
    )
    updates = {
        'seml.command': cmd,
        'seml.command_unresolved': cmd_unresolved,
    }

    if use_stored_sources:
        temp_dir = args.stored_sources_dir
        # Store the temp dir for debugging purposes
        updates['seml.temp_dir'] = temp_dir
        cmd = get_shell_command(interpreter, os.path.join(temp_dir, exe), config)

    if not args.unobserved:
        collection.update_one({'_id': exp_id}, {'$set': updates})

    print(cmd)
    exit(0)
