import argparse
import os
import logging

from seml.start import get_command_from_exp, get_shell_command
from seml.database import get_collection
from seml.sources import load_sources_from_db
from seml.settings import SETTINGS

States = SETTINGS.STATES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the config and executable of the experiment with given ID and "
                    "check whether it has been cancelled before its start.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--experiment_id", type=int, help="The experiment ID.")
    parser.add_argument("--db_collection_name", type=str, help="The collection in the database to use.")
    parser.add_argument("--verbose", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Display more log messages.")
    parser.add_argument("--unobserved", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Run the experiments without Sacred observers.")
    parser.add_argument("--post-mortem", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Activate post-mortem debugging with pdb.")
    parser.add_argument("--stored-sources-dir", default=None, type=str,
                        help="Load source files into this directory before starting.")
    parser.add_argument('--debug-server', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Run the experiment with a debug server.")
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    exp_id = args.experiment_id
    db_collection_name = args.db_collection_name

    collection = get_collection(db_collection_name)

    # This returns the document as it was BEFORE the update. So we first have to check whether its state was
    # PENDING. This is to avoid race conditions, since find_one_and_update is an atomic operation.
    if args.unobserved:
        exp = collection.find_one({'_id': exp_id})
    else:
        exp = collection.find_one_and_update({'_id': exp_id, "status": {"$in": States.PENDING}},
                                             {"$set": {"status": States.RUNNING[0]}})

    if exp is None:
        # check whether experiment is actually missing from the DB or has the wrong state
        if collection.count_documents({'_id': exp_id}) == 0:
            exit(2)
        else:
            exit(1)

    use_stored_sources = args.stored_sources_dir is not None
    if use_stored_sources and not os.listdir(args.stored_sources_dir):
        assert "source_files" in exp['seml'],\
               "--stored-sources-dir was supplied but staged experiment does not contain stored source files."
        load_sources_from_db(exp, collection, to_directory=args.stored_sources_dir)

    interpreter, exe, config = get_command_from_exp(exp, db_collection_name, verbose=args.verbose,
                                                    unobserved=args.unobserved, post_mortem=args.post_mortem,
                                                    debug_server=args.debug_server)
    cmd = get_shell_command(interpreter, exe, config)
    updates = {'seml.command': cmd}

    if use_stored_sources:
        temp_dir = args.stored_sources_dir
        # Store the temp dir for debugging purposes
        updates["seml.temp_dir"] = temp_dir
        cmd = get_shell_command(interpreter, os.path.join(temp_dir, exe), config)

    if not args.unobserved:
        collection.update_one({'_id': exp_id}, {'$set': updates})

    print(cmd)
    exit(0)
