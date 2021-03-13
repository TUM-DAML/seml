import argparse
import os

from seml.start import get_command_from_exp
from seml.database import get_collection, find_one_and_update
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
    args = parser.parse_args()

    exp_id = args.experiment_id
    db_collection_name = args.db_collection_name

    collection = get_collection(db_collection_name)

    if args.unobserved:
        find_dict = {'_id': exp_id}
    else:
        find_dict = {'_id': exp_id, "status": {"$in": States.PENDING}}

    # this returns the document as it was BEFORE the update. So we first have to check whether its state was
    # PENDING. This is to avoid race conditions, since find_one_and_update is an atomic operation.
    exp = find_one_and_update(collection, args.unobserved,
                              find_dict, {"$set": {"status": States.RUNNING[0]}})
    if exp is None:
        # check whether experiment is actually missing from the DB or has the wrong state
        exp = collection.find_one({'_id': exp_id})
        if exp is None:
            exit(2)
        else:
            exit(1)

    use_stored_sources = args.stored_sources_dir is not None
    if use_stored_sources and not os.listdir(args.stored_sources_dir):
        assert "source_files" in exp['seml'],\
               "--stored-sources-dir was supplied but staged experiment does not contain stored source files."
        load_sources_from_db(exp, collection, to_directory=args.stored_sources_dir)

    exe, config = get_command_from_exp(exp, db_collection_name, verbose=args.verbose,
                                       unobserved=args.unobserved, post_mortem=args.post_mortem)
    config_args = ' '.join(config)

    cmd = f"python {exe} with {config_args}"
    if use_stored_sources:
        # add command without the temp_dir prefix
        # also add the temp dir for debugging purposes
        if not args.unobserved:
            collection.update_one(
                {'_id': exp_id},
                {'$set': {'seml.command': cmd, 'seml.temp_dir': args.stored_sources_dir}})
        # add the temp_dir prefix to the command
        cmd = f"python {args.stored_sources_dir}/{exe} with {config_args}"
    else:
        if not args.unobserved:
            collection.update_one(
                    {'_id': exp_id},
                    {'$set': {'seml.command': cmd}})

    print(cmd)
    exit(0)
