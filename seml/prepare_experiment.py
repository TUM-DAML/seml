import argparse
import os

from seml.start import get_command_from_exp
from seml.database import get_collection
from seml.sources import load_sources_from_db

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

    exp = collection.find_one({'_id': exp_id})
    use_stored_sources = args.stored_sources_dir is not None
    if use_stored_sources and not os.listdir(args.stored_sources_dir):
        assert "source_files" in exp['seml'],\
               "--stored-sources-dir was supplied but queued experiment does not contain stored source files."
        load_sources_from_db(exp, collection, to_directory=args.stored_sources_dir)

    exe, config = get_command_from_exp(exp, db_collection_name, verbose=args.verbose,
                                       unobserved=args.unobserved, post_mortem=args.post_mortem)

    cmd = f"python {exe} with {' '.join(config)}"
    if use_stored_sources:
        # add command without the temp_dir prefix
        # also add the temp dir for debugging purposes
        collection.update_one(
            {'_id': exp_id},
            {'$set': {'seml.command': cmd, 'seml.temp_dir': args.stored_sources_dir}})
        # add the temp_dir prefix to the command
        cmd = f"python {args.stored_sources_dir}/{exe} with {' '.join(config)}"
    else:
        collection.update_one(
                {'_id': exp_id},
                {'$set': {'seml.command': cmd}})

    if exp is None:
        exit(2)
    if exp['status'] != "PENDING":
        exit(1)
    else:
        print(cmd)
        exit(0)
