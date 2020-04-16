import argparse
from seml import database_utils as db_utils
from seml.misc import get_config_from_exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the config and executable of the experiment with given ID and "
                    "check whether it has been cancelled before its start.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--experiment_id", type=int, help="The experiment ID.")
    parser.add_argument("--database_collection", type=str, help="The collection in the database to use.")
    parser.add_argument("--log-verbose", default=False, type=lambda x: (str(x).lower() == 'true'), help="Display more log messages.")
    parser.add_argument("--unobserved", default=False, type=lambda x: (str(x).lower() == 'true'), help="Run the experiments without Sacred observers.")
    parser.add_argument("--post-mortem", default=False, type=lambda x: (str(x).lower() == 'true'), help="Activate post-mortem debugging with pdb.")
    parser.add_argument("--stored-sources-dir", default=None, type=str, help="Load source files into this directory before starting.")
    args = parser.parse_args()

    exp_id = args.experiment_id
    db_collection = args.database_collection

    collection = db_utils.get_collection(db_collection)

    exp = collection.find_one({'_id': exp_id})
    use_stored_sources = args.stored_sources_dir is not None
    if use_stored_sources:
        assert "source_files" in exp, \
            "--stored-sources-dir was supplied but queued experiment does not contain stored source files."
        db_utils.load_sources_from_db(exp, to_directory=args.stored_sources_dir)

    exe, config = get_config_from_exp(exp, log_verbose=args.log_verbose,
                                      unobserved=args.unobserved, post_mortem=args.post_mortem,
                                      relative=use_stored_sources)

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
