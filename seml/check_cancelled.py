import argparse
import database_utils as db_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check whether the experiment with given ID has been cancelled before its start.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--experiment_id", type=int, help="The experiment ID to check in the database.")
    parser.add_argument("--database_collection", type=str, help="The collection in the database to use.")
    args = parser.parse_args()

    exp_id = args.experiment_id
    db_collection = args.database_collection

    mongodb_config = db_utils.get_mongodb_config()
    collection = db_utils.get_collection(db_collection, mongodb_config)

    exp = collection.find_one({'_id': exp_id})

    if exp is None:
        exit(2)
    if exp['status'] not in  ["QUEUED", "PENDING"]:
        exit(1)
    else:
        exit(0)
