import argparse
from seml import database_utils as db_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save Slurm job status information (retrieved via sstat) to MongoDB.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--experiment_id", type=int, help="The experiment ID.")
    parser.add_argument("--database_collection", type=str, help="The collection in the database to use.")
    parser.add_argument("--job-status", type=str, help="Job status information from sstat.")
    args = parser.parse_args()

    exp_id = args.experiment_id
    db_collection = args.database_collection
    job_status = args.job_status.split('|')

    collection = db_utils.get_collection(db_collection)
    collection.update_one(
            {'_id': exp_id},
            {'$set': {'job_status.MaxRSS': job_status[0],
                      'job_status.MaxVMSize': job_status[1],
                      'job_status.AveCPU': job_status[2],
                      'job_status.AvePages': job_status[3],
                      'job_status.AveDiskRead': job_status[4],
                      'job_status.AveDiskWrite': job_status[5],
                      'job_status.ConsumedEnergy': job_status[6]}})
