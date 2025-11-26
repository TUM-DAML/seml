import math
import uuid

from pymongo import MongoClient


class ASHA:
    def __init__(
        self,
        asha_collection_name: str,
        eta: int | float,
        min_r: int,
        max_r: int,
        metric_increases: bool,
        mongodb_configurations,
        _log,
    ):
        """Doc string pretty please ^^

        Args:
            asha_collection_name (str): _description_
            eta (float): _description_
            min_r (int): _description_
            max_r (int): _description_
            metric_increases (bool): _description_
            mongodb_configurations (_type_): _description_
        """
        # ! TODO: Adding argument verification
        self.asha_collection_name = asha_collection_name
        self._log = _log
        self.job_uuid = str(uuid.uuid4())
        # ! TODO: No printing, use _log instead, also please make sure to properly distinguish between info/warning/debug, etc.
        self._log.info(
            f'--------------------------JobUUID:{self.job_uuid}---------------------'
        )
        self.metric_history = []
        self.others_metric_at_stage = {}
        self.eta = eta
        self.min_r = min_r
        self.max_r = max_r
        self.metric_increases = metric_increases
        self.rungs = self.generate_rungs(self.min_r, self.eta, self.max_r)
        self.mongodb_configurations = mongodb_configurations
        self.samples = 5  # <- Not sure what this does? But it appears like it was hardcoded to 5 in the original main.py, comment: this was to make the isbest function that doesn't work yet
        self.collection = self._get_mongo_collection(
            self.mongodb_configurations, self.asha_collection_name
        )

    def _get_mongo_collection(self, mongodb_configurations, experiment_name):
        """
        Connecting to the MongoDB, credentials from SEML config
        returns connection
        """
        # ? TODO: Adding some retry logic if we exeperience transient connection issues?
        auth_source = mongodb_configurations.get(
            'authSource', mongodb_configurations['db_name']
        )
        if mongodb_configurations.get('username') and mongodb_configurations.get(
            'password'
        ):
            uri = f'mongodb://{mongodb_configurations["username"]}:{mongodb_configurations["password"]}@{mongodb_configurations["host"]}:{mongodb_configurations["port"]}/?authSource={auth_source}'
        else:
            uri = f'mongodb://{mongodb_configurations["host"]}:{mongodb_configurations["port"]}'

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)  # 5 sec timeout
        client.admin.command('ping')  # check connection

        db = client[mongodb_configurations['db_name']]
        collection = db[experiment_name]
        self._log.debug(
            f"Connected to MongoDB and accessed collection '{experiment_name}' successfully."
        )
        return collection

    def save_metric_to_db(self, collection, job_id, stage, metric):
        """
        Insert or update metric for the given job_id and stage in the MongoDB collection.
        """
        collection.update_one(
            {'job_id': job_id, 'stage': stage},
            {'$set': {'metric': metric}},
            upsert=True,
        )

    def store_stage_metric(self, stage: int, metric: float):
        """
        Accuracy added and other metrics compaired,
        probably should break this into different functions,
        as of now this is our running function
        """
        other_job_metrics = {}

        # ? TODO: Should we really submitt and pull from the database on every stage, even if it isn't a decision rung yet?
        self._log.debug('Trying MongoDB access...')
        # ? TODO: Keep the connection open instead of recreating it each stage?
        self.collection = self._get_mongo_collection(
            self.mongodb_configurations, self.asha_collection_name
        )
        self.save_metric_to_db(self.collection, self.job_uuid, stage, metric)
        self._log.debug('storage in mongodb')
        other_job_metrics = self.get_metric_at_stage_db(
            self.collection, stage, self.job_uuid
        )

        self.metric_history.append(metric)
        self.others_metric_at_stage[str(stage)] = other_job_metrics

        promote = True
        should_terminate = False

        # ! TODO: Asha should not be required to know about the number of stages
        # if stage == self.num_stages - 1:
        #     self.set_status_db("Completed")
        #     # self.isbest()
        # elif stage in self.rungs:
        if stage in self.rungs:
            self._log.info(f'checking stage {stage}')
            self._print_stage_info(stage, metric, other_job_metrics)
            promote = self._job_promotion(metric, other_job_metrics, self.eta)
            if promote:
                self._log.info(f'this job was promoted at {stage}')
                pass
            else:
                self._log.info(f'this job should be terminated at {stage}')
                should_terminate = True
                self.set_status_db('Completed')
        return should_terminate

    def metric_in_rungs(self, stage):
        """
        if user wants to check if their stage/resource is in a rung
        """
        if stage in self.rungs:
            return True
        else:
            return False

    def get_metric_at_stage_db(self, collection, stage, current_job_id=None):
        """
        Retrieve metrics of all jobs at the specified stage from the MongoDB collection.
        Returns a dict: {job_id: metric}, excluding current_job_id if provided.
        """
        results = collection.find({'stage': stage})
        metrics = {}
        for doc in results:
            job_id = doc.get('job_id')
            metric = doc.get('metric', -1.0)
            if job_id and job_id != current_job_id:
                metrics[job_id] = metric
        return metrics

    def _print_stage_info(self, stage, metric, other_job_metrics):
        self._log.info(f'[Epoch {stage}] Own metric: {metric}')
        self._log.info(f"[Epoch {stage}] Other jobs' metrics: {other_job_metrics}")
        pass

    def _job_promotion(self, metric, other_job_metrics, eta):
        """
        returns cutoff metric at which jobs should be promoted
        """
        self._log.info('Checking if this job progresses')

        valid_metrics = [acc for acc in other_job_metrics.values() if acc > -1] + [
            metric
        ]
        sorted_vals = sorted(valid_metrics, reverse=True)
        cutoff_metric = 0.0
        promotion = 'True'

        if self.metric_increases:
            # ? TODO: Should this be ceil, round or floor
            top_k = max(1, math.floor(len(sorted_vals) // eta))
            cutoff_metric = sorted_vals[top_k - 1]
            promotion = metric >= cutoff_metric or math.isclose(
                metric, cutoff_metric, rel_tol=1e-9
            )
        else:
            sorted_vals = sorted(valid_metrics)  # ascending: lowest first
            # ? TODO: Should this be ceil, round or floor
            bottom_k = max(1, math.floor(len(sorted_vals) // eta))
            cutoff_metric = sorted_vals[bottom_k - 1]  # kth lowest value
            promotion = metric <= cutoff_metric or math.isclose(
                metric, cutoff_metric, rel_tol=1e-9
            )

        self._log.info(f'Valid metrics (sorted): {sorted_vals}')
        self._log.info(f'Cutoff metric for promotion: {cutoff_metric:.8f}')
        self._log.info(f'Current job metric: {metric:.8f}')
        self._log.info(f'Promotion decision: {promotion}')
        self._log.info('--------------------------------------------------')

        return promotion

    def generate_rungs(self, min_r, eta, max_r):
        """
        generates rungs at which promotion will be checked
        """
        rungs = []
        resource = min_r
        if min_r > max_r:
            raise ValueError('min_r must be <= max_r')

        while resource <= max_r:
            # Rounding allows for eta to be a floating point value
            resource = int(round(resource))
            rungs.append(resource)
            resource *= eta

        self._log.info(f'Generated rungs of the following shape: {rungs}')
        per_sample_avg_stages = sum(
            [stages / (eta**i) for (i, stages) in enumerate(rungs)]
            + [max_r / (eta ** len(rungs))]
        )
        self._log.info(
            f'Given this ASHA configuration, the expected average number of stages per sample is: {per_sample_avg_stages:.2f}'
        )
        return rungs

    def set_status_db(self, status):
        """
        set status in mongodb collection to mark if process is still running
        """
        self.collection.update_one(
            {'job_id': self.job_uuid}, {'$set': {'Status': status}}, upsert=True
        )

    # def isbest(self,metric,other_job_metrics):
    def isbest(self):
        """
        this function doesn't is incorrect,
        working on it to use the status to see if all jobs are completed
        """

        completed_jobs = list(self.collection.find({'Status': 'Completed'}))
        if len(completed_jobs) != self.samples:
            return

        best_job = max(completed_jobs, key=lambda doc: doc.get('metric', -1.0))
        best_job_id = best_job.get('job_id')

        # Mark all completed jobs as not best
        self.collection.update_many({'Status': 'Completed'}, {'$set': {'BEST': 'NO'}})

        # Mark the best job
        self.collection.update_one(
            {'job_id': best_job_id}, {'$set': {'BEST': 'YES'}}, upsert=True
        )

        # if best_job_id == self.job_uuid:
        #     self._run.log_scalar("Finished Last but was the best")

        # valid_metrics = [acc for acc in other_job_metrics.values() if acc > -1] + [metric]
        # if len(other_job_metrics)+1 == len(valid_metrics):
        #     if metric == max(valid_metrics):
        #         return True
        # else:
        #     return False
