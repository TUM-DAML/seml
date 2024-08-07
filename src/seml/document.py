from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import NotRequired, Required, TypedDict

if TYPE_CHECKING:
    from bson import ObjectId


class SemlDoc(TypedDict, total=False):
    conda_environment: Required[str | None]
    description: str
    executable: Required[str]
    name: str
    output_file: str
    source_files: list[tuple[str, ObjectId]]
    version: tuple[int, int, int]
    working_dir: Required[str]


class SemlConfigDoc(SemlDoc, total=False):
    project_root_dir: str
    use_uploaded_sources: Required[bool]


# For job-name we must use the functional syntax
SBatchOptions = TypedDict(
    'SBatchOptions',
    {
        'array': str,
        'comment': str,
        'cpus_per_task': int,
        'gres': str,
        'job-name': Optional[str],
        'mem': str,
        'nodes': int,
        'ntasks': int,
        'partition': str,
        'output': str,
        'time': str,
    },
    total=False,
    closed=True,
)


class SlurmDoc(TypedDict):
    experiments_per_job: int
    sbatch_options_template: NotRequired[str | None]
    sbatch_options: SBatchOptions
    array_id: int
    num_tasks: NotRequired[int]
    output_files_template: str


class GitDoc(TypedDict):
    path: str
    commit: str
    dirty: bool


class ExecutionDoc(TypedDict):
    cluster: str
    array_id: int
    task_id: int
    slurm_output_file: str


class SacredExperimentDoc(TypedDict):
    base_dir: str
    dependencies: list[str]
    mainfile: str
    name: str
    repositories: list[str]
    sources: list[tuple[str, ObjectId]]


class GPUDoc(TypedDict):
    model: str
    total_emmory: int
    persistence_mode: bool


class GPUsDoc(TypedDict):
    driver_version: str
    gpus: list[GPUDoc]


class HostDoc(TypedDict):
    hostname: str
    os: list[str]
    python_version: str
    cpu: str
    gpus: GPUsDoc
    ENV: dict[str, str]


class MetaDoc(TypedDict):
    command: str
    options: dict[str, Any]
    named_configs: list[str]
    config_updates: dict[str, Any]


class TimeDoc(TypedDict):
    user_time: float
    system_time: float
    max_memory_bytes: int


class GPUStatDoc(TypedDict):
    gpu_max_memory_bytes: int


class StatsDoc(TypedDict):
    real_time: float
    self: TimeDoc
    children: TimeDoc
    pytorch: NotRequired[GPUStatDoc]
    tensorflow: NotRequired[GPUStatDoc]


# One could technically set total=False and properly check if attributes exist. However, this is frequently redundant with the MongoDB query!
# This could be changed in the future. This applies to many other places as well.
class ExperimentDoc(TypedDict, total=True):
    # Set at init
    _id: Required[int]
    add_time: Required[datetime]
    batch_id: Required[int]
    config: Required[dict[str, Any]]
    config_hash: Required[str]
    config_unresolved: Required[dict[str, Any]]
    git: Required[GitDoc | None]
    seml: Required[SemlDoc]
    slurm: Required[list[SlurmDoc]]
    status: Required[str]

    # Set during runtime
    artifacts: list
    captured_out: str
    command: str
    execution: ExecutionDoc
    experiment: SacredExperimentDoc
    fail_trace: list[str]
    format: str
    heartbeat: datetime
    host: HostDoc
    info: dict
    meta: MetaDoc
    resources: list
    result: dict
    start_time: datetime
    stats: StatsDoc
    stop_time: datetime
