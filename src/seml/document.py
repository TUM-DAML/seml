from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

from typing_extensions import NotRequired, Required, TypeAlias, TypedDict

if TYPE_CHECKING:
    from datetime import datetime

    from bson import ObjectId


Version: TypeAlias = List[Union[int, str]]


class SemlDocBase(TypedDict, total=False):
    """
    All configurations of the `seml` block that are shared between the configuration file
    and the MongoDB entry

    Attributes
    ----------
    conda_environment : str | None
        The conda environment to use. None will use the current environment.
    description : str, optional
        The description of the experiment.
    executable : str
        The executable python script to run. The path is relative to the working_dir or project_root_dir.
    name: str, optional
        The name of the experiment. Will be used for the slurm job. If None, the collection name will be used.
    output_dir : str
        The output directory of the experiment. The path is relative to the project root directory.
    stash_all_py_files : bool
        Whether to stash all python files or not. Otherwise, only imported files will be stashed.
    """

    conda_environment: Required[str | None]
    executable: Required[str]
    output_dir: str
    description: str
    name: str
    stash_all_py_files: bool
    reschedule_timeout: int | None


class SemlFileConfig(SemlDocBase, total=False):
    """
    The configuration of the `seml` block in the configuration file.

    Attributes
    ----------
    project_root_dir : str
        The root directory of the project.
    """

    project_root_dir: str
    working_dir: str


class SemlDoc(SemlDocBase, total=False):
    """
    output_file: str
        The output file of the experiment where the output is stored.
    source_files: list[tuple[str, ObjectId]]
        The source files of the experiment. The first entry is the relative path to the source file.
        The second entry is the ObjectId of the file in the database.
    version: Version
        The version of seml which was used to create the experiment.
    working_dir: str
        The working directory of the experiment. This is an absolute path that must exist on all machines.
    command: str
        The CLI command that has been executed to run the experiment.
    command_unresolved: str
        The CLI command with unresolved named configurations that runs the experiment.
    temp_dir: str
        The temporary directory which has been used to restore source files from the DB.
    env: dict[str, str]
        The environment variables that were used to run the experiment.
    """

    output_file: str
    source_files: list[tuple[str, ObjectId]]
    version: Required[Version]
    working_dir: Required[str]
    # Runtime populated
    command: str | None
    command_unresolved: str | None
    temp_dir: str | None
    env: dict[str, str] | None


class SemlConfig(SemlDoc, total=False):
    """
    Intermediate configuration of the `seml` block in the configuration file.

    Attributes
    ----------
    version: Version
        The version of seml which was used to create the experiment.
    use_uploaded_sources : bool
        Whether to use the uploaded sources or not.
    working_dir: str
        The working directory of the experiment. This is an absolute path that must exist on all machines.
    """

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
)


class SlurmConfig(TypedDict):
    """
    The valid configuration for the SLURM block in a seml config yaml file.

    Attributes
    ----------
    experiments_per_job : int
        The number of experiments to run in parallel per Slurm job.
    sbatch_options_template : str | None, optional
        The template for the sbatch options. If None, the options will be set directly. Templates must be defined via the settings.py.
    sbatch_options : SBatchOptions
        The sbatch options for the SLURM job.
    """

    experiments_per_job: int
    sbatch_options_template: NotRequired[str | None]
    sbatch_options: SBatchOptions


class SlurmDoc(SlurmConfig):
    """
    The slurm block of a document retrieved from the database.

    Attributes
    ----------
    array_id : int
        The array ID of the SLURM job.
    num_tasks : int
        The number of tasks in the SLURM job.
    output_files_template : str
        The template for the output files. The template must contain the placeholders {array_id} and {task_id}.
    reschedule_file : str
        The path to the reschedule file for this SLURM job. This is set regardless of whether an experiment
        is actually executed by this SLURM job, i.e. whether the job manages to claim the experiment.
    """

    array_id: int
    num_tasks: int
    output_files_template: str
    reschedule_file: str


class GitDoc(TypedDict):
    """
    The git block of a document retrieved from the database.

    Attributes
    ----------
    path: str
        Path at the remote repository, e.g., git@github.com:TUM-DAML/seml.git
    commit: str
        The commit hash at the time of submission.
    dirty: bool
        Whether the repository was dirty at the time of submission.
    """

    path: str
    commit: str
    dirty: bool


class ExecutionDoc(TypedDict):
    """
    The execution block of a document retrieved from the database.

    Attributes
    ----------
    cluster: str
        The name of the Slurm cluster on which the experiment has been scheduled on.
    array_id: int
        The array ID of the SLURM job.
    task_id: int
        The task ID of the SLURM job.
    slurm_output_file: str
        The output file of the SLURM job.
    reschedule_file: str
        The reschedule file of the SLURM job. This is only set once an experiment is actually
        executed by the SLURM job.
    """

    cluster: str
    array_id: int
    task_id: int
    slurm_output_file: str
    reschedule_file: str


class SacredExperimentDoc(TypedDict):
    """
    The sacred block of a document retrieved from the database. Populated by sacred.
    TODO: Honestly no idea what this is good for, it looks like it's not correctly populated. 🤷‍♂️
    """

    base_dir: str
    dependencies: list[str]
    mainfile: str
    name: str
    repositories: list[str]
    sources: list[tuple[str, ObjectId]]


class GPUDoc(TypedDict):
    """
    The gpu block in the gpus block of the host document. Populated by sacred.

    Attributes
    ----------
    model: str
        The model of the GPU.
    total_emmory: int
        The total memory of the GPU in MB.
    persistence_mode: bool
        Whether the persistence mode is enabled.
    """

    model: str
    total_emmory: int
    persistence_mode: bool


class GPUsDoc(TypedDict):
    """
    The gpus block of a document retrieved from the database. Populated by sacred.

    Attributes
    ----------
    driver_version: str
        The version of the GPU driver.
    gpus: list[GPUDoc]
        The list of GPUs used for the experiment.
    """

    driver_version: str
    gpus: list[GPUDoc]


class HostDoc(TypedDict):
    """
    The host block of a document retrieved from the database. Populated by sacred.

    Attributes
    ----------
    hostname: str
        The hostname of the machine.
    os: list[str]
        The operating system of the machine.
        Typically a list, e.g., ['Linux', 'Linux-5.15.0-107-generic-x86_64-with-glibc2.35'] is returned for Ubuntu 22.04.
    python_version: str
        The version of the Python interpreter.
    cpu: str
        The model of the CPU.
    gpus: GPUsDoc
        The GPUs used for this experiment.
    ENV: dict[str, str]
        The environment variables that are overwritten during the experiment.
        TODO: This doesn't look properly populated
    """

    hostname: str
    os: list[str]
    python_version: str
    cpu: str
    gpus: GPUsDoc
    ENV: dict[str, str]


class MetaDoc(TypedDict):
    """
    The meta block of a document retrieved from the database. Populated by sacred.

    Attributes
    ----------
    command: str
        The command that has been executed.
        TODO: This doesn't look properly populated.
    options: dict[str, Any]
        TODO: Figure out what this one does.
    named_configs: list[str]
        The named configurations that have been used.
        TODO: This will always be empty since we populate them at add time.
    config_updates: dict[str, Any]
        The configuration updates that have been used.
    """

    command: str
    options: dict[str, Any]
    named_configs: list[str]
    config_updates: dict[str, Any]


class StatDoc(TypedDict):
    """
    Statics of the time and memory usage of a process. Used in StatsDoc.

    Attributes
    ----------
    user_time: float
        The user time of the process.
    system_time: float
        The system time of the process.
    max_memory_bytes: int
        The maximum memory usage of the process in bytes.
    """

    user_time: float
    system_time: float
    max_memory_bytes: int


class GPUStatDoc(TypedDict):
    """
    Attributes
    ----------
    gpu_max_memory_bytes: int
        The maximum number of bytes allocated at the GPU.
    """

    gpu_max_memory_bytes: int


class StatsDoc(TypedDict):
    real_time: float
    self: StatDoc
    children: StatDoc
    pytorch: NotRequired[GPUStatDoc]
    tensorflow: NotRequired[GPUStatDoc]


# One could technically set total=False and properly check if attributes exist. However, this is frequently redundant with the MongoDB query!
# This could be changed in the future. This applies to many other places as well.
class ExperimentDoc(TypedDict, total=True):
    """
    The document retrieved from the database.

    Attributes
    ----------
    These attributes should always exist:
    _id: int
        The ID of the document. Set by sacred.
    add_time: datetime
        The time at which this experiment has been added to the collection.
    batch_id: int
        ID of the batch which this experiment belongs to.
    config: dict[str, Any]
        The configuration of the experiment after resolving named configurations.
    config_hash: str
        A hash of the config to identify duplicates.
    config_unresolved: dict[str, Any]
        The configuration of the experiment before resolving named configurations.
    git: GitDoc | None
        Information about the git status at the time of staging.
    seml: SemlDoc
        Seml specific information about source codes, working directories and commands.
    slurm: list[SlurmDoc]
        A list of Slurm configurations for the experiment. For each configuration a
        separate Slurm job will be started and the job will be assigned on the first come
        first serve basis.
    statis: str
        The status of the experiment.


    The following are set during runtime:
    artifacts: list
        A list of artifacts that are created during runtime and are stored in the mongodb.
    captured_out: str
        The captured output of the experiment. This is only popultaed if SETTINGS.EXPERIMENT.CAPUTRE_OUPUT=True.
    command: str
        The command that has been executed. Populated by sacred.
        TODO: This is likely incorrectly populated.
    execution: ExecutionDoc
        Information about the Slurm cluster and job that executed this experiment.
    experiment: SacredExperimentDoc
        Sacred's experiment information. Populated by sacred.
    fail_trace: list[str]
        A list of lines of the stack trace if the experiment failed.
    format: str
        The Observer format.
    heartbeat: datetime
        The last heartbeat of the experiment.
    host: HostDoc
        Information about the host that executed the experiment. Populated by sacred.
    info: dict
        Additional optional information that is populated during runtime.
    meta: MetaDoc
        Meta information about the experiment. Populated by sacred.
    resources: list
        TODO: Unknown - populated by sacred.
    result: Any
        The result of the experiment.
    start_time: datetime
        The time at which the experiment has been started.
    stats: StatsDoc
        The statistics about runtime and memory consumption of the experiment.
    stop_time: datetime
        The time at which the experiment has been stopped.
    reschedule_config_update: dict[str, Any] | None
        If the experiment has been rescheduled, this contains the configuration update
        that must be applied during rescheduling.
    """

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
    result: Any
    start_time: datetime
    stats: StatsDoc
    stop_time: datetime
    reschedule_config_update: dict[str, Any] | None


class ExperimentConfig(TypedDict, total=False, closed=False):
    # We have this base class to represent also sub-configurations.
    # TODO: properly type the configurations files.
    fixed: Any
    grid: Any
    random: Any
    __extra_items__: dict[str, ExperimentConfig]


class SemlExperimentFile(ExperimentConfig, total=False, closed=True):
    seml: SemlFileConfig
    slurm: list[SlurmConfig]
