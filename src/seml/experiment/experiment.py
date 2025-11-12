import datetime
import logging
import resource
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union, cast

from pymongo.collection import Collection

from seml.database import States, get_collection
from seml.document import ExperimentDoc
from seml.experiment.observers import create_mongodb_observer
from seml.settings import SETTINGS
from seml.utils.multi_process import is_main_process

# These are only used for type hints
if TYPE_CHECKING:
    from sacred import Experiment as ExperimentBase
    from sacred import Ingredient
    from sacred.commandline_options import CLIOption
    from sacred.host_info import HostInfoGetter
    from sacred.utils import PathType
from sacred import SETTINGS as SACRED_SETTINGS
from sacred import Experiment as ExperimentBase
from sacred.config.utils import (
    dogmatize,
    recursive_fill_in,
    undogmatize,
)
from sacred.utils import SacredInterrupt


class LoggerOptions(Enum):
    NONE = None
    DEFAULT = 'default'
    RICH = 'rich'


class RescheduleInterrupt(SacredInterrupt):
    STATUS = States.PENDING[0]


class Experiment(ExperimentBase):
    _reschedule_watch_path: Path | None | Literal['INVALID']

    def __init__(
        self,
        name: Optional[str] = None,
        ingredients: Sequence['Ingredient'] = (),
        interactive: bool = False,
        base_dir: Optional['PathType'] = None,
        additional_host_info: Optional[List['HostInfoGetter']] = None,
        additional_cli_options: Optional[Sequence['CLIOption']] = None,
        save_git_info: bool = True,
        add_mongodb_observer: bool = True,
        logger: Optional[Union[LoggerOptions, str]] = LoggerOptions.RICH,
        capture_output: Optional[bool] = None,
        collect_stats: bool = True,
    ):
        super().__init__(
            name=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            additional_host_info=additional_host_info,
            additional_cli_options=additional_cli_options,
            save_git_info=save_git_info,
        )
        self.capture_output = capture_output
        if add_mongodb_observer:
            self.configurations.append(MongoDbObserverConfig(self))
        self.configurations.append(ClearObserverForMultiTaskConfig(self))
        if logger:
            setup_logger(self, LoggerOptions(logger))
        if collect_stats:
            self.post_run_hook(lambda _run: _collect_exp_stats(_run))

        self._reschedule_watch_path = None

    def run(
        self,
        command_name: Optional[str] = None,
        config_updates: Optional[dict] = None,
        named_configs: Sequence[str] = (),
        info: Optional[dict] = None,
        meta_info: Optional[dict] = None,
        options: Optional[dict] = None,
    ):
        if (
            not SETTINGS.EXPERIMENT.CAPTURE_OUTPUT and not self.capture_output
        ) or self.capture_output is False:
            SACRED_SETTINGS.CAPTURE_MODE = 'no'  # type: ignore
        return super().run(
            command_name=command_name,
            config_updates=config_updates,
            named_configs=named_configs,
            info=info,
            meta_info=meta_info,
            options=options,
        )

    def reschedule_hook(self, f):
        """Decorator to register the reschedule hook.

        In case of a required rescheduling, the decorated function will be called.
        It should return a config dictionary that will be used to update the current
        configuration before submitting the rescheduled job.

        Parameters
        ----------
        f: Callable[[..., dict]]
            User-defined function to be called when rescheduling is triggered.

        Returns
        -------
        Callable
            Wrapped function that checks for rescheduling and calls the user-defined function.


        Example
        -------

        ```python
        from seml import Experiment
        ex = Experiment()

        @ex.reschedule_hook
        def reschedule(step: int):
            print(f'Reschedule triggered at step {step}.')
            return {'checkpoint': step}

        @ex.automain
        def run(n_steps: int, checkpoint: int | None = None):
            if checkpoint is not None:
                print(f'Resuming from checkpoint: {checkpoint}')
            for step in range(checkpoint or 0, n_steps):
                reschedule(step)
                print(f'Processing step {step + 1}/{n_steps}')
            print('Experiment completed successfully.')
        ```
        """

        def _reschedule_hook(*args, **kwargs):
            # Check whether we must reschedule
            if self._reschedule_watch_path is None:
                self._reschedule_watch_path = (
                    self._get_reschedule_watch_path() or 'INVALID'
                )
            if self._reschedule_watch_path == 'INVALID' or not is_main_process():
                return

            _must_reschedule = Path(self._reschedule_watch_path).exists()
            if not _must_reschedule:
                return

            # If yes, call the user-defined function
            logging.info('Caught reschedule signal, calling reschedule_hook.')
            new_config = f(*args, **kwargs)
            assert isinstance(new_config, dict), (
                'Reschedule hook must return a configuration dictionary.'
                f' Got type:  + {type(new_config)}'
            )
            logging.info('Reschedule hook returned new configuration: ' f'{new_config}')

            # Merge config update with experiment doc
            exp = self._get_exp_document_from_db()
            assert exp is not None, (
                'Could not retrieve experiment document from database.'
                ' Rescheduling not possible.'
            )
            collection = self._get_db_collection()
            assert collection is not None
            # TODO: TIGRU: Update config of experiment document in the database

            raise RescheduleInterrupt

        return _reschedule_hook

    def _get_db_collection(self) -> Optional[Collection[ExperimentDoc]]:
        assert self.current_run is not None
        db_collection = self.current_run.config.get('db_collection')
        if db_collection is None:
            logging.warning(
                'Reschedule hook called outside of a SEML-managed experiment.'
                ' Rescheduling hook is ignored.'
            )
            return None
        db_collection = get_collection(db_collection)
        return db_collection

    def _get_exp_id(self) -> Optional[int]:
        assert self.current_run is not None
        exp_id = self.current_run.config.get('overwrite')
        if exp_id is None:
            logging.warning(
                'Reschedule hook called without a SEML experiment ID.'
                ' Is this execution unobserved? Rescheduling hook is ignored.'
            )
        return exp_id

    def _get_exp_document_from_db(self) -> Optional[ExperimentDoc]:
        db_collection = self._get_db_collection()
        exp_id = self._get_exp_id()
        if db_collection is None or exp_id is None:
            return None
        exp = db_collection.find_one(exp_id)
        if exp is None:
            logging.warning(
                f'No experiment with ID {exp_id} found in database.'
                ' Cannot retrieve experiment document.'
            )
        return exp

    def _get_reschedule_watch_path(self) -> Optional[Path]:
        exp = self._get_exp_document_from_db()
        if exp is None:
            return

        if exp.get('execution', {}).get('cluster', None) == 'local':
            logging.info('Experiment is executed locally. Reschedule hook is ignored.')
            return

        reschedule_path = exp.get('execution', {}).get('reschedule_file', None)
        assert (
            reschedule_path is not None
        ), 'No SLURM reschedule file recorded for this experiment in the database.'
        reschedule_path = Path(reschedule_path)
        logging.info(
            'Found signal path for reschedule hook.'
            f' Watching for file: {reschedule_path.as_posix()}'
        )
        return reschedule_path


class MongoDbObserverConfig:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def __call__(self, fixed=None, preset=None, fallback=None):
        from sacred.config.config_summary import ConfigSummary
        from sacred.config.custom_containers import DogmaticDict

        result = cast(DogmaticDict, dogmatize(fixed or {}))
        defaults = dict(overwrite=None, db_collection=None)
        recursive_fill_in(result, defaults)
        recursive_fill_in(result, preset or {})
        added = result.revelation()
        config_summary = ConfigSummary(added, result.modified, result.typechanges)
        config_summary.update(undogmatize(result))

        if config_summary['db_collection'] is not None and is_main_process():
            self.experiment.observers.append(
                create_mongodb_observer(
                    config_summary['db_collection'],
                    overwrite=config_summary['overwrite'],
                )
            )
        return config_summary


class ClearObserverForMultiTaskConfig:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def __call__(self, fixed=None, preset=None, fallback=None):
        from sacred.config.config_summary import ConfigSummary
        from sacred.config.custom_containers import DogmaticDict

        result = cast(DogmaticDict, dogmatize(fixed or {}))
        defaults = dict(overwrite=None, db_collection=None)
        recursive_fill_in(result, defaults)
        recursive_fill_in(result, preset or {})
        added = result.revelation()
        config_summary = ConfigSummary(added, result.modified, result.typechanges)
        config_summary.update(undogmatize(result))

        # We only want observers on the main process
        if not is_main_process():
            self.experiment.observers.clear()
        return config_summary


def setup_logger(
    ex: 'ExperimentBase',
    logger_option: LoggerOptions = LoggerOptions.RICH,
    level: Optional[Union[str, int]] = None,
):
    """
    Set up logger for experiment.

    Parameters
    ----------
    ex: sacred.Experiment
    Sacred experiment to set the logger of.
    level: str or int
    Set the threshold for the logger to this level. Default is logging.INFO.

    Returns
    -------
    None

    """
    if hasattr(ex, 'logger') and ex.logger:
        logging.warn(
            'Logger already set up for this experiment.\n'
            'The new seml.experiment.Experiment class already includes the logger setup.\n'
            'Either remove the explicit call to setup_logger or disable the logger setup in the Experiment constructor.'
        )
        return
    if logger_option is LoggerOptions.NONE:
        return
    logger = logging.getLogger()
    logger.handlers = []
    if level is None:
        if is_main_process():
            level = logging.INFO
        else:
            level = logging.ERROR
    if logger_option is LoggerOptions.RICH:
        from rich.logging import RichHandler

        from seml.console import console

        logger.addHandler(
            RichHandler(
                level,
                console=console,
                show_time=True,
                show_level=True,
                log_time_format='[%X]',
            )
        )
    elif logger_option is LoggerOptions.DEFAULT:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(level)
    ex.logger = logger  # type: ignore


def _collect_exp_stats(run):
    """
    Collect information such as CPU user time, maximum memory usage,
    and maximum GPU memory usage and save it in the MongoDB.

    Parameters
    ----------
    run: Sacred run
        Current Sacred run.

    Returns
    -------
    None
    """
    exp_id = run.config['overwrite']
    if exp_id is None or run.unobserved:
        return

    stats = {}

    stats['real_time'] = (datetime.datetime.utcnow() - run.start_time).total_seconds()

    stats['self'] = {}
    stats['self']['user_time'] = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    stats['self']['system_time'] = resource.getrusage(resource.RUSAGE_SELF).ru_stime
    stats['self']['max_memory_bytes'] = (
        1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    stats['children'] = {}
    stats['children']['user_time'] = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_utime
    stats['children']['system_time'] = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_stime
    stats['children']['max_memory_bytes'] = (
        1024 * resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )

    if 'torch' in sys.modules:
        import torch  # type: ignore

        stats['pytorch'] = {}
        if torch.cuda.is_available():
            stats['pytorch']['gpu_max_memory_bytes'] = torch.cuda.max_memory_allocated()

    if 'tensorflow' in sys.modules:
        import tensorflow as tf  # type: ignore

        stats['tensorflow'] = {}
        if int(tf.__version__.split('.')[0]) < 2:
            if tf.test.is_gpu_available():
                with tf.Session() as sess:
                    stats['tensorflow']['gpu_max_memory_bytes'] = int(
                        sess.run(tf.contrib.memory_stats.MaxBytesInUse())
                    )
        else:
            if len(tf.config.experimental.list_physical_devices('GPU')) >= 1:
                if int(tf.__version__.split('.')[1]) >= 5:
                    stats['tensorflow']['gpu_max_memory_bytes'] = (
                        tf.config.experimental.get_memory_info('GPU:0')['peak']
                    )
                else:
                    logging.info(
                        'SEML stats: There is no way to get actual peak GPU memory usage in TensorFlow 2.0-2.4.'
                    )

    collection = get_collection(run.config['db_collection'])
    collection.update_one({'_id': exp_id}, {'$set': {'stats': stats}})


def collect_exp_stats(run):
    logging.warn(
        'seml.collect_exp_stats is deprecated.\n'
        'Use seml.experiment.Experiment instead of sacred.Experiment.\n'
        'seml.experiment.Experiment already includes the statistics collection.\n'
        'See https://github.com/TUM-DAML/seml/blob/master/examples/example_experiment.py'
    )
    _collect_exp_stats(run)


__all__ = [
    'setup_logger',
    'collect_exp_stats',
    'Experiment',
]
