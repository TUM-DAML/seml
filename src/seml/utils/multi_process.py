import functools
import os
import sys
from typing import Callable, Optional, TypeVar, Union, overload

from typing_extensions import ParamSpec

_LOCAL_ID = 'SLURM_LOCALID'
_PROCESS_ID = 'SLURM_PROCID'
_PROCESS_COUNT = 'SLURM_NTASKS'


def process_id():
    return int(os.environ.get(_PROCESS_ID, 0))


def local_id():
    return int(os.environ.get(_LOCAL_ID, 0))


def process_count():
    return int(os.environ.get(_PROCESS_COUNT, 1))


def is_main_process():
    return process_id() == 0


def is_local_main_process():
    return local_id() == 0


def is_running_in_multi_process():
    return process_count() > 1


class ChildProcessSkip(Exception): ...


class MainProcessExecuteContext:
    def __enter__(self):
        if not is_main_process():
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise ChildProcessSkip()

    def __exit__(self, type, value, traceback):
        if type is None:
            return  # No exception
        if issubclass(type, ChildProcessSkip):
            return True  # Suppress special SkipWithBlock exception


P = ParamSpec('P')
R = TypeVar('R')


@overload
def only_on_main_process(func: Callable[P, R]) -> Callable[P, Optional[R]]: ...


@overload
def only_on_main_process(func: None = None) -> MainProcessExecuteContext: ...


def only_on_main_process(
    func: Optional[Callable[P, R]] = None,
) -> Union[Callable[P, Optional[R]], MainProcessExecuteContext]:
    if callable(func):

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            if is_main_process():
                return func(*args, **kwargs)
            return None

        return wrapper
    else:
        return MainProcessExecuteContext()
