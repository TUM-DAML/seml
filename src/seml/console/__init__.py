import functools
import os
from contextlib import contextmanager
from typing import Sequence

import rich
import rich.box
import rich.progress
import rich.table
from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule

from seml.utils.typer import prompt as typer_prompt

try:
    terminal_width = os.get_terminal_size().columns
except OSError:
    from seml.settings import SETTINGS

    terminal_width = SETTINGS.EXPERIMENT.TERMINAL_WIDTH

console = Console(
    width=terminal_width,
)

Table = functools.partial(
    rich.table.Table,
    collapse_padding=True,
    show_lines=False,
    show_edge=False,
    box=rich.box.SIMPLE,
    row_styles=['none', 'dim'],
    padding=(
        0,
        0,
    ),
    highlight=True,
)


@contextmanager
def pause_live_widget():
    prev_live = console._live
    if prev_live:
        prev_live.stop()
    console.clear_live()
    yield
    if prev_live:
        prev_live.start()


@functools.wraps(rich.progress.track)
def track(*args, **kwargs):
    """
    Wrapper for `rich.progress.track` that uses the global console instance and
    avoids creating empty ipython widgets in jupyter notebooks.

    Parameters
    ----------
    *args : Any
        Positional arguments to pass to `rich.progress.track`.
    **kwargs : Any
        Keyword arguments to pass to `rich.progress.track`.
    """
    # Directly return the sequence if the track is disabled. This avoids empty
    # ipywidgets in jupyter instances.
    if kwargs.get('disable', False):
        if len(args) == 0:
            yield from kwargs['sequence']
        else:
            yield from args[0]
        return

    if console not in kwargs:
        kwargs['console'] = console

    # Since there can only be one live instance at a time, we first need to stop
    # the previous one and then restart it after the new one is done.
    with pause_live_widget():
        yield from rich.progress.track(*args, **kwargs)


@functools.wraps(typer_prompt)
def prompt(*args, **kwargs):
    with pause_live_widget():
        return typer_prompt(*args, **kwargs)


def Heading(text: str):
    """
    Convenience function to create a seperator in the console.

    Parameters
    ----------
    text : str
        The text to display in the seperator.
    """
    return Padding(Rule(text, style='red'), pad=(1, 0, 0, 0))


def list_items(items: Sequence[str]):
    """
    Print a list of items in columns, using as many columns as possible
    while keeping the items readable. This will look similar to bash autcompletition suggestions.

    Parameters
    ----------
    items : Sequence[str]
        The items to print.
    """
    # Calculate available width
    available_width = console.width

    # Determine the number of columns based on available width
    max_len = max(len(s) for s in items)
    num_columns = max(1, available_width // (max_len + 1))

    # Create a table with the calculated number of columns
    table = rich.table.Table.grid(expand=True, pad_edge=True)
    for i in range(num_columns):
        table.add_column()

    # Add suggestions to the table, distributing them across columns
    for i in range(0, len(items), num_columns):
        table.add_row(*items[i : i + num_columns])

    # Print the table
    console.print(table)
