import functools
import os
from typing import Sequence

import rich
import rich.box
import rich.progress
import rich.table
from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule


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
    if kwargs['disable']:
        if len(args) == 0:
            return kwargs['sequence']
        return args[0]
    if console not in kwargs:
        kwargs['console'] = console

    def iterator():
        # This iterator is used to make sure that there is only one live object at a time.
        prev_live = console._live
        if prev_live:
            prev_live.stop()
        console.clear_live()
        for result in rich.progress.track(*args, **kwargs):
            yield result
        if prev_live:
            prev_live.start()

    return iterator()


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
