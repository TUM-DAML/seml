import functools
from typing import Sequence

import rich
import rich.box
import rich.table
from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule

console = Console()

Table = functools.partial(
    rich.table.Table,
    collapse_padding=True,
    show_lines=False,
    show_edge=False,
    box=rich.box.SIMPLE,
    row_styles=['none', 'dim'],
    padding=(0,0,),
    highlight=True
)


def Heading(text: str):
    return Padding(
        Rule(text, style='red'),
        pad=(1, 0, 0, 0)
    )


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
        table.add_row(*items[i:i+num_columns])

    # Print the table
    console.print(table)
