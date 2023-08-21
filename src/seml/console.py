import functools

import rich
import rich.box
import rich.table
from rich.align import Align
from rich.console import Console
from rich.panel import Panel

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
    return Panel(
        Align(
            text,
            align="center"
        )
    )
