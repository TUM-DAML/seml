from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def tail_file(path: str | Path, n: int = 1):
    """
    Returns the last n lines of a file.

    Args:
        path (str | Path): Path to the file.
        n (int, optional): Number of lines to return. Defaults to 1.

    Returns:
        str: The last n lines of the file.
    """
    if n == 0:
        return ''
    num_newlines = 0
    with open(path, 'rb') as f:
        try:
            f.seek(-1, os.SEEK_END)
            if f.read(1) == b'\n':
                num_newlines += 1
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) in (b'\n', b'\r'):
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.read().decode(errors='replace')
    return last_line
