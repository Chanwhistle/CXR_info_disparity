"""Rank-aware logging helpers.

Kept separate from core.utils so that core.dataloader can use them without a
circular import.
"""

import os


def is_main_process():
    """Return True for the only process that should emit shared logs/files."""
    return int(os.environ.get("RANK", "0")) == 0


def rank_zero_print(*args, **kwargs):
    """Print once during distributed execution."""
    if is_main_process():
        print(*args, **kwargs)
