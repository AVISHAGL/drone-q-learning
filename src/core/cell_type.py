"""Module defining the CellType enum for grid cell classification."""

from enum import Enum

__all__ = ["CellType"]


class CellType(Enum):
    """Enumeration of possible cell types in the 2-D grid world.

    Each cell belongs to exactly one type which determines passability
    and the reward delivered when the drone enters it.
    """

    EMPTY = 1
    BUILDING = 2
    TRAP = 3
    WIND = 4
    TARGET = 5
    START = 6
