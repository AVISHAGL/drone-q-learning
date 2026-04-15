"""Utilities for saving and loading Q-Tables and grid layouts."""

import json
from pathlib import Path

import numpy as np

from src.core.cell_type import CellType

__all__ = ["save_q_table", "load_q_table", "save_grid", "load_grid"]


def save_q_table(q: np.ndarray, path: str | Path) -> None:
    """Save a Q-Table to a .npy file.

    Args:
        q: NumPy array of shape (num_states, num_actions).
        path: Destination file path (should end in .npy).
    """
    np.save(str(path), q)


def load_q_table(path: str | Path) -> np.ndarray:
    """Load a Q-Table from a .npy file.

    Args:
        path: Source file path.

    Returns:
        NumPy array of shape (num_states, num_actions).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path)
    if not resolved.exists():
        msg = f"Q-Table file not found: {resolved}"
        raise FileNotFoundError(msg)
    return np.load(str(resolved))


def save_grid(grid: list[list[CellType]], path: str | Path) -> None:
    """Serialise a grid to a JSON file (stores cell type names).

    Args:
        grid: 2-D list of CellType values.
        path: Destination JSON file path.
    """
    data = [[cell.name for cell in row] for row in grid]
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def load_grid(path: str | Path) -> list[list[CellType]]:
    """Deserialise a grid from a JSON file.

    Args:
        path: Source JSON file path.

    Returns:
        2-D list of CellType values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path)
    if not resolved.exists():
        msg = f"Grid file not found: {resolved}"
        raise FileNotFoundError(msg)
    with resolved.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [[CellType[name] for name in row] for row in data]
