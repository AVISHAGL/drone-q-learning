"""GridSDK — grid and environment management operations for DroneSimSDK."""

from copy import deepcopy
from pathlib import Path

from src.core.cell_type import CellType
from src.core.grid_world import GridWorld
from src.sdk.persistence import load_grid, save_grid

__all__ = ["GridSDK"]


class GridSDK:
    """Manages grid/environment mutation, persistence, and default-layout restore.

    Held as a private component of DroneSimSDK; all public access goes through
    the DroneSimSDK facade.

    Args:
        env: Live GridWorld instance shared with the training loop.
    """

    def __init__(self, env: GridWorld) -> None:
        """Store environment reference and snapshot the startup grid layout."""
        self._env = env
        self._default_grid = deepcopy(env.get_grid())

    def update_grid(self, row: int, col: int, cell_type: CellType) -> None:
        """Change a single cell in the live environment.

        Args:
            row: Row index.
            col: Column index.
            cell_type: New cell type to apply.
        """
        self._env.set_cell(row, col, cell_type)

    def get_grid(self) -> list[list[CellType]]:
        """Return the current grid layout (deep copy)."""
        return self._env.get_grid()

    def load_default_grid(self) -> None:
        """Restore the grid to the layout captured at startup."""
        for r, row in enumerate(self._default_grid):
            for c, cell in enumerate(row):
                self._env.set_cell(r, c, cell)

    def save_grid_to_file(self, path: str | Path) -> None:
        """Save the current grid layout to a JSON file.

        Args:
            path: Destination file path.
        """
        save_grid(self._env.get_grid(), path)

    def load_grid_from_file(self, path: str | Path) -> None:
        """Load a grid layout from a JSON file and apply it to the environment.

        Args:
            path: Source JSON file path.
        """
        grid = load_grid(path)
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                self._env.set_cell(r, c, cell)
