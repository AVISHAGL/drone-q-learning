"""2-D grid-world environment for the Drone Q-Learning Simulation."""

from copy import deepcopy

from src.core._grid_helpers import build_grid, find_cell_state
from src.core.cell_type import CellType

__all__ = ["GridWorld"]

_ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # UP/DOWN/LEFT/RIGHT


class GridWorld:
    """Tabular grid-world environment implementing the standard RL interface.

    The environment is a 2-D rectangular grid whose cells are typed
    according to CellType.  The drone navigates by choosing one of four
    discrete actions (UP=0, DOWN=1, LEFT=2, RIGHT=3).

    Args:
        config: Dictionary loaded from config/env.json.  Must contain keys
            'rows', 'cols', 'max_steps_per_episode', 'rewards', and
            'default_grid'.
    """

    def __init__(self, config: dict) -> None:
        """Initialise grid from config dict."""
        self._rows: int = config["rows"]
        self._cols: int = config["cols"]
        self._max_steps: int = config["max_steps_per_episode"]
        self._rewards: dict = config["rewards"]
        self._grid: list[list[CellType]] = build_grid(self._rows, self._cols, config)
        self._start_state: int = find_cell_state(
            self._grid, self._rows, self._cols, CellType.START
        )
        self._step_count: int = 0

    def reset(self) -> int:
        """Reset the episode step counter and return the start state index."""
        self._step_count = 0
        return self._start_state

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        """Execute one action from the given state.

        Args:
            state: Current state index.
            action: Action integer (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

        Returns:
            Tuple of (next_state, reward, done).
        """
        self._step_count += 1
        row, col = self.state_to_pos(state)
        dr, dc = _ACTIONS[action]
        nr, nc = row + dr, col + dc

        if not self._in_bounds(nr, nc) or self._grid[nr][nc] is CellType.BUILDING:
            reward = float(self._rewards["building"])
            done = self._step_count >= self._max_steps
            return state, reward, done

        cell = self._grid[nr][nc]
        next_state = self.pos_to_state(nr, nc)
        reward, done = self._cell_reward(cell)

        if not done and self._step_count >= self._max_steps:
            done = True

        return next_state, reward, done

    def set_cell(self, row: int, col: int, cell_type: CellType) -> None:
        """Change the type of a single cell.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).
            cell_type: New cell type to assign.
        """
        self._grid[row][col] = cell_type

    def get_grid(self) -> list[list[CellType]]:
        """Return a deep copy of the grid.

        Returns:
            2-D list of CellType values.
        """
        return deepcopy(self._grid)

    def state_to_pos(self, state: int) -> tuple[int, int]:
        """Convert a flat state index to (row, col).

        Args:
            state: Flat state index in [0, num_states).

        Returns:
            Tuple (row, col).
        """
        return divmod(state, self._cols)

    def pos_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to a flat state index.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Flat state index.
        """
        return row * self._cols + col

    @property
    def num_states(self) -> int:
        """Total number of states (rows x cols)."""
        return self._rows * self._cols

    @property
    def num_actions(self) -> int:
        """Number of discrete actions (always 4)."""
        return 4

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._rows and 0 <= col < self._cols

    def _cell_reward(self, cell: CellType) -> tuple[float, bool]:
        mapping = {
            CellType.EMPTY: (float(self._rewards["empty"]), False),
            CellType.WIND: (float(self._rewards["wind"]), False),
            CellType.TRAP: (float(self._rewards["trap"]), False),
            CellType.TARGET: (float(self._rewards["target"]), True),
            CellType.START: (float(self._rewards["empty"]), False),
        }
        return mapping.get(cell, (float(self._rewards["empty"]), False))
