"""Internal helpers for GridWorld — grid construction and cell search."""

from src.core.cell_type import CellType

__all__: list[str] = []


def build_grid(
    rows: int, cols: int, config: dict
) -> list[list[CellType]]:
    """Construct the initial grid from config.

    If 'default_grid' is non-empty, parse cell type names from it.
    Otherwise create an empty grid with START at (0,0) and TARGET at
    (rows-1, cols-1).

    Args:
        rows: Number of rows.
        cols: Number of columns.
        config: Environment config dict.

    Returns:
        2-D list of CellType values.
    """
    default = config.get("default_grid", [])
    if default:
        return [[CellType[name] for name in row] for row in default]
    grid: list[list[CellType]] = [
        [CellType.EMPTY for _ in range(cols)] for _ in range(rows)
    ]
    grid[0][0] = CellType.START
    grid[rows - 1][cols - 1] = CellType.TARGET
    return grid


def find_cell_state(
    grid: list[list[CellType]], rows: int, cols: int, target: CellType
) -> int:
    """Return the flat state index of the first matching cell.

    Args:
        grid: 2-D CellType grid.
        rows: Number of rows.
        cols: Number of columns.
        target: Cell type to search for.

    Returns:
        Flat state index row * cols + col.

    Raises:
        ValueError: If the target cell type is not found.
    """
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is target:
                return r * cols + c
    msg = f"No {target} cell found in grid"
    raise ValueError(msg)
