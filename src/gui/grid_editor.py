"""GridEditorMixin — click/drag cell painting with constraint enforcement."""

import tkinter as tk
from collections import deque
from tkinter import ttk

from src.core.cell_type import CellType

__all__ = ["GridEditorMixin"]

# Ordered for display; set used for constraint checks elsewhere
_PAINTABLE = {CellType.EMPTY, CellType.BUILDING, CellType.TRAP, CellType.WIND,
              CellType.TARGET, CellType.START}

# Fixed display order for the 2-column palette grid
_PALETTE_ORDER = [
    CellType.EMPTY,    CellType.BUILDING,
    CellType.TRAP,     CellType.WIND,
    CellType.TARGET,   CellType.START,
]


class GridEditorMixin:
    """Mixin that adds click/drag grid editing to a GridCanvas subclass.

    Contract: host must expose ``self._sdk``, ``self._cell_px``,
    ``self._rows``, ``self._cols``.

    Responsibility: cell painting and constraint enforcement only.
    """

    def bind_edit_events(self) -> None:
        """Bind mouse events for click and drag cell editing."""
        self.bind("<Button-1>", self._on_cell_click)
        self.bind("<B1-Motion>", self._on_cell_drag)

    def get_selected_cell_type(self) -> CellType:
        """Return the currently selected palette cell type."""
        return getattr(self, "_selected_cell_type", CellType.EMPTY)

    def add_cell_type_palette(self, master: tk.Widget) -> None:
        """Create a 2-column radio-button palette for each paintable cell type.

        Args:
            master: Parent widget for the palette frame.
        """
        self._selected_var = tk.StringVar(value=CellType.EMPTY.name)
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        for i, ct in enumerate(_PALETTE_ORDER):
            row, col = divmod(i, 2)
            ttk.Radiobutton(
                master,
                text=ct.name.capitalize(),
                variable=self._selected_var,
                value=ct.name,
            ).grid(row=row, column=col, sticky="w", padx=6, pady=2)

    def show_edit_warning(self, message: str) -> None:
        """Show a warning message box.

        Args:
            message: Warning text to display.
        """
        tk.messagebox.showwarning("Grid Editor", message)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_cell_click(self, event: tk.Event) -> None:
        row, col = self._pixel_to_cell(event.x, event.y)
        if self._valid(row, col):
            self._paint_cell(row, col)

    def _on_cell_drag(self, event: tk.Event) -> None:
        row, col = self._pixel_to_cell(event.x, event.y)
        if self._valid(row, col):
            self._paint_cell(row, col)

    def _pixel_to_cell(self, px: int, py: int) -> tuple[int, int]:
        return py // self._cell_px, px // self._cell_px

    def _valid(self, row: int, col: int) -> bool:
        return 0 <= row < self._rows and 0 <= col < self._cols

    def _paint_cell(self, row: int, col: int) -> None:
        ct_name = getattr(self, "_selected_var", None)
        cell_type = (
            CellType[ct_name.get()] if ct_name else CellType.EMPTY
        )
        prev = self._sdk.get_grid()[row][col]
        if not self._enforce_start_target(row, col, cell_type):
            return
        self._sdk.update_grid(row, col, cell_type)
        if not self._check_reachability():
            self._sdk.update_grid(row, col, prev)
            self.show_edit_warning("No walkable path from Start to Target — edit reverted.")
            return
        if hasattr(self, "refresh"):
            self.refresh(policy=self._sdk.get_policy())

    def _enforce_start_target(self, row: int, col: int, ct: CellType) -> bool:
        grid = self._sdk.get_grid()
        if ct is CellType.START:
            for r in range(self._rows):
                for c in range(self._cols):
                    if grid[r][c] is CellType.START and (r, c) != (row, col):
                        self._sdk.update_grid(r, c, CellType.EMPTY)
        if ct is CellType.TARGET:
            for r in range(self._rows):
                for c in range(self._cols):
                    if grid[r][c] is CellType.TARGET and (r, c) != (row, col):
                        self._sdk.update_grid(r, c, CellType.EMPTY)
        return True

    def _find_cell(self, ct: CellType) -> tuple[int, int] | None:
        for r, row in enumerate(self._sdk.get_grid()):
            for c, cell in enumerate(row):
                if cell is ct:
                    return r, c
        return None

    def _check_reachability(self) -> bool:
        start = self._find_cell(CellType.START)
        target = self._find_cell(CellType.TARGET)
        if start is None or target is None:
            return False
        grid = self._sdk.get_grid()
        visited = set()
        q: deque = deque([start])
        visited.add(start)
        while q:
            r, c = q.popleft()
            if (r, c) == target:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self._rows and 0 <= nc < self._cols
                        and (nr, nc) not in visited
                        and grid[nr][nc] is not CellType.BUILDING):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False
