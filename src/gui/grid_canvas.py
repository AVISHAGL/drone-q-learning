"""GridCanvas — colour-coded tile map with value heatmap, policy arrows, and drone overlay."""

import tkinter as tk

import numpy as np

from src.core.cell_type import CellType
from src.gui.drone_sprite import DroneSpriteMixin
from src.gui.grid_editor import GridEditorMixin

__all__ = ["GridCanvas"]

_ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# Semantic cell colours — kept as-is per spec; only EMPTY gets value shading.
_COLORS: dict[CellType, str] = {
    CellType.EMPTY:    "#1e1e2f",  # dark base; overridden by value heatmap
    CellType.BUILDING: "#808080",
    CellType.TRAP:     "#FF4444",
    CellType.WIND:     "#4444FF",
    CellType.TARGET:   "#44FF44",
    CellType.START:    "#FFFF00",
}

# Value heatmap gradient stops for EMPTY cells
#   low  → dark burgundy  (unfavourable)
#   mid  → dark base      (neutral / uninitialised)
#   high → deep purple    (favourable)
_VAL_LOW  = "#6b1a1a"   # burgundy
_VAL_MID  = "#1e1e2f"   # matches EMPTY base / theme.BG
_VAL_HIGH = "#4c1d95"   # deep purple

# Grid chrome colours
_CELL_OUTLINE  = "#3a3a5c"   # subtle dark border between cells
_ARROW_COLOR   = "#e2e8f0"   # light arrows, visible on dark backgrounds

# Non-EMPTY cell types that skip the value overlay
_OVERLAY_SKIP = frozenset({
    CellType.BUILDING, CellType.TRAP, CellType.WIND,
    CellType.TARGET, CellType.START,
})


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _hex_lerp(a: str, b: str, t: float) -> str:
    """Linearly interpolate between two '#rrggbb' hex colours; t in [0, 1]."""
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return f"#{int(ar + (br - ar) * t):02x}{int(ag + (bg - ag) * t):02x}{int(ab + (bb - ab) * t):02x}"


def _value_to_color(t: float) -> str:
    """Map a normalised value t ∈ [0, 1] to a heatmap colour.

    0.0 → VAL_LOW (burgundy / unfavourable)
    0.5 → VAL_MID (dark base / neutral)
    1.0 → VAL_HIGH (deep purple / favourable)
    """
    if t <= 0.5:
        return _hex_lerp(_VAL_LOW, _VAL_MID, t * 2.0)
    return _hex_lerp(_VAL_MID, _VAL_HIGH, (t - 0.5) * 2.0)


# ---------------------------------------------------------------------------
# GridCanvas
# ---------------------------------------------------------------------------

class GridCanvas(GridEditorMixin, DroneSpriteMixin, tk.Canvas):
    """tk.Canvas that renders the grid with a value heatmap, policy arrows,
    episode trail, and drone sprite.

    Args:
        master: Parent widget.
        sdk:    DroneSimSDK instance providing grid, Q-table, and policy data.
        kwargs: Forwarded to tk.Canvas.
    """

    def __init__(self, master: tk.Widget, sdk, **kwargs) -> None:  # type: ignore[type-arg]
        super().__init__(master, **kwargs)
        self._sdk = sdk
        cfg = sdk._env_cfg
        self._rows: int = cfg["rows"]
        self._cols: int = cfg["cols"]
        self._cell_px: int = 50          # updated dynamically on <Configure>
        self._trail: list[int] = []
        self._drone_state: int | None = None
        self._edit_mode_enabled: bool = False
        self._show_heatmap: bool = True
        self._show_arrows: bool = True
        self.set_edit_mode(True)
        self.bind("<Configure>", self._on_resize)

    # ------------------------------------------------------------------
    # Edit mode
    # ------------------------------------------------------------------

    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode_enabled = enabled
        if enabled:
            self.bind_edit_events()
        else:
            self.unbind("<Button-1>")
            self.unbind("<B1-Motion>")

    def is_edit_mode(self) -> bool:
        return self._edit_mode_enabled

    def _on_resize(self, event: tk.Event) -> None:
        new_px = min(event.width // self._cols, event.height // self._rows)
        if new_px > 0 and new_px != self._cell_px:
            self._cell_px = new_px
            self.after_idle(lambda: self.refresh(policy=self._sdk.get_policy()))

    # ------------------------------------------------------------------
    # Public refresh API
    # ------------------------------------------------------------------

    def toggle_heatmap(self) -> None:
        """Toggle value-heatmap colouring on EMPTY cells."""
        self._show_heatmap = not self._show_heatmap
        self.after_idle(lambda: self.refresh(policy=self._sdk.get_policy()))

    def toggle_arrows(self) -> None:
        """Toggle policy-arrow overlay."""
        self._show_arrows = not self._show_arrows
        self.after_idle(lambda: self.refresh(policy=self._sdk.get_policy()))

    def refresh(
        self,
        policy: dict[int, int] | None = None,
        drone_state: int | None = None,
        trail: list[int] | None = None,
    ) -> None:
        """Full redraw: heatmap grid → policy arrows → trail → drone."""
        self.draw_grid()
        if policy and self._show_arrows:
            self.draw_policy_arrows(policy)
        if trail:
            self.render_trail(trail)
        if drone_state is not None:
            self.render_drone(drone_state)

    # ------------------------------------------------------------------
    # Grid drawing with value heatmap
    # ------------------------------------------------------------------

    def draw_grid(self) -> None:
        """Draw all cells; EMPTY cells are coloured by normalised Q-value."""
        self.delete("grid_tile")
        grid   = self._sdk.get_grid()
        vmap   = self._build_value_map() if self._show_heatmap else None

        for r in range(self._rows):
            for c in range(self._cols):
                ct    = grid[r][c]
                color = self._cell_color(ct, r, c, vmap)
                x0, y0 = c * self._cell_px,       r * self._cell_px
                x1, y1 = x0 + self._cell_px,      y0 + self._cell_px
                self.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color,
                    outline=_CELL_OUTLINE,
                    tags="grid_tile",
                )

    def _cell_color(
        self,
        ct: CellType,
        row: int,
        col: int,
        vmap: np.ndarray | None,
    ) -> str:
        """Return the display colour for a cell.

        Semantic cells keep their fixed colour.
        EMPTY cells are tinted by their normalised state value when a
        meaningful value map is available.
        """
        if ct in _OVERLAY_SKIP:
            return _COLORS[ct]

        # EMPTY — apply value heatmap when trained
        if vmap is not None:
            state = row * self._cols + col
            if 0 <= state < len(vmap):
                return _value_to_color(float(vmap[state]))

        return _COLORS[CellType.EMPTY]

    def _build_value_map(self) -> np.ndarray | None:
        """Return per-state normalised V(s) = max_a Q(s,a) in [0, 1].

        Returns None when the Q-table is still uniform (no information yet).
        """
        try:
            q = self._sdk._agent._q          # live array — no copy needed
            vals: np.ndarray = q.max(axis=1) # shape (n_states,)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-6:
                return None
            return (vals - lo) / (hi - lo)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Policy arrows
    # ------------------------------------------------------------------

    def draw_policy_arrows(self, policy: dict[int, int]) -> None:
        """Overlay directional arrows for the greedy policy on visited states only.

        Arrows are only drawn for states the agent has visited (i.e. taken at
        least one action from) so that uninitialised zero-Q states remain clean.
        """
        self.delete("policy_arrow")
        grid    = self._sdk.get_grid()
        visited = self._sdk.get_visited_states()
        fsize   = max(8, self._cell_px // 3)

        for state, action in policy.items():
            if state not in visited:
                continue
            row, col = self._sdk._env.state_to_pos(state)
            if grid[row][col] in (CellType.BUILDING, CellType.TARGET):
                continue
            cx = col * self._cell_px + self._cell_px // 2
            cy = row * self._cell_px + self._cell_px // 2
            self.create_text(
                cx, cy,
                text=_ARROW[action],
                font=("TkDefaultFont", fsize, "bold"),
                fill=_ARROW_COLOR,
                tags="policy_arrow",
            )

    # ------------------------------------------------------------------
    # Delegation helpers (used by App)
    # ------------------------------------------------------------------

    def draw_drone(self, state: int) -> None:
        self.render_drone(state)

    def draw_trail(self, path: list[int]) -> None:
        self.render_trail(path)
