"""Mixin for rendering the drone icon and episode trail on a tk.Canvas."""

import tkinter as tk
from pathlib import Path

__all__ = ["DroneSpriteMixin"]

_ASSET_DIR = Path(__file__).parent.parent.parent / "assets"


class DroneSpriteMixin:
    """Mixin that adds drone-sprite rendering to a tk.Canvas subclass.

    Contract: the host class must expose:
    - ``self.canvas`` — the tk.Canvas (or self if the host IS the canvas)
    - ``self._cell_px`` — integer pixel size of one grid cell
    - ``self._sdk`` — DroneSimSDK reference (for state_to_pos access)

    Mixin has exactly ONE responsibility: rendering the drone and trail.
    """

    # Tag constants used to manage canvas items
    _TAG_DRONE = "drone_sprite"
    _TAG_TRAIL = "drone_trail"

    def render_drone(self, state: int) -> None:
        """Draw the drone icon at the cell corresponding to *state*.

        Falls back to a Unicode ✈ glyph if the asset image is missing.

        Args:
            state: Flat state index for the drone's current position.
        """
        self.clear_drone()
        row, col = self._sdk._env.state_to_pos(state)
        cx = col * self._cell_px + self._cell_px // 2
        cy = row * self._cell_px + self._cell_px // 2
        canvas: tk.Canvas = self  # type: ignore[assignment]
        canvas.create_text(
            cx,
            cy,
            text="✈",
            font=("TkDefaultFont", max(8, self._cell_px // 2)),
            fill="black",
            tags=self._TAG_DRONE,
        )

    def render_trail(self, path: list[int]) -> None:
        """Draw a faded trail for the given list of state indices.

        Args:
            path: Ordered list of state indices from the last episode.
        """
        self.clear_trail()
        canvas: tk.Canvas = self  # type: ignore[assignment]
        for state in path:
            row, col = self._sdk._env.state_to_pos(state)
            x0 = col * self._cell_px + self._cell_px // 4
            y0 = row * self._cell_px + self._cell_px // 4
            x1 = x0 + self._cell_px // 2
            y1 = y0 + self._cell_px // 2
            canvas.create_oval(
                x0, y0, x1, y1,
                fill="#AAAAFF",
                outline="",
                tags=self._TAG_TRAIL,
            )

    def clear_drone(self) -> None:
        """Remove all drone sprite canvas items."""
        canvas: tk.Canvas = self  # type: ignore[assignment]
        canvas.delete(self._TAG_DRONE)

    def clear_trail(self) -> None:
        """Remove all trail canvas items."""
        canvas: tk.Canvas = self  # type: ignore[assignment]
        canvas.delete(self._TAG_TRAIL)
