"""App — root tkinter window for the Drone Q-Learning Simulation."""

import queue
import tkinter as tk
from tkinter import ttk

from src.core.version import __version__
from src.gui import theme
from src.gui.control_panel import ControlPanel
from src.gui.dashboard import Dashboard
from src.gui.graph_panel import GraphPanel
from src.gui.grid_canvas import GridCanvas
from src.gui.status_bar import StatusBar
from src.sdk.drone_sim_sdk import DroneSimSDK

__all__ = ["App"]

# Right panel minimum width (px). Controls layout must fit inside this.
_RIGHT_W = 255
# Approximate fixed height of the graph panel (figure + toolbar + ctrl row).
# Used to compute how much vertical space remains for the canvas square.
_GRAPH_H = 250


class App(tk.Tk):
    """Root application window.

    Layout (horizontal pack):
    - Left column : square GridCanvas (dominant) + GraphPanel below it
    - Right column: narrow ControlPanel + Stats; expands to fill remainder
    """

    def __init__(self) -> None:
        """Initialise window, SDK, and all child widgets."""
        super().__init__()
        theme.configure_style()
        self.configure(bg=theme.BG)
        self.title(f"Drone Q-Learning Simulation v{__version__}")
        self.minsize(700, 560)

        try:
            self._sdk = DroneSimSDK()
        except FileNotFoundError as exc:
            import tkinter.messagebox as mb
            mb.showerror("Config Error", str(exc))
            self.destroy()
            return

        self._vis_n: int = int(self._sdk._rl_cfg.get("vis_every_n", 10))
        self._last_trail: list[int] = []
        self._syncing: bool = False          # re-entrancy guard for _sync_canvas_size
        self._last_size: int = 0             # last size applied to canvas
        self._build_layout()
        # Realize natural widget sizes, then pin root geometry so that
        # canvas.config() calls cannot cause the root window to auto-shrink.
        self.update_idletasks()
        self.geometry(self.geometry())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_queue)
        self.bind("<Configure>", self._sync_canvas_size)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # Left column: wraps tightly around canvas + graph; does NOT expand
        # horizontally so the graph always aligns with the grid width exactly.
        left = ttk.Frame(self)

        # Right column: fills all remaining horizontal space.
        right = ttk.Frame(self)

        # Canvas created before StatusBar so it can be passed in directly.
        self._canvas = GridCanvas(left, self._sdk, width=700, height=700,
                                  bg=theme.BG, highlightthickness=0)

        # Status bar packed first (bottom) so it spans the full window width.
        self._bar = StatusBar(self, self._sdk, self._canvas)
        self._bar.pack(side="bottom", fill="x")

        left.pack(side="left", fill="y")
        right.pack(side="left", fill="both", expand=True)

        self._canvas.pack()
        self._canvas.refresh(policy=self._sdk.get_policy())

        # Graph is naturally the same width as the canvas because both live
        # inside the left frame; fill="x" spans that exact width.
        self._graph = GraphPanel(left, vis_every_n=self._vis_n)
        self._graph.pack(fill="x")

        self._ctrl = ControlPanel(
            right,
            self._sdk,
            canvas=self._canvas,
        )
        self._ctrl.pack(fill="x")

        stats_frame = ttk.LabelFrame(right, text="Stats", padding=(4, 2))
        stats_frame.pack(fill="both", expand=True, padx=4, pady=(4, 4))

        self._dash = Dashboard(stats_frame)
        self._dash.pack(fill="both", expand=True)

    def _sync_canvas_size(self, event: tk.Event) -> None:
        """Keep the canvas a perfect square filling the available left space.

        Called on every root <Configure>.  Guards against redundant updates
        so it cannot create an infinite Configure loop.
        """
        if event.widget is not self:
            return
        if self._syncing:
            return
        graph_h = self._graph.winfo_reqheight() or _GRAPH_H
        bar_h = self._bar.winfo_reqheight()
        avail_w = max(300, event.width  - _RIGHT_W)
        avail_h = max(300, event.height - graph_h - bar_h)
        size = min(avail_w, avail_h)
        if size == self._last_size:
            return
        self._syncing = True
        self._last_size = size
        self._canvas.config(width=size, height=size)
        self.after_idle(self._clear_syncing)

    def _clear_syncing(self) -> None:
        self._syncing = False

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        q = self._sdk.get_stats_queue()
        try:
            while True:
                item = q.get_nowait()
                if hasattr(item, "episode"):
                    # EpisodeStats — update dashboard, graph, and canvas
                    self._dash.update(item)
                    self._dash.set_epsilon(self._sdk._agent._epsilon)
                    self._graph.append(item)
                    if item.episode % self._vis_n == 0:
                        self._canvas.refresh(
                            policy=self._sdk.get_policy(),
                            trail=self._last_trail if self._last_trail else None,
                        )
                else:
                    # StepUpdate — render live drone position and trail
                    self._last_trail = item.trail
                    self._canvas.refresh(
                        policy=self._sdk.get_policy(),
                        drone_state=item.state,
                        trail=item.trail,
                    )
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_greedy(self, path: list[int], stats) -> None:  # type: ignore[type-arg]
        """Show the greedy evaluation path on the canvas."""
        self._last_trail = path
        self._canvas.refresh(
            policy=self._sdk.get_policy(),
            drone_state=path[-1] if path else None,
            trail=path,
        )

    def _on_grid_reset(self) -> None:
        """Redraw canvas after a grid reset."""
        self._canvas.refresh(policy=self._sdk.get_policy())

    def on_cell_edit(self, row: int, col: int, cell_type) -> None:  # type: ignore[type-arg]
        """Handle a cell edit from the grid editor (public API for tests).

        Args:
            row: Row index.
            col: Column index.
            cell_type: New CellType.
        """
        self._sdk.update_grid(row, col, cell_type)
        self._canvas.refresh(policy=self._sdk.get_policy())

    def _on_close(self) -> None:
        self._sdk.stop()
        self.destroy()
