"""GraphPanel — live convergence graph embedded in a tkinter frame."""

import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.core.episode_stats import EpisodeStats
from src.gui import theme

__all__ = ["GraphPanel"]


class GraphPanel(ttk.Frame):
    """Frame containing a live matplotlib convergence graph.

    Shows total reward per episode (top) and max ΔQ per episode (bottom).
    Provides zoom/pan via NavigationToolbar2Tk and PNG export.
    """

    def __init__(self, master: tk.Widget, vis_every_n: int = 10, **kwargs) -> None:
        """Create figure, axes, canvas, and toolbar.

        Args:
            master: Parent widget.
            vis_every_n: Redraw every N episodes for performance.
            kwargs: Forwarded to ttk.Frame.
        """
        super().__init__(master, **kwargs)
        self._vis_n = max(1, vis_every_n)
        self._episodes: list[int] = []
        self._rewards: list[float] = []
        self._delta_qs: list[float] = []
        self._show_dq = tk.BooleanVar(value=True)

        self._fig = Figure(figsize=(6, 2.0), tight_layout=True)
        self._ax_reward = self._fig.add_subplot(2, 1, 1)
        self._ax_dq = self._fig.add_subplot(2, 1, 2)
        theme.apply_to_figure(self._fig, [self._ax_reward, self._ax_dq])
        self._ax_reward.set_ylabel("Total Reward")
        self._ax_dq.set_xlabel("Episode")
        self._ax_dq.set_ylabel("Max ΔQ")

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().configure(bg=theme.BG)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self._canvas, self)
        toolbar.configure(bg=theme.BG)
        toolbar.update()

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x")
        ttk.Checkbutton(ctrl, text="Show ΔQ Line", variable=self._show_dq,
                        command=self._redraw).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Export PNG", command=self._export_png).pack(side="left")

    def append(self, stats: EpisodeStats) -> None:
        """Add episode stats and conditionally redraw.

        Args:
            stats: EpisodeStats from the latest episode.
        """
        self._episodes.append(stats.episode)
        self._rewards.append(stats.total_reward)
        self._delta_qs.append(stats.max_delta_q)
        if len(self._episodes) % self._vis_n == 0:
            self._redraw()

    def reset(self) -> None:
        """Clear data and redraw empty axes."""
        self._episodes.clear()
        self._rewards.clear()
        self._delta_qs.clear()
        self._redraw()

    def export_png(self, path: str) -> None:
        """Save the figure as a PNG file.

        Args:
            path: Destination file path.
        """
        self._fig.savefig(path, dpi=150)

    def _redraw(self) -> None:
        self._ax_reward.cla()
        self._ax_dq.cla()
        # Re-apply axis dark styling after cla() clears it
        theme.apply_to_figure(self._fig, [self._ax_reward, self._ax_dq])
        if self._episodes:
            self._ax_reward.plot(self._episodes, self._rewards,
                                 color=theme.PLOT_LINE1, linewidth=1.5)
            if self._show_dq.get():
                self._ax_dq.plot(self._episodes, self._delta_qs,
                                 color=theme.PLOT_LINE2, linewidth=1.5)
        self._ax_reward.set_ylabel("Total Reward")
        self._ax_dq.set_xlabel("Episode")
        self._ax_dq.set_ylabel("Max ΔQ")
        self._canvas.draw_idle()

    def _export_png(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            self.export_png(path)
