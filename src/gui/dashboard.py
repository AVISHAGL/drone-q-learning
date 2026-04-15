"""Dashboard — live statistics panel updated after every episode."""

import tkinter as tk
from tkinter import ttk

from src.core.episode_stats import EpisodeStats
from src.gui import theme

__all__ = ["Dashboard"]


class Dashboard(ttk.Frame):
    """Side-panel frame displaying live training statistics.

    Stats updated via :meth:`update` after each episode.
    """

    def __init__(self, master: tk.Widget, **kwargs) -> None:
        """Build labelled rows for all seven statistics."""
        super().__init__(master, **kwargs)
        self._vars: dict[str, tk.StringVar] = {}
        rows = [
            ("episode", "Episode"),
            ("steps", "Steps"),
            ("total_reward", "Cumulative Reward"),
            ("epsilon", "Epsilon (ε)"),
            ("elapsed", "Elapsed Time"),
            ("best_reward", "Best Reward"),
            ("last_result", "Last Episode"),
        ]
        for key, label in rows:
            frame = ttk.Frame(self)
            frame.pack(fill="x", padx=4, pady=0)
            ttk.Label(frame, text=label + ":", width=16, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            self._vars[key] = var
            lbl = ttk.Label(frame, textvariable=var, anchor="w")
            lbl.pack(side="left")
            if key == "last_result":
                self._result_label = lbl

        self._best_reward: float = float("-inf")
        self._start_time: float | None = None

    def update(self, stats: EpisodeStats) -> None:
        """Refresh all statistics labels from *stats*.

        Args:
            stats: EpisodeStats from the most recently completed episode.
        """
        import time

        if self._start_time is None:
            self._start_time = time.monotonic()

        elapsed = time.monotonic() - self._start_time
        if stats.total_reward > self._best_reward:
            self._best_reward = stats.total_reward

        self._vars["episode"].set(str(stats.episode))
        self._vars["steps"].set(str(stats.steps))
        self._vars["total_reward"].set(f"{stats.total_reward:.2f}")
        self._vars["elapsed"].set(f"{elapsed:.1f} s")
        self._vars["best_reward"].set(f"{self._best_reward:.2f}")

        if stats.reached_target:
            self._vars["last_result"].set("✓ Success")
            self._result_label.configure(foreground=theme.SUCCESS)
        else:
            self._vars["last_result"].set("✗ Fail")
            self._result_label.configure(foreground=theme.FAIL)

    def reset(self) -> None:
        """Clear all labels to their default values."""
        for var in self._vars.values():
            var.set("—")
        self._best_reward = float("-inf")
        self._start_time = None
        self._result_label.configure(foreground=theme.NEUTRAL)

    def set_epsilon(self, epsilon: float) -> None:
        """Update the epsilon display (called separately from the agent).

        Args:
            epsilon: Current epsilon value.
        """
        self._vars["epsilon"].set(f"{epsilon:.4f}")
