"""HyperparamPanel — hyperparameter config widget used by ControlPanel."""

import tkinter as tk
from tkinter import messagebox, ttk

from src.gui.tooltip import add_tooltip

__all__ = ["HyperparamPanel"]

_SPEED_TIP = "Control animation delay per step (0 ms = fastest)"

_PARAMS = [
    ("Alpha (α)",   "alpha",         "0.1"),
    ("Gamma (γ)",   "gamma",         "0.9"),
    ("Epsilon (ε)", "epsilon",       "1.0"),
    ("ε Decay",     "epsilon_decay", "0.995"),
    ("Episodes",    "episodes",      "1000"),
    ("Max Steps",   "max_steps",     "200"),
]


class HyperparamPanel(ttk.Frame):
    """Config-tab panel: hyperparameter entries, speed slider, and Apply button.

    Args:
        master: Parent widget (Config ttk.Frame inside ControlPanel Notebook).
        sdk:    DroneSimSDK instance for applying parameter changes.
    """

    def __init__(self, master: tk.Widget, sdk, **kw) -> None:  # type: ignore[type-arg]
        super().__init__(master, **kw)
        self._sdk = sdk
        self._build_hyperparams()
        self._build_speed()

    def _build_hyperparams(self) -> None:
        frame = ttk.LabelFrame(self, text="Hyperparameters", padding=(6, 4))
        frame.pack(fill="x", padx=4, pady=(6, 2))
        self._hp_vars: dict[str, tk.StringVar] = {}
        for label, key, default in _PARAMS:
            row_f = ttk.Frame(frame)
            row_f.pack(fill="x", pady=1)
            ttk.Label(row_f, text=label, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self._hp_vars[key] = var
            ttk.Entry(row_f, textvariable=var, width=8).pack(side="left")
        ttk.Button(frame, text="Apply", command=self._apply).pack(fill="x", pady=(4, 0))

    def _build_speed(self) -> None:
        frame = ttk.LabelFrame(self, text="Animation Speed", padding=(6, 4))
        frame.pack(fill="x", padx=4, pady=(4, 6))
        self._speed_var = tk.IntVar(value=0)
        s = ttk.Scale(frame, from_=0, to=500, orient="horizontal",
                      variable=self._speed_var, command=self._on_speed)
        s.pack(fill="x")
        add_tooltip(s, _SPEED_TIP)

    def _apply(self) -> None:
        kwargs: dict[str, float] = {}
        for key, var in self._hp_vars.items():
            try:
                kwargs[key] = float(var.get())
            except ValueError:
                messagebox.showwarning("Invalid", f"{key} must be numeric.")
                return
        kwargs.pop("episodes", None)
        kwargs.pop("max_steps", None)
        if not 0 < kwargs.get("alpha", 0.1) <= 1:
            messagebox.showwarning("Invalid", "Alpha must be in (0, 1].")
            return
        self._sdk.set_hyperparams(**kwargs)

    def _on_speed(self, _val: str) -> None:
        self._sdk.set_vis_delay(self._speed_var.get())
