"""ControlPanel — tabbed side-panel for grid editing and configuration."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.core.cell_type import CellType
from src.gui.grid_canvas import _COLORS
from src.gui.tooltip import add_tooltip

__all__ = ["ControlPanel"]

_TIP = {
    "save":  "Export current Q-Table to .npy file",
    "load":  "Import Q-Table from .npy file",
    "speed": "Control animation delay per step (0 ms = fastest)",
}


class ControlPanel(ttk.Frame):
    """Tabbed side-panel frame.

    Tab "Train"  — edit mode toggle, cell-type palette, Q-Table save/load.
    Tab "Config" — hyperparameters, speed slider.

    Training is controlled entirely via keyboard hotkeys (see StatusBar).
    """

    def __init__(self, master, sdk, canvas=None, **kw) -> None:  # type: ignore[type-arg]
        super().__init__(master, **kw)
        self._sdk = sdk
        self._canvas = canvas

        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=2, pady=(4, 0))

        train_tab = ttk.Frame(nb)
        nb.add(train_tab, text="  Train  ")

        config_tab = ttk.Frame(nb)
        nb.add(config_tab, text="  Config  ")

        self._build_train_tab(train_tab)
        self._build_config_tab(config_tab)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        self._build_edit_section(parent)
        self._build_qtable_section(parent)

    def _build_config_tab(self, parent: ttk.Frame) -> None:
        self._build_hyperparam_panel(parent)
        self._build_speed_slider(parent)

    # ------------------------------------------------------------------
    # Edit section
    # ------------------------------------------------------------------

    def _build_edit_section(self, parent: ttk.Frame) -> None:
        """Edit mode toggle, colour preview, and 2-column cell type palette."""
        if not self._canvas:
            return

        outer = ttk.LabelFrame(parent, text="Grid Editing", padding=(6, 4))
        outer.pack(fill="x", padx=4, pady=(6, 2))

        top_row = ttk.Frame(outer)
        top_row.pack(fill="x", pady=(0, 4))

        self._edit_mode_btn = ttk.Button(
            top_row, text="✓ Edit Mode: ON", command=self._toggle_edit_mode,
            width=16,
        )
        self._edit_mode_btn.pack(side="left", padx=(0, 6))

        ttk.Label(top_row, text="Brush:").pack(side="left", padx=(0, 3))

        self._preview_canvas = tk.Canvas(
            top_row, width=22, height=22, bg="#FFFFFF",
            relief="sunken", bd=1, highlightthickness=0,
        )
        self._preview_canvas.pack(side="left", padx=(0, 4))

        self._selected_type_label = ttk.Label(top_row, text="EMPTY", width=9)
        self._selected_type_label.pack(side="left")

        palette_frame = ttk.Frame(outer)
        palette_frame.pack(fill="x")
        self._palette_frame = palette_frame
        self._canvas.add_cell_type_palette(palette_frame)

        if hasattr(self._canvas, "_selected_var"):
            self._canvas._selected_var.trace("w", self._update_preview)
            self._update_preview()

    # ------------------------------------------------------------------
    # Q-Table persistence section
    # ------------------------------------------------------------------

    def _build_qtable_section(self, parent: ttk.Frame) -> None:
        """Save and Load Q-Table buttons."""
        frame = ttk.LabelFrame(parent, text="Q-Table", padding=(4, 2))
        frame.pack(fill="x", padx=4, pady=(4, 2))
        btn_save = ttk.Button(frame, text="Save Q-Table", command=self._save)
        btn_save.pack(fill="x", padx=2, pady=1)
        add_tooltip(btn_save, _TIP["save"])
        btn_load = ttk.Button(frame, text="Load Q-Table", command=self._load)
        btn_load.pack(fill="x", padx=2, pady=1)
        add_tooltip(btn_load, _TIP["load"])

    # ------------------------------------------------------------------
    # Hyperparameter panel
    # ------------------------------------------------------------------

    def _build_hyperparam_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Hyperparameters", padding=(6, 4))
        frame.pack(fill="x", padx=4, pady=(6, 2))
        params = [
            ("Alpha (α)",   "alpha",         "0.1"),
            ("Gamma (γ)",   "gamma",         "0.9"),
            ("Epsilon (ε)", "epsilon",       "1.0"),
            ("ε Decay",     "epsilon_decay", "0.995"),
            ("Episodes",    "episodes",      "1000"),
            ("Max Steps",   "max_steps",     "200"),
        ]
        self._hp_vars: dict[str, tk.StringVar] = {}
        for label, key, default in params:
            row_f = ttk.Frame(frame)
            row_f.pack(fill="x", pady=1)
            ttk.Label(row_f, text=label, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self._hp_vars[key] = var
            ttk.Entry(row_f, textvariable=var, width=8).pack(side="left")
        ttk.Button(frame, text="Apply", command=self._apply).pack(fill="x", pady=(4, 0))

    # ------------------------------------------------------------------
    # Speed slider
    # ------------------------------------------------------------------

    def _build_speed_slider(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Animation Speed", padding=(6, 4))
        frame.pack(fill="x", padx=4, pady=(4, 6))
        self._speed_var = tk.IntVar(value=0)
        s = ttk.Scale(frame, from_=0, to=500, orient="horizontal",
                      variable=self._speed_var, command=self._on_speed)
        s.pack(fill="x")
        add_tooltip(s, _TIP["speed"])

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _toggle_edit_mode(self) -> None:
        if not self._canvas:
            return
        new_state = not self._canvas.is_edit_mode()
        self._canvas.set_edit_mode(new_state)
        self._edit_mode_btn.configure(
            text="✓ Edit Mode: ON" if new_state else "✗ Edit Mode: OFF"
        )
        if hasattr(self, "_palette_frame"):
            state = "normal" if new_state else "disabled"
            for widget in self._palette_frame.winfo_children():
                if isinstance(widget, (tk.Radiobutton, ttk.Radiobutton)):
                    widget.configure(state=state)

    def _update_preview(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if not self._canvas or not hasattr(self._canvas, "_selected_var"):
            return
        try:
            type_name = self._canvas._selected_var.get()
            cell_type = CellType[type_name]
            self._preview_canvas.configure(bg=_COLORS.get(cell_type, "#FFFFFF"))
            self._selected_type_label.configure(text=type_name)
        except (KeyError, ValueError):
            pass

    def _save(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".npy",
                                         filetypes=[("NumPy", "*.npy")])
        if p:
            try:
                self._sdk.save_q_table(p)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Save Error", str(exc))

    def _load(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("NumPy", "*.npy")])
        if p:
            try:
                self._sdk.load_q_table(p)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Load Error", str(exc))

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
