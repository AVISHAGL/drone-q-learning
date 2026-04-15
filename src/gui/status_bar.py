"""StatusBar — bottom bar with mode display and keyboard hotkeys."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

__all__ = ["StatusBar"]

_HINTS = (
    "[SPACE] Start/Pause  [F] Fast  [H] Heatmap  [A] Arrows"
    "  [E] Editor  [S] Save  [L] Load  [R] Reset"
)
_IDLE, _TRAINING, _PAUSED = "Idle", "Training", "Paused"


class StatusBar(ttk.Frame):
    """Horizontal bottom bar: current mode label + keyboard shortcut hints.

    Binds hotkeys to the root window; all logic delegated to SDK / canvas.

    Args:
        master: Root Tk window (hotkeys bound here).
        sdk:    DroneSimSDK instance.
        canvas: GridCanvas instance.
    """

    def __init__(self, master: tk.Tk, sdk, canvas, **kw) -> None:
        super().__init__(master, **kw)
        self._sdk = sdk
        self._canvas = canvas
        self._state = _IDLE
        self._saved_delay: int = 0
        self._fast: bool = False
        self._build_ui()
        self._bind_keys(master)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        row = ttk.Frame(self)
        row.pack(fill="x", padx=6, pady=3)
        self._mode_var = tk.StringVar(value=f"Mode: {self._state}")
        ttk.Label(row, textvariable=self._mode_var, width=18, anchor="w").pack(side="left")
        ttk.Separator(row, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(row, text=_HINTS, anchor="w").pack(side="left")

    def _set_state(self, state: str) -> None:
        self._state = state
        self._mode_var.set(f"Mode: {state}")

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self, root: tk.Tk) -> None:
        keys = ("<space>", "<f>", "<F>", "<h>", "<H>",
                "<a>", "<A>", "<e>", "<E>", "<s>", "<S>",
                "<l>", "<L>", "<r>", "<R>")
        for key in keys:
            root.bind(key, self._on_key, add="+")

    def _is_typing(self) -> bool:
        """Return True when an Entry has focus (suppress hotkeys)."""
        try:
            w = self.winfo_toplevel().focus_get()
            return isinstance(w, (tk.Entry, ttk.Entry))
        except Exception:  # noqa: BLE001
            return False

    def _on_key(self, event: tk.Event) -> None:
        if self._is_typing():
            return
        k = event.keysym.lower()
        if k == "space":
            self._toggle_training()
        elif k == "f":
            self._toggle_fast()
        elif k == "h":
            self._canvas.toggle_heatmap()
        elif k == "a":
            self._canvas.toggle_arrows()
        elif k == "e":
            self._canvas.set_edit_mode(not self._canvas.is_edit_mode())
        elif k == "s":
            self._save_brain()
        elif k == "l":
            self._load_brain()
        elif k == "r":
            self._hard_reset()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _toggle_training(self) -> None:
        if self._state == _IDLE:
            self._sdk.start_training()
            self._set_state(_TRAINING)
        elif self._state == _TRAINING:
            self._sdk.pause()
            self._set_state(_PAUSED)
        else:
            self._sdk.resume()
            self._set_state(_TRAINING)

    _DEFAULT_DELAY_MS = 100  # fallback when saved delay was 0

    def _toggle_fast(self) -> None:
        if not self._fast:
            self._saved_delay = self._sdk.get_vis_delay()
            self._sdk.set_vis_delay(0)
            self._fast = True
        else:
            restore = self._saved_delay if self._saved_delay > 0 else self._DEFAULT_DELAY_MS
            self._sdk.set_vis_delay(restore)
            self._fast = False

    def _save_brain(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".npy", filetypes=[("NumPy", "*.npy")]
        )
        if path:
            try:
                self._sdk.save_q_table(path)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Save Error", str(exc))

    def _load_brain(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NumPy", "*.npy")])
        if path:
            try:
                self._sdk.load_q_table(path)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Load Error", str(exc))

    def _hard_reset(self) -> None:
        self._sdk.reset()
        self._set_state(_IDLE)
        self._canvas.refresh(policy=self._sdk.get_policy())
