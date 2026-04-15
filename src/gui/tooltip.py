"""Simple tooltip widget helper for tkinter."""

import tkinter as tk

from src.gui import theme

__all__ = ["add_tooltip"]


def add_tooltip(widget: tk.Widget, text: str) -> None:
    """Attach a hover tooltip to *widget* showing *text*.

    Args:
        widget: Any tkinter widget.
        text: Text to display in the tooltip.
    """
    tip: tk.Toplevel | None = None

    def show(event: tk.Event) -> None:
        nonlocal tip
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 10}")
        tk.Label(tip, text=text, background=theme.TOOLTIP_BG,
                 foreground=theme.TOOLTIP_FG, relief="solid",
                 borderwidth=1, padx=4, pady=2).pack()

    def hide(_event: tk.Event) -> None:
        nonlocal tip
        if tip:
            tip.destroy()
            tip = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)
