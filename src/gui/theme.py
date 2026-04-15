"""theme.py — Dark theme: centralized palette + ttk.Style configuration.

Semantic grid-cell colors live in grid_canvas.py and are intentionally
excluded here — they are domain values, not UI chrome.

Usage
-----
Call ``configure_style()`` once, before any widgets are created, from
``App.__init__``.  Then set the root window and any plain-tk containers
(PanedWindow, Canvas) to ``BG`` explicitly, since ttk.Style cannot reach them.

Adjusting the theme
-------------------
All palette constants are at the top of this file.  Change a constant and
every widget that references it (via configure_style) picks it up on the
next application start.  No other file needs to be touched for colour
changes.

  BG          — primary background applied to the root window and frames
  BG_PANEL    — slightly lighter surface used inside LabelFrames
  BG_INPUT    — entry / text-field background
  BG_BTN      — button resting background
  BG_BTN_HOV  — button background on hover / active state
  FG          — primary text colour
  FG_DIM      — muted text (disabled widgets, secondary labels)
  ACCENT      — highlight colour used on pressed buttons and focus rings
  BORDER      — subtle border / trough colour
  SUCCESS     — dashboard "reached target" label
  FAIL        — dashboard "failed" label
  NEUTRAL     — dashboard reset / default label colour (= FG)
  TOOLTIP_BG  — tooltip popup background
  TOOLTIP_FG  — tooltip popup foreground
  PLOT_LINE1  — matplotlib reward-curve line colour
  PLOT_LINE2  — matplotlib ΔQ-curve line colour
  PLOT_GRID   — matplotlib axis grid / spine colour
"""

from tkinter import ttk

__all__ = [
    "configure_style",
    "apply_to_figure",
    # Palette constants consumed by other modules
    "BG",
    "BG_PANEL",
    "FG",
    "FG_DIM",
    "ACCENT",
    "SUCCESS",
    "FAIL",
    "NEUTRAL",
    "TOOLTIP_BG",
    "TOOLTIP_FG",
    "PLOT_LINE1",
    "PLOT_LINE2",
    "PLOT_GRID",
]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BG = "#1e1e2f"           # Root / primary background
BG_PANEL = "#252538"     # LabelFrame / panel surface
BG_INPUT = "#2d2d44"     # Entry field background
BG_BTN = "#35354f"       # Button resting background
BG_BTN_HOV = "#4a4a72"  # Button hover / active background

FG = "#dcdcf0"           # Primary text
FG_DIM = "#7878a0"       # Disabled / secondary text

ACCENT = "#7c6af7"       # Focus ring, pressed button, sash highlight
BORDER = "#3a3a5c"       # Borders, troughs, separators

SUCCESS = "#4ade80"      # Dashboard "✓ Success"
FAIL = "#f87171"         # Dashboard "✗ Fail"
NEUTRAL = FG             # Dashboard neutral (reset) — same as primary text

TOOLTIP_BG = "#2d2d44"   # Tooltip popup background
TOOLTIP_FG = FG          # Tooltip popup text

PLOT_LINE1 = "#7c6af7"   # Reward curve (accent purple)
PLOT_LINE2 = "#f59e0b"   # ΔQ curve (amber)
PLOT_GRID = "#3a3a5c"    # Axis spines, tick marks


# ---------------------------------------------------------------------------
# ttk.Style configuration
# ---------------------------------------------------------------------------

def configure_style() -> None:
    """Configure the global ttk.Style with the dark palette.

    Must be called after ``tk.Tk()`` is created and before any widgets.
    """
    s = ttk.Style()
    s.theme_use("clam")  # clam exposes the most configurable style hooks

    # --- Frame / container ---------------------------------------------------
    s.configure("TFrame", background=BG)
    s.configure("TLabelframe", background=BG_PANEL, bordercolor=BORDER,
                relief="groove")
    s.configure("TLabelframe.Label", background=BG_PANEL, foreground=FG,
                font=("TkDefaultFont", 9, "bold"))

    # --- Labels --------------------------------------------------------------
    s.configure("TLabel", background=BG, foreground=FG)

    # --- Buttons -------------------------------------------------------------
    s.configure(
        "TButton",
        background=BG_BTN,
        foreground=FG,
        bordercolor=BORDER,
        darkcolor=BG_BTN,
        lightcolor=BG_BTN,
        relief="flat",
        padding=(4, 2),
        focuscolor=ACCENT,
    )
    s.map(
        "TButton",
        background=[("pressed", ACCENT), ("active", BG_BTN_HOV), ("disabled", BG)],
        foreground=[("pressed", "#ffffff"), ("active", "#ffffff"), ("disabled", FG_DIM)],
        bordercolor=[("active", ACCENT), ("pressed", ACCENT)],
        darkcolor=[("active", BG_BTN_HOV), ("pressed", ACCENT)],
        lightcolor=[("active", BG_BTN_HOV), ("pressed", ACCENT)],
        relief=[("pressed", "sunken")],
    )

    # --- Entry ---------------------------------------------------------------
    s.configure(
        "TEntry",
        fieldbackground=BG_INPUT,
        foreground=FG,
        insertcolor=FG,        # text cursor colour
        bordercolor=BORDER,
        lightcolor=BG_INPUT,
        darkcolor=BG_INPUT,
        padding=(4, 3),
    )
    s.map(
        "TEntry",
        bordercolor=[("focus", ACCENT)],
        lightcolor=[("focus", ACCENT)],
    )

    # --- Radiobutton ---------------------------------------------------------
    s.configure(
        "TRadiobutton",
        background=BG_PANEL,
        foreground=FG,
        indicatorcolor=BG_INPUT,
        focuscolor=ACCENT,
    )
    s.map(
        "TRadiobutton",
        background=[("active", BG_PANEL), ("disabled", BG_PANEL)],
        foreground=[("disabled", FG_DIM)],
        indicatorcolor=[("selected", ACCENT), ("active", BG_BTN_HOV)],
    )

    # --- Checkbutton ---------------------------------------------------------
    s.configure(
        "TCheckbutton",
        background=BG,
        foreground=FG,
        indicatorcolor=BG_INPUT,
        focuscolor=ACCENT,
    )
    s.map(
        "TCheckbutton",
        background=[("active", BG)],
        foreground=[("disabled", FG_DIM)],
        indicatorcolor=[("selected", ACCENT), ("active", BG_BTN_HOV)],
    )

    # --- Scale ---------------------------------------------------------------
    s.configure(
        "TScale",
        background=BG,
        troughcolor=BG_INPUT,
        bordercolor=BORDER,
        darkcolor=BG_INPUT,
        lightcolor=BG_INPUT,
        sliderrelief="flat",
    )
    s.map(
        "TScale",
        background=[("active", BG)],
        troughcolor=[("active", BG_INPUT)],
    )

    # --- Scrollbar (used internally by some widgets) -------------------------
    s.configure(
        "TScrollbar",
        background=BG_BTN,
        troughcolor=BG_INPUT,
        bordercolor=BORDER,
        arrowcolor=FG,
        darkcolor=BG_BTN,
        lightcolor=BG_BTN,
    )
    s.map("TScrollbar", background=[("active", BG_BTN_HOV)])

    # --- Separator -----------------------------------------------------------
    s.configure("TSeparator", background=BORDER)

    # --- Notebook (tabbed panels) --------------------------------------------
    s.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(2, 2, 0, 0))
    s.configure(
        "TNotebook.Tab",
        background=BG_BTN,
        foreground=FG_DIM,
        padding=(10, 4),
        focuscolor=ACCENT,
    )
    s.map(
        "TNotebook.Tab",
        background=[("selected", BG_PANEL), ("active", BG_BTN_HOV)],
        foreground=[("selected", FG), ("active", FG)],
        expand=[("selected", (1, 1, 1, 0))],
    )

    # --- Progressbar (not currently used, but covered for completeness) ------
    s.configure(
        "TProgressbar",
        troughcolor=BG_INPUT,
        background=ACCENT,
        bordercolor=BORDER,
    )


# ---------------------------------------------------------------------------
# Matplotlib helper
# ---------------------------------------------------------------------------

def apply_to_figure(fig, axes: list) -> None:  # type: ignore[type-arg]
    """Apply dark palette to a matplotlib Figure and its Axes list.

    Args:
        fig:  matplotlib Figure instance.
        axes: List of Axes objects belonging to *fig*.

    Call once after the Figure is created.  The _redraw() method in
    GraphPanel re-applies axis labels on each draw, so label colours
    are set inside this function and again after each cla() call via
    ``_style_axes()`` in GraphPanel.
    """
    fig.patch.set_facecolor(BG)
    for ax in axes:
        _style_ax(ax)


def _style_ax(ax) -> None:  # type: ignore[type-arg]
    """Apply dark styling to a single matplotlib Axes object."""
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=FG_DIM, labelsize=8)
    ax.xaxis.label.set_color(FG_DIM)
    ax.yaxis.label.set_color(FG_DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_GRID)
