"""Entry point for the Drone Q-Learning Simulation desktop application."""

import contextlib


def main() -> None:
    """Launch the application."""
    from src.gui.app import App

    app = App()
    app.mainloop()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
