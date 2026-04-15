"""Utility functions for loading JSON configuration files."""

import json
from pathlib import Path

__all__ = ["load_env_config", "load_rl_config"]

_DEFAULT_ENV = Path(__file__).parent.parent.parent / "config" / "env.json"
_DEFAULT_RL = Path(__file__).parent.parent.parent / "config" / "rl.json"


def load_env_config(path: str | Path | None = None) -> dict:
    """Load the environment configuration from a JSON file.

    Args:
        path: Path to the JSON file. Defaults to config/env.json relative
              to the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    resolved = Path(path) if path is not None else _DEFAULT_ENV
    with resolved.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_rl_config(path: str | Path | None = None) -> dict:
    """Load the reinforcement-learning configuration from a JSON file.

    Args:
        path: Path to the JSON file. Defaults to config/rl.json relative
              to the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    resolved = Path(path) if path is not None else _DEFAULT_RL
    with resolved.open(encoding="utf-8") as fh:
        return json.load(fh)
