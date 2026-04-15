"""Tests for src.core.version — version string declaration."""

import re

from src.core.version import __version__


def test_version_module_has_version_attribute() -> None:
    """__version__ attribute exists in the version module."""
    import src.core.version as v

    assert hasattr(v, "__version__")


def test_version_is_string() -> None:
    """__version__ is a string."""
    assert isinstance(__version__, str)


def test_version_matches_pyproject() -> None:
    """__version__ equals the expected release value."""
    assert __version__ == "1.00"


def test_version_format_x_dot_xx() -> None:
    """__version__ matches the pattern <digits>.<two digits>."""
    assert re.match(r"^\d+\.\d{2}$", __version__) is not None


def test_version_importable_from_core_init() -> None:
    """__version__ can be imported from the core package init."""
    from src.core import __version__ as v

    assert isinstance(v, str)
