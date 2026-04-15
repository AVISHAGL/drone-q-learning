"""Tests for src.core.cell_type — CellType enum."""

import enum

import pytest

from src.core.cell_type import CellType


def test_cell_type_empty_exists() -> None:
    """CellType.EMPTY is a member of the enum."""
    assert CellType.EMPTY is not None


def test_cell_type_building_exists() -> None:
    """CellType.BUILDING is a member of the enum."""
    assert CellType.BUILDING is not None


def test_cell_type_trap_exists() -> None:
    """CellType.TRAP is a member of the enum."""
    assert CellType.TRAP is not None


def test_cell_type_wind_exists() -> None:
    """CellType.WIND is a member of the enum."""
    assert CellType.WIND is not None


def test_cell_type_target_exists() -> None:
    """CellType.TARGET is a member of the enum."""
    assert CellType.TARGET is not None


def test_cell_type_start_exists() -> None:
    """CellType.START is a member of the enum."""
    assert CellType.START is not None


def test_cell_type_has_six_members() -> None:
    """Enum has exactly 6 members."""
    assert len(CellType) == 6


def test_cell_type_all_values_unique() -> None:
    """All enum values are distinct."""
    values = [m.value for m in CellType]
    assert len(values) == len(set(values))


def test_cell_type_is_enum_subclass() -> None:
    """CellType is a subclass of enum.Enum."""
    assert issubclass(CellType, enum.Enum)


def test_cell_type_iteration_yields_all_members() -> None:
    """Iterating over CellType yields exactly 6 items."""
    assert len(list(CellType)) == 6


def test_cell_type_lookup_by_name_empty() -> None:
    """CellType['EMPTY'] returns CellType.EMPTY."""
    assert CellType["EMPTY"] is CellType.EMPTY


def test_cell_type_lookup_by_name_building() -> None:
    """CellType['BUILDING'] returns CellType.BUILDING."""
    assert CellType["BUILDING"] is CellType.BUILDING


def test_cell_type_lookup_by_name_trap() -> None:
    """CellType['TRAP'] returns CellType.TRAP."""
    assert CellType["TRAP"] is CellType.TRAP


def test_cell_type_lookup_by_name_wind() -> None:
    """CellType['WIND'] returns CellType.WIND."""
    assert CellType["WIND"] is CellType.WIND


def test_cell_type_lookup_by_name_target() -> None:
    """CellType['TARGET'] returns CellType.TARGET."""
    assert CellType["TARGET"] is CellType.TARGET


def test_cell_type_lookup_by_name_start() -> None:
    """CellType['START'] returns CellType.START."""
    assert CellType["START"] is CellType.START


def test_cell_type_invalid_name_raises_key_error() -> None:
    """CellType['INVALID'] raises KeyError."""
    with pytest.raises(KeyError):
        _ = CellType["INVALID"]


def test_cell_type_members_not_equal_to_each_other() -> None:
    """Two different members are not equal."""
    assert CellType.EMPTY != CellType.TRAP


def test_cell_type_same_member_is_equal_to_itself() -> None:
    """A member is equal to itself."""
    assert CellType.TARGET == CellType.TARGET


def test_cell_type_name_attribute_matches_string() -> None:
    """CellType.TRAP.name equals the string 'TRAP'."""
    assert CellType.TRAP.name == "TRAP"
