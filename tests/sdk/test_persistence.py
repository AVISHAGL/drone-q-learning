"""Tests for src.sdk.persistence — Q-Table and grid save/load."""

from pathlib import Path

import numpy as np
import pytest

from src.core.cell_type import CellType
from src.sdk.persistence import load_grid, load_q_table, save_grid, save_q_table


def _sample_q(shape: tuple = (10, 4)) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(shape)


def _sample_grid() -> list[list[CellType]]:
    return [
        [CellType.START, CellType.EMPTY, CellType.WIND],
        [CellType.BUILDING, CellType.TRAP, CellType.EMPTY],
        [CellType.EMPTY, CellType.EMPTY, CellType.TARGET],
    ]


# ---------------------------------------------------------------------------
# Q-Table Save / Load
# ---------------------------------------------------------------------------


def test_save_q_table_creates_file(tmp_path: Path) -> None:
    """File exists after save_q_table."""
    p = tmp_path / "q.npy"
    save_q_table(_sample_q(), p)
    assert p.exists()


def test_save_q_table_file_has_npy_extension(tmp_path: Path) -> None:
    """Saved file ends in .npy."""
    p = tmp_path / "q.npy"
    save_q_table(_sample_q(), p)
    assert p.suffix == ".npy"


def test_load_q_table_returns_ndarray(tmp_path: Path) -> None:
    """load_q_table returns np.ndarray."""
    p = tmp_path / "q.npy"
    save_q_table(_sample_q(), p)
    assert isinstance(load_q_table(p), np.ndarray)


def test_save_load_q_table_roundtrip_shape(tmp_path: Path) -> None:
    """Loaded array has same shape as saved."""
    q = _sample_q()
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert load_q_table(p).shape == q.shape


def test_save_load_q_table_roundtrip_values(tmp_path: Path) -> None:
    """np.allclose(original, loaded) is True."""
    q = _sample_q()
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert np.allclose(load_q_table(p), q)


def test_save_load_q_table_roundtrip_10x4(tmp_path: Path) -> None:
    """Shape (10, 4) round-trips correctly."""
    q = _sample_q((10, 4))
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert load_q_table(p).shape == (10, 4)


def test_save_load_q_table_roundtrip_100x4(tmp_path: Path) -> None:
    """Shape (100, 4) round-trips correctly."""
    q = _sample_q((100, 4))
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert load_q_table(p).shape == (100, 4)


def test_save_load_q_table_nonzero_values(tmp_path: Path) -> None:
    """Non-zero values are preserved."""
    q = _sample_q()
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert np.allclose(load_q_table(p), q)


def test_save_load_q_table_negative_values(tmp_path: Path) -> None:
    """Negative values are preserved."""
    q = -_sample_q()
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert np.allclose(load_q_table(p), q)


def test_load_q_table_nonexistent_path_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """FileNotFoundError on bad path."""
    with pytest.raises(FileNotFoundError):
        load_q_table(tmp_path / "missing.npy")


def test_save_q_table_overwrites_existing_file(tmp_path: Path) -> None:
    """Saving twice to same path overwrites cleanly."""
    p = tmp_path / "q.npy"
    save_q_table(_sample_q(), p)
    q2 = _sample_q((5, 4))
    save_q_table(q2, p)
    assert load_q_table(p).shape == (5, 4)


def test_save_q_table_all_zeros_table(tmp_path: Path) -> None:
    """All-zero table round-trips correctly."""
    q = np.zeros((10, 4))
    p = tmp_path / "q.npy"
    save_q_table(q, p)
    assert np.allclose(load_q_table(p), q)


# ---------------------------------------------------------------------------
# Grid Save / Load
# ---------------------------------------------------------------------------


def test_save_grid_creates_file(tmp_path: Path) -> None:
    """File exists after save_grid."""
    p = tmp_path / "grid.json"
    save_grid(_sample_grid(), p)
    assert p.exists()


def test_save_grid_produces_valid_json(tmp_path: Path) -> None:
    """Saved file is parseable by json.loads."""
    import json

    p = tmp_path / "grid.json"
    save_grid(_sample_grid(), p)
    data = json.loads(p.read_text())
    assert isinstance(data, list)


def test_load_grid_returns_list_of_lists(tmp_path: Path) -> None:
    """load_grid returns list[list[...]]."""
    p = tmp_path / "grid.json"
    save_grid(_sample_grid(), p)
    g = load_grid(p)
    assert isinstance(g, list)
    assert all(isinstance(row, list) for row in g)


def test_load_grid_elements_are_cell_types(tmp_path: Path) -> None:
    """Every element is a CellType."""
    p = tmp_path / "grid.json"
    save_grid(_sample_grid(), p)
    for row in load_grid(p):
        for cell in row:
            assert isinstance(cell, CellType)


def test_save_load_grid_roundtrip_empty_cell(tmp_path: Path) -> None:
    """CellType.EMPTY round-trips."""
    g = [[CellType.EMPTY, CellType.TARGET], [CellType.START, CellType.EMPTY]]
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert load_grid(p)[0][0] is CellType.EMPTY


def test_save_load_grid_roundtrip_building_cell(tmp_path: Path) -> None:
    """CellType.BUILDING round-trips."""
    g = [[CellType.BUILDING, CellType.TARGET], [CellType.START, CellType.EMPTY]]
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert load_grid(p)[0][0] is CellType.BUILDING


def test_save_load_grid_roundtrip_trap_cell(tmp_path: Path) -> None:
    """CellType.TRAP round-trips."""
    g = [[CellType.TRAP, CellType.TARGET], [CellType.START, CellType.EMPTY]]
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert load_grid(p)[0][0] is CellType.TRAP


def test_save_load_grid_roundtrip_wind_cell(tmp_path: Path) -> None:
    """CellType.WIND round-trips."""
    g = [[CellType.WIND, CellType.TARGET], [CellType.START, CellType.EMPTY]]
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert load_grid(p)[0][0] is CellType.WIND


def test_save_load_grid_roundtrip_target_cell(tmp_path: Path) -> None:
    """CellType.TARGET round-trips."""
    g = [[CellType.TARGET, CellType.EMPTY], [CellType.START, CellType.EMPTY]]
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert load_grid(p)[0][0] is CellType.TARGET


def test_save_load_grid_roundtrip_start_cell(tmp_path: Path) -> None:
    """CellType.START round-trips."""
    g = [[CellType.START, CellType.EMPTY], [CellType.EMPTY, CellType.TARGET]]
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert load_grid(p)[0][0] is CellType.START


def test_save_load_grid_roundtrip_full_3x3_grid(tmp_path: Path) -> None:
    """Full 3×3 grid with mixed types round-trips."""
    g = _sample_grid()
    p = tmp_path / "grid.json"
    save_grid(g, p)
    loaded = load_grid(p)
    for r in range(3):
        for c in range(3):
            assert loaded[r][c] is g[r][c]


def test_save_load_grid_row_count_preserved(tmp_path: Path) -> None:
    """Row count unchanged after round-trip."""
    g = _sample_grid()
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert len(load_grid(p)) == 3


def test_save_load_grid_col_count_preserved(tmp_path: Path) -> None:
    """Column count unchanged after round-trip."""
    g = _sample_grid()
    p = tmp_path / "grid.json"
    save_grid(g, p)
    assert all(len(row) == 3 for row in load_grid(p))


def test_load_grid_nonexistent_path_raises_file_not_found(tmp_path: Path) -> None:
    """FileNotFoundError on bad path."""
    with pytest.raises(FileNotFoundError):
        load_grid(tmp_path / "missing.json")
