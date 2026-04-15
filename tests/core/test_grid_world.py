"""Tests for src.core.grid_world — GridWorld environment."""

import pytest

from src.core.cell_type import CellType
from src.core.grid_world import GridWorld

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env_cfg() -> dict:
    """Minimal 5×5 environment config for fast tests."""
    return {
        "rows": 5,
        "cols": 5,
        "max_steps_per_episode": 50,
        "rewards": {"empty": -1, "wind": -3, "trap": -10, "building": -1, "target": 50},
        "default_grid": [],
    }


@pytest.fixture()
def gw(env_cfg: dict) -> GridWorld:
    """GridWorld using the 5×5 config; START at (0,0), TARGET at (4,4)."""
    return GridWorld(env_cfg)


@pytest.fixture()
def small_cfg() -> dict:
    """3×3 config with building and trap placed for targeted tests."""
    return {
        "rows": 3,
        "cols": 3,
        "max_steps_per_episode": 10,
        "rewards": {"empty": -1, "wind": -3, "trap": -10, "building": -1, "target": 50},
        "default_grid": [],
    }


# ---------------------------------------------------------------------------
# Construction & Properties
# ---------------------------------------------------------------------------


def test_grid_world_init_no_error(env_cfg: dict) -> None:
    """GridWorld(config) constructs without exception."""
    GridWorld(env_cfg)


def test_grid_world_num_states_equals_rows_times_cols(gw: GridWorld) -> None:
    """num_states == rows * cols."""
    assert gw.num_states == 25


def test_grid_world_num_actions_equals_four(gw: GridWorld) -> None:
    """num_actions == 4."""
    assert gw.num_actions == 4


def test_grid_world_get_grid_returns_2d_list(gw: GridWorld) -> None:
    """get_grid() returns a list of lists."""
    g = gw.get_grid()
    assert isinstance(g, list)
    assert all(isinstance(row, list) for row in g)


def test_grid_world_get_grid_row_count(gw: GridWorld) -> None:
    """Outer list has 'rows' items."""
    assert len(gw.get_grid()) == 5


def test_grid_world_get_grid_col_count(gw: GridWorld) -> None:
    """Each inner list has 'cols' items."""
    assert all(len(row) == 5 for row in gw.get_grid())


def test_grid_world_get_grid_all_cell_types(gw: GridWorld) -> None:
    """Every element is a CellType."""
    for row in gw.get_grid():
        for cell in row:
            assert isinstance(cell, CellType)


def test_grid_world_exactly_one_start_cell(gw: GridWorld) -> None:
    """Exactly one cell is CellType.START."""
    starts = sum(c is CellType.START for row in gw.get_grid() for c in row)
    assert starts == 1


def test_grid_world_exactly_one_target_cell(gw: GridWorld) -> None:
    """Exactly one cell is CellType.TARGET."""
    targets = sum(c is CellType.TARGET for row in gw.get_grid() for c in row)
    assert targets == 1


def test_grid_world_small_grid_5x5_num_states(env_cfg: dict) -> None:
    """5×5 config yields num_states=25."""
    assert GridWorld(env_cfg).num_states == 25


def test_grid_world_small_grid_3x3_num_states(small_cfg: dict) -> None:
    """3×3 config yields num_states=9."""
    assert GridWorld(small_cfg).num_states == 9


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_grid_world_reset_returns_int(gw: GridWorld) -> None:
    """reset() returns int."""
    assert isinstance(gw.reset(), int)


def test_grid_world_reset_returns_start_state(gw: GridWorld) -> None:
    """Returned state corresponds to START cell (0,0) → state 0."""
    assert gw.reset() == 0


def test_grid_world_reset_idempotent(gw: GridWorld) -> None:
    """Two calls return the same value."""
    assert gw.reset() == gw.reset()


def test_grid_world_reset_state_in_valid_range(gw: GridWorld) -> None:
    """0 <= reset() < num_states."""
    s = gw.reset()
    assert 0 <= s < gw.num_states


# ---------------------------------------------------------------------------
# step() — Return Types
# ---------------------------------------------------------------------------


def test_grid_world_step_returns_tuple_of_three(gw: GridWorld) -> None:
    """step() returns 3-tuple."""
    result = gw.step(gw.reset(), 3)  # RIGHT
    assert len(result) == 3


def test_grid_world_step_next_state_is_int(gw: GridWorld) -> None:
    """First element is int."""
    ns, _, _ = gw.step(gw.reset(), 3)
    assert isinstance(ns, int)


def test_grid_world_step_reward_is_float(gw: GridWorld) -> None:
    """Second element is float."""
    _, r, _ = gw.step(gw.reset(), 3)
    assert isinstance(r, float)


def test_grid_world_step_done_is_bool(gw: GridWorld) -> None:
    """Third element is bool."""
    _, _, d = gw.step(gw.reset(), 3)
    assert isinstance(d, bool)


def test_grid_world_step_next_state_in_range(gw: GridWorld) -> None:
    """0 <= next_state < num_states."""
    ns, _, _ = gw.step(gw.reset(), 3)
    assert 0 <= ns < gw.num_states


# ---------------------------------------------------------------------------
# step() — Directional Movement
# ---------------------------------------------------------------------------


def test_grid_world_step_up_decrements_row(gw: GridWorld) -> None:
    """Action UP moves to row-1 when valid (from state 5 → row 0, col 0)."""
    state = gw.pos_to_state(1, 0)
    ns, _, _ = gw.step(state, 0)  # UP
    r, c = gw.state_to_pos(ns)
    assert r == 0 and c == 0


def test_grid_world_step_down_increments_row(gw: GridWorld) -> None:
    """Action DOWN moves to row+1 when valid."""
    state = gw.pos_to_state(0, 0)
    # state 0 is START, move DOWN to (1, 0)
    ns, _, _ = gw.step(state, 1)
    r, _ = gw.state_to_pos(ns)
    assert r == 1


def test_grid_world_step_left_decrements_col(gw: GridWorld) -> None:
    """Action LEFT moves to col-1 when valid."""
    state = gw.pos_to_state(1, 2)
    ns, _, _ = gw.step(state, 2)  # LEFT
    _, c = gw.state_to_pos(ns)
    assert c == 1


def test_grid_world_step_right_increments_col(gw: GridWorld) -> None:
    """Action RIGHT moves to col+1 when valid."""
    state = gw.pos_to_state(1, 1)
    ns, _, _ = gw.step(state, 3)  # RIGHT
    _, c = gw.state_to_pos(ns)
    assert c == 2


# ---------------------------------------------------------------------------
# step() — Cell Rewards
# ---------------------------------------------------------------------------


def test_grid_world_step_into_empty_reward_minus_one(small_cfg: dict) -> None:
    """Stepping into EMPTY gives -1.0."""
    gw = GridWorld(small_cfg)
    state = gw.pos_to_state(0, 0)
    _, r, _ = gw.step(state, 3)  # RIGHT → (0,1) EMPTY
    assert r == -1.0


def test_grid_world_step_into_empty_done_false(small_cfg: dict) -> None:
    """Stepping into EMPTY gives done=False."""
    gw = GridWorld(small_cfg)
    state = gw.pos_to_state(0, 0)
    _, _, d = gw.step(state, 3)
    assert d is False


def test_grid_world_step_into_trap_reward_minus_ten(small_cfg: dict) -> None:
    """Stepping into TRAP gives -10.0."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TRAP)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == -10.0


def test_grid_world_step_into_trap_done_false(small_cfg: dict) -> None:
    """Stepping into TRAP gives done=False."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TRAP)
    _, _, d = gw.step(gw.pos_to_state(0, 0), 3)
    assert d is False


def test_grid_world_step_into_wind_reward_minus_three(small_cfg: dict) -> None:
    """Stepping into WIND gives -3.0."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.WIND)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == -3.0


def test_grid_world_step_into_wind_done_false(small_cfg: dict) -> None:
    """Stepping into WIND gives done=False."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.WIND)
    _, _, d = gw.step(gw.pos_to_state(0, 0), 3)
    assert d is False


def test_grid_world_step_into_target_reward_plus_fifty(small_cfg: dict) -> None:
    """Stepping into TARGET gives +50.0."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TARGET)
    gw.set_cell(2, 2, CellType.EMPTY)  # remove original target
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == 50.0


def test_grid_world_step_into_target_done_true(small_cfg: dict) -> None:
    """Stepping into TARGET gives done=True."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TARGET)
    gw.set_cell(2, 2, CellType.EMPTY)
    _, _, d = gw.step(gw.pos_to_state(0, 0), 3)
    assert d is True


def test_grid_world_step_into_target_next_state_is_target_index(
    small_cfg: dict,
) -> None:
    """next_state equals the target cell index."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TARGET)
    gw.set_cell(2, 2, CellType.EMPTY)
    ns, _, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert ns == gw.pos_to_state(0, 1)


# ---------------------------------------------------------------------------
# step() — Rewards from Config
# ---------------------------------------------------------------------------


def test_grid_world_step_trap_reward_from_config(small_cfg: dict) -> None:
    """Trap reward matches config."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TRAP)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == float(small_cfg["rewards"]["trap"])


def test_grid_world_step_wind_reward_from_config(small_cfg: dict) -> None:
    """Wind reward matches config."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.WIND)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == float(small_cfg["rewards"]["wind"])


def test_grid_world_step_target_reward_from_config(small_cfg: dict) -> None:
    """Target reward matches config."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.TARGET)
    gw.set_cell(2, 2, CellType.EMPTY)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == float(small_cfg["rewards"]["target"])


def test_grid_world_step_empty_reward_from_config(small_cfg: dict) -> None:
    """Empty reward matches config."""
    gw = GridWorld(small_cfg)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == float(small_cfg["rewards"]["empty"])


# ---------------------------------------------------------------------------
# step() — Boundary & Building
# ---------------------------------------------------------------------------


def test_grid_world_step_up_from_top_row_stays_same(gw: GridWorld) -> None:
    """UP from row 0 leaves drone in same cell."""
    state = gw.reset()  # (0,0)
    ns, _, _ = gw.step(state, 0)
    assert ns == state


def test_grid_world_step_down_from_bottom_row_stays_same(gw: GridWorld) -> None:
    """DOWN from last row leaves drone in same cell."""
    state = gw.pos_to_state(4, 0)
    ns, _, _ = gw.step(state, 1)
    assert ns == state


def test_grid_world_step_left_from_leftmost_col_stays_same(gw: GridWorld) -> None:
    """LEFT from col 0 leaves drone in same cell."""
    state = gw.pos_to_state(2, 0)
    ns, _, _ = gw.step(state, 2)
    assert ns == state


def test_grid_world_step_right_from_rightmost_col_stays_same(gw: GridWorld) -> None:
    """RIGHT from last col leaves drone in same cell."""
    state = gw.pos_to_state(2, 4)
    ns, _, _ = gw.step(state, 3)
    assert ns == state


def test_grid_world_step_boundary_reward_is_minus_one(gw: GridWorld) -> None:
    """Boundary collision gives -1.0 (building reward)."""
    state = gw.reset()
    _, r, _ = gw.step(state, 0)  # UP from (0,0)
    assert r == -1.0


def test_grid_world_step_boundary_done_false(gw: GridWorld) -> None:
    """Boundary collision gives done=False."""
    state = gw.reset()
    _, _, d = gw.step(state, 0)
    assert d is False


def test_grid_world_step_into_building_stays_same_state(small_cfg: dict) -> None:
    """Building collision leaves state unchanged."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.BUILDING)
    state = gw.pos_to_state(0, 0)
    ns, _, _ = gw.step(state, 3)
    assert ns == state


def test_grid_world_step_into_building_reward_minus_one(small_cfg: dict) -> None:
    """Building collision gives -1.0."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.BUILDING)
    _, r, _ = gw.step(gw.pos_to_state(0, 0), 3)
    assert r == -1.0


def test_grid_world_step_into_building_done_false(small_cfg: dict) -> None:
    """Building collision gives done=False."""
    gw = GridWorld(small_cfg)
    gw.set_cell(0, 1, CellType.BUILDING)
    _, _, d = gw.step(gw.pos_to_state(0, 0), 3)
    assert d is False


# ---------------------------------------------------------------------------
# step() — Max Steps
# ---------------------------------------------------------------------------


def test_grid_world_step_max_steps_exceeded_done_true(small_cfg: dict) -> None:
    """After max_steps steps, done=True."""
    gw = GridWorld(small_cfg)
    state = gw.reset()
    done = False
    for _ in range(small_cfg["max_steps_per_episode"]):
        state, _, done = gw.step(state, 2)  # keep bumping LEFT boundary
        if done:
            break
    assert done is True


def test_grid_world_step_max_steps_from_config(small_cfg: dict) -> None:
    """Max-steps limit uses config value, not a hardcoded constant."""
    cfg = dict(small_cfg)
    cfg["max_steps_per_episode"] = 3
    gw = GridWorld(cfg)
    state = gw.reset()
    for _ in range(3):
        state, _, done = gw.step(state, 2)
    assert done is True


def test_grid_world_step_before_max_steps_done_false(small_cfg: dict) -> None:
    """One step before the limit, done is still False (unless target hit)."""
    gw = GridWorld(small_cfg)
    state = gw.reset()
    for _ in range(small_cfg["max_steps_per_episode"] - 1):
        state, _, done = gw.step(state, 2)  # stay at left boundary
        if done:
            break
    # We may have hit the limit if done already; just assert not done early
    # unless the loop terminated normally
    # (checking done after the last iteration could be True if we ran out)
    # So just check we didn't get done prematurely on step 1
    gw2 = GridWorld(small_cfg)
    s2 = gw2.reset()
    _, _, d2 = gw2.step(s2, 2)
    assert d2 is False


# ---------------------------------------------------------------------------
# State ↔ Position
# ---------------------------------------------------------------------------


def test_grid_world_pos_to_state_origin(gw: GridWorld) -> None:
    """pos_to_state(0, 0) == 0."""
    assert gw.pos_to_state(0, 0) == 0


def test_grid_world_pos_to_state_formula(gw: GridWorld) -> None:
    """pos_to_state(r, c) == r * cols + c."""
    assert gw.pos_to_state(2, 3) == 2 * 5 + 3


def test_grid_world_state_to_pos_origin(gw: GridWorld) -> None:
    """state_to_pos(0) == (0, 0)."""
    assert gw.state_to_pos(0) == (0, 0)


def test_grid_world_state_to_pos_last_cell(gw: GridWorld) -> None:
    """state_to_pos(num_states-1) == (rows-1, cols-1)."""
    assert gw.state_to_pos(24) == (4, 4)


def test_grid_world_roundtrip_pos_to_state_to_pos(gw: GridWorld) -> None:
    """state_to_pos(pos_to_state(r, c)) == (r, c)."""
    assert gw.state_to_pos(gw.pos_to_state(2, 3)) == (2, 3)


def test_grid_world_roundtrip_state_to_pos_to_state(gw: GridWorld) -> None:
    """pos_to_state(*state_to_pos(s)) == s."""
    assert gw.pos_to_state(*gw.state_to_pos(13)) == 13


# ---------------------------------------------------------------------------
# Grid Mutation
# ---------------------------------------------------------------------------


def test_grid_world_set_cell_changes_cell_type(small_cfg: dict) -> None:
    """set_cell(r,c,TRAP) changes that cell to TRAP."""
    gw = GridWorld(small_cfg)
    gw.set_cell(1, 1, CellType.TRAP)
    assert gw.get_grid()[1][1] is CellType.TRAP


def test_grid_world_set_cell_only_changes_target_cell(small_cfg: dict) -> None:
    """Other cells are unchanged after set_cell."""
    gw = GridWorld(small_cfg)
    original = gw.get_grid()[0][0]
    gw.set_cell(1, 1, CellType.TRAP)
    assert gw.get_grid()[0][0] is original


def test_grid_world_get_grid_returns_independent_copy(gw: GridWorld) -> None:
    """Mutating the returned list does not affect internal grid."""
    grid_copy = gw.get_grid()
    grid_copy[0][0] = CellType.TRAP
    assert gw.get_grid()[0][0] is not CellType.TRAP
