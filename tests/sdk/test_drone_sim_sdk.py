"""Tests for src.sdk.drone_sim_sdk — DroneSimSDK façade."""

import queue
from pathlib import Path

import numpy as np
import pytest

from src.core.cell_type import CellType
from src.core.episode_stats import EpisodeStats
from src.sdk.drone_sim_sdk import DroneSimSDK


@pytest.fixture()
def sdk() -> DroneSimSDK:
    """Fresh SDK instance (not training)."""
    return DroneSimSDK()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_sdk_init_no_error() -> None:
    """DroneSimSDK() constructs without error."""
    DroneSimSDK()


def test_sdk_get_stats_queue_returns_queue(sdk: DroneSimSDK) -> None:
    """get_stats_queue() returns queue.Queue."""
    assert isinstance(sdk.get_stats_queue(), queue.Queue)


def test_sdk_get_q_table_returns_ndarray(sdk: DroneSimSDK) -> None:
    """get_q_table() returns np.ndarray."""
    assert isinstance(sdk.get_q_table(), np.ndarray)


def test_sdk_get_q_table_initially_all_zeros(sdk: DroneSimSDK) -> None:
    """Freshly constructed SDK has all-zero Q-Table."""
    assert np.all(sdk.get_q_table() == 0.0)


def test_sdk_get_policy_returns_dict(sdk: DroneSimSDK) -> None:
    """get_policy() returns dict."""
    assert isinstance(sdk.get_policy(), dict)


def test_sdk_get_policy_all_actions_valid(sdk: DroneSimSDK) -> None:
    """All values in policy dict are in {0,1,2,3}."""
    assert all(v in {0, 1, 2, 3} for v in sdk.get_policy().values())


# ---------------------------------------------------------------------------
# Training Lifecycle
# ---------------------------------------------------------------------------


def test_sdk_start_training_no_error(sdk: DroneSimSDK) -> None:
    """start_training() does not raise."""
    sdk.start_training()
    sdk.stop()


def test_sdk_start_training_queue_receives_stats(sdk: DroneSimSDK) -> None:
    """Queue has at least 1 stat within 10s."""
    sdk.start_training()
    item = sdk.get_stats_queue().get(timeout=10.0)
    assert item is not None
    sdk.stop()


def test_sdk_start_training_twice_no_double_thread(sdk: DroneSimSDK) -> None:
    """Second start_training() call does not spawn extra thread."""
    sdk.start_training()
    t1 = sdk._loop._thread if sdk._loop else None
    sdk.start_training()
    t2 = sdk._loop._thread if sdk._loop else None
    assert t1 is t2
    sdk.stop()


def test_sdk_pause_no_error(sdk: DroneSimSDK) -> None:
    """pause() does not raise."""
    sdk.start_training()
    sdk.pause()
    sdk.stop()


def test_sdk_resume_no_error(sdk: DroneSimSDK) -> None:
    """resume() does not raise."""
    sdk.start_training()
    sdk.pause()
    sdk.resume()
    sdk.stop()


def test_sdk_stop_no_error(sdk: DroneSimSDK) -> None:
    """stop() does not raise."""
    sdk.start_training()
    sdk.stop()


def test_sdk_stop_before_start_no_error(sdk: DroneSimSDK) -> None:
    """stop() without prior start_training() is safe."""
    sdk.stop()


def test_sdk_stop_then_start_resumes_training(sdk: DroneSimSDK) -> None:
    """Can call start_training() after stop()."""
    sdk.start_training()
    sdk.stop()
    sdk.start_training()
    item = sdk.get_stats_queue().get(timeout=10.0)
    assert item is not None
    sdk.stop()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_sdk_reset_q_table_to_zeros(sdk: DroneSimSDK) -> None:
    """After reset(), get_q_table() is all zeros."""
    sdk.start_training()
    sdk.get_stats_queue().get(timeout=10.0)
    sdk.stop()
    sdk.reset()
    assert np.all(sdk.get_q_table() == 0.0)


def test_sdk_reset_after_training_clears_table(sdk: DroneSimSDK) -> None:
    """Q-Table was non-zero; reset zeros it."""
    sdk.start_training()
    sdk.get_stats_queue().get(timeout=10.0)
    sdk.stop()
    # Q-Table should have been updated
    sdk.reset()
    assert np.all(sdk.get_q_table() == 0.0)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def test_sdk_evaluate_returns_tuple(sdk: DroneSimSDK) -> None:
    """evaluate() returns a 2-tuple."""
    result = sdk.evaluate()
    assert isinstance(result, tuple) and len(result) == 2


def test_sdk_evaluate_path_is_list(sdk: DroneSimSDK) -> None:
    """First element is list."""
    path, _ = sdk.evaluate()
    assert isinstance(path, list)


def test_sdk_evaluate_stats_is_episode_stats(sdk: DroneSimSDK) -> None:
    """Second element is EpisodeStats."""
    _, stats = sdk.evaluate()
    assert isinstance(stats, EpisodeStats)


def test_sdk_evaluate_path_contains_ints(sdk: DroneSimSDK) -> None:
    """All elements of path are int."""
    path, _ = sdk.evaluate()
    assert all(isinstance(s, int) for s in path)


def test_sdk_evaluate_path_states_in_valid_range(sdk: DroneSimSDK) -> None:
    """All path states in [0, num_states)."""
    path, _ = sdk.evaluate()
    ns = sdk._env.num_states
    assert all(0 <= s < ns for s in path)


def test_sdk_evaluate_does_not_modify_q_table(sdk: DroneSimSDK) -> None:
    """Q-Table unchanged before and after evaluate()."""
    q_before = sdk.get_q_table().copy()
    sdk.evaluate()
    assert np.allclose(sdk.get_q_table(), q_before)


def test_sdk_evaluate_with_all_zero_q_table_returns_valid_path(
    sdk: DroneSimSDK,
) -> None:
    """Works even with zero Q-Table."""
    path, _ = sdk.evaluate()
    assert len(path) >= 1


def test_sdk_evaluate_path_starts_at_start_state(sdk: DroneSimSDK) -> None:
    """First element of path equals reset() state."""
    start = sdk._env.reset()
    path, _ = sdk.evaluate()
    assert path[0] == start


# ---------------------------------------------------------------------------
# Hyperparams & Grid Update
# ---------------------------------------------------------------------------


def test_sdk_set_hyperparams_alpha_changes(sdk: DroneSimSDK) -> None:
    """set_hyperparams(alpha=0.5) takes effect."""
    sdk.set_hyperparams(alpha=0.5)
    assert abs(sdk._agent._alpha - 0.5) < 1e-9


def test_sdk_set_hyperparams_gamma_changes(sdk: DroneSimSDK) -> None:
    """set_hyperparams(gamma=0.8) takes effect."""
    sdk.set_hyperparams(gamma=0.8)
    assert abs(sdk._agent._gamma - 0.8) < 1e-9


def test_sdk_set_hyperparams_epsilon_changes(sdk: DroneSimSDK) -> None:
    """set_hyperparams(epsilon=0.1) takes effect."""
    sdk.set_hyperparams(epsilon=0.1)
    assert abs(sdk._agent._epsilon - 0.1) < 1e-9


def test_sdk_update_grid_changes_cell_type(sdk: DroneSimSDK) -> None:
    """update_grid(r, c, TRAP) reflects in grid."""
    sdk.update_grid(1, 1, CellType.TRAP)
    assert sdk.get_grid()[1][1] is CellType.TRAP


def test_sdk_update_grid_invalid_row_raises(sdk: DroneSimSDK) -> None:
    """Out-of-bounds row raises IndexError."""
    with pytest.raises((IndexError, ValueError)):
        sdk.update_grid(999, 0, CellType.TRAP)


def test_sdk_update_grid_invalid_col_raises(sdk: DroneSimSDK) -> None:
    """Out-of-bounds col raises IndexError."""
    with pytest.raises((IndexError, ValueError)):
        sdk.update_grid(0, 999, CellType.TRAP)


# ---------------------------------------------------------------------------
# Persistence via SDK
# ---------------------------------------------------------------------------


def test_sdk_save_q_table_creates_file(sdk: DroneSimSDK, tmp_path: Path) -> None:
    """save_q_table(path) writes file at path."""
    p = tmp_path / "q.npy"
    sdk.save_q_table(p)
    assert p.exists()


def test_sdk_load_q_table_restores_table(sdk: DroneSimSDK, tmp_path: Path) -> None:
    """Loaded table matches previously saved table."""
    q = np.ones(sdk.get_q_table().shape)
    sdk._agent.set_q_table(q)
    p = tmp_path / "q.npy"
    sdk.save_q_table(p)
    sdk.reset()
    sdk.load_q_table(p)
    assert np.allclose(sdk.get_q_table(), q)


def test_sdk_load_q_table_nonexistent_path_raises(
    sdk: DroneSimSDK, tmp_path: Path
) -> None:
    """FileNotFoundError propagated to caller."""
    with pytest.raises(FileNotFoundError):
        sdk.load_q_table(tmp_path / "missing.npy")


# ---------------------------------------------------------------------------
# load_default_grid
# ---------------------------------------------------------------------------


def test_sdk_load_default_grid_no_error(sdk: DroneSimSDK) -> None:
    """load_default_grid() does not raise."""
    sdk.load_default_grid()


def test_sdk_load_default_grid_grid_has_one_start(sdk: DroneSimSDK) -> None:
    """Exactly one CellType.START after reset."""
    sdk.update_grid(0, 0, CellType.EMPTY)  # remove start temporarily
    sdk.load_default_grid()
    starts = sum(c is CellType.START for row in sdk.get_grid() for c in row)
    assert starts == 1


def test_sdk_load_default_grid_grid_has_one_target(sdk: DroneSimSDK) -> None:
    """Exactly one CellType.TARGET after reset."""
    sdk.load_default_grid()
    targets = sum(c is CellType.TARGET for row in sdk.get_grid() for c in row)
    assert targets == 1


def test_sdk_load_default_grid_matches_config_dimensions(sdk: DroneSimSDK) -> None:
    """Grid size matches config/env.json."""
    sdk.load_default_grid()
    cfg = sdk._env_cfg
    g = sdk.get_grid()
    assert len(g) == cfg["rows"]
    assert all(len(row) == cfg["cols"] for row in g)


def test_sdk_set_vis_delay_delegates_to_loop(sdk: DroneSimSDK) -> None:
    """sdk.set_vis_delay(200) changes training_loop.vis_delay."""
    sdk.start_training()
    sdk.set_vis_delay(200)
    assert abs(sdk._loop.vis_delay - 0.2) < 1e-9
    sdk.stop()
