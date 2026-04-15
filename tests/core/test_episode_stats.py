"""Tests for src.core.episode_stats — EpisodeStats dataclass."""

import dataclasses

from src.core.episode_stats import EpisodeStats


def _make(**kwargs) -> EpisodeStats:
    defaults = {
        "episode": 1, "steps": 10, "total_reward": -5.0,
        "max_delta_q": 0.5, "reached_target": False,
    }
    defaults.update(kwargs)
    return EpisodeStats(**defaults)


def test_episode_stats_is_dataclass() -> None:
    """EpisodeStats has __dataclass_fields__ attribute."""
    assert hasattr(EpisodeStats, "__dataclass_fields__")


def test_episode_stats_construction_all_fields() -> None:
    """EpisodeStats constructs with all five fields without error."""
    stats = _make()
    assert stats is not None


def test_episode_stats_field_episode_type_is_int() -> None:
    """stats.episode is int."""
    assert isinstance(_make().episode, int)


def test_episode_stats_field_steps_type_is_int() -> None:
    """stats.steps is int."""
    assert isinstance(_make().steps, int)


def test_episode_stats_field_total_reward_type_is_float() -> None:
    """stats.total_reward is float."""
    assert isinstance(_make().total_reward, float)


def test_episode_stats_field_max_delta_q_type_is_float() -> None:
    """stats.max_delta_q is float."""
    assert isinstance(_make().max_delta_q, float)


def test_episode_stats_field_reached_target_type_is_bool() -> None:
    """stats.reached_target is bool."""
    assert isinstance(_make().reached_target, bool)


def test_episode_stats_equality_same_values() -> None:
    """Two instances with same values compare equal."""
    a = _make()
    b = _make()
    assert a == b


def test_episode_stats_inequality_different_episode() -> None:
    """Different episode values are not equal."""
    assert _make(episode=1) != _make(episode=2)


def test_episode_stats_inequality_different_reward() -> None:
    """Different total_reward values are not equal."""
    assert _make(total_reward=-1.0) != _make(total_reward=-2.0)


def test_episode_stats_reached_target_true() -> None:
    """reached_target=True stores correctly."""
    assert _make(reached_target=True).reached_target is True


def test_episode_stats_reached_target_false() -> None:
    """reached_target=False stores correctly."""
    assert _make(reached_target=False).reached_target is False


def test_episode_stats_negative_reward_stored() -> None:
    """Negative total_reward stores correctly."""
    assert _make(total_reward=-999.0).total_reward == -999.0


def test_episode_stats_zero_steps_valid() -> None:
    """steps=0 is a valid value."""
    stats = _make(steps=0)
    assert stats.steps == 0


def test_episode_stats_zero_delta_q_valid() -> None:
    """max_delta_q=0.0 is a valid value."""
    stats = _make(max_delta_q=0.0)
    assert stats.max_delta_q == 0.0


def test_episode_stats_fields_list_has_five_items() -> None:
    """EpisodeStats has exactly 5 dataclass fields."""
    assert len(dataclasses.fields(EpisodeStats)) == 5
