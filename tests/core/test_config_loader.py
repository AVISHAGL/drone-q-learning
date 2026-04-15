"""Tests for src.core.config_loader — JSON configuration loading."""

import json
from pathlib import Path

import pytest

from src.core.config_loader import load_env_config, load_rl_config

CONFIG_ENV = Path(__file__).parent.parent.parent / "config" / "env.json"
CONFIG_RL = Path(__file__).parent.parent.parent / "config" / "rl.json"


# --- load_env_config ---


def test_load_env_config_returns_dict() -> None:
    """Return type is dict."""
    assert isinstance(load_env_config(), dict)


def test_load_env_config_has_rows_key() -> None:
    """'rows' key exists in env config."""
    assert "rows" in load_env_config()


def test_load_env_config_has_cols_key() -> None:
    """'cols' key exists in env config."""
    assert "cols" in load_env_config()


def test_load_env_config_has_rewards_key() -> None:
    """'rewards' key exists in env config."""
    assert "rewards" in load_env_config()


def test_load_env_config_has_max_steps_key() -> None:
    """'max_steps_per_episode' key exists in env config."""
    assert "max_steps_per_episode" in load_env_config()


def test_load_env_config_has_default_grid_key() -> None:
    """'default_grid' key exists in env config."""
    assert "default_grid" in load_env_config()


def test_load_env_config_has_version_key() -> None:
    """'version' key exists in env config."""
    assert "version" in load_env_config()


def test_load_env_config_rows_is_int() -> None:
    """config['rows'] is int."""
    assert isinstance(load_env_config()["rows"], int)


def test_load_env_config_cols_is_int() -> None:
    """config['cols'] is int."""
    assert isinstance(load_env_config()["cols"], int)


def test_load_env_config_rewards_is_dict() -> None:
    """config['rewards'] is dict."""
    assert isinstance(load_env_config()["rewards"], dict)


def test_load_env_config_rewards_has_empty_key() -> None:
    """config['rewards']['empty'] exists."""
    assert "empty" in load_env_config()["rewards"]


def test_load_env_config_rewards_has_wind_key() -> None:
    """config['rewards']['wind'] exists."""
    assert "wind" in load_env_config()["rewards"]


def test_load_env_config_rewards_has_trap_key() -> None:
    """config['rewards']['trap'] exists."""
    assert "trap" in load_env_config()["rewards"]


def test_load_env_config_rewards_has_building_key() -> None:
    """config['rewards']['building'] exists."""
    assert "building" in load_env_config()["rewards"]


def test_load_env_config_rewards_has_target_key() -> None:
    """config['rewards']['target'] exists."""
    assert "target" in load_env_config()["rewards"]


def test_load_env_config_default_grid_is_list() -> None:
    """config['default_grid'] is list."""
    assert isinstance(load_env_config()["default_grid"], list)


# --- load_rl_config ---


def test_load_rl_config_returns_dict() -> None:
    """Return type is dict."""
    assert isinstance(load_rl_config(), dict)


def test_load_rl_config_has_alpha_key() -> None:
    """'alpha' key exists."""
    assert "alpha" in load_rl_config()


def test_load_rl_config_has_gamma_key() -> None:
    """'gamma' key exists."""
    assert "gamma" in load_rl_config()


def test_load_rl_config_has_epsilon_key() -> None:
    """'epsilon' key exists."""
    assert "epsilon" in load_rl_config()


def test_load_rl_config_has_epsilon_decay_key() -> None:
    """'epsilon_decay' key exists."""
    assert "epsilon_decay" in load_rl_config()


def test_load_rl_config_has_epsilon_min_key() -> None:
    """'epsilon_min' key exists."""
    assert "epsilon_min" in load_rl_config()


def test_load_rl_config_has_episodes_key() -> None:
    """'episodes' key exists."""
    assert "episodes" in load_rl_config()


def test_load_rl_config_has_max_steps_key() -> None:
    """'max_steps' key exists."""
    assert "max_steps" in load_rl_config()


def test_load_rl_config_has_vis_every_n_key() -> None:
    """'vis_every_n' key exists."""
    assert "vis_every_n" in load_rl_config()


def test_load_rl_config_has_version_key() -> None:
    """'version' key exists."""
    assert "version" in load_rl_config()


def test_load_rl_config_has_vis_delay_ms_key() -> None:
    """'vis_delay_ms' key exists."""
    assert "vis_delay_ms" in load_rl_config()


def test_load_rl_config_alpha_is_float() -> None:
    """config['alpha'] is float."""
    assert isinstance(load_rl_config()["alpha"], float)


def test_load_rl_config_gamma_in_range() -> None:
    """0 < gamma <= 1."""
    g = load_rl_config()["gamma"]
    assert 0 < g <= 1


def test_load_rl_config_epsilon_in_range() -> None:
    """0 <= epsilon <= 1."""
    e = load_rl_config()["epsilon"]
    assert 0 <= e <= 1


def test_load_env_config_missing_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_env_config(tmp_path / "no_such_file.json")


def test_load_rl_config_missing_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_rl_config(tmp_path / "no_such_file.json")


def test_load_env_config_malformed_json_raises_json_decode_error(
    tmp_path: Path,
) -> None:
    """Malformed JSON raises json.JSONDecodeError."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_env_config(bad)


def test_load_rl_config_malformed_json_raises_json_decode_error(
    tmp_path: Path,
) -> None:
    """Malformed JSON raises json.JSONDecodeError."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_rl_config(bad)
