"""Tests for src.sdk.training_loop — TrainingLoop threading."""

import queue

import pytest

from src.core.episode_stats import EpisodeStats
from src.core.grid_world import GridWorld
from src.core.q_agent import QLearningAgent
from src.sdk.training_loop import TrainingLoop


@pytest.fixture()
def small_cfg() -> dict:
    """Minimal configs for fast loop tests (5 episodes)."""
    env = {
        "rows": 3,
        "cols": 3,
        "max_steps_per_episode": 20,
        "rewards": {"empty": -1, "wind": -3, "trap": -10, "building": -1, "target": 50},
        "default_grid": [],
    }
    rl = {
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "episodes": 5,
        "max_steps": 20,
        "vis_delay_ms": 0,
    }
    return {"env": env, "rl": rl}


@pytest.fixture()
def loop(small_cfg: dict) -> TrainingLoop:
    """Fresh TrainingLoop for a 3×3 grid."""
    env = GridWorld(small_cfg["env"])
    agent = QLearningAgent(env.num_states, env.num_actions, small_cfg["rl"])
    q: queue.Queue = queue.Queue()
    return TrainingLoop(env, agent, q, small_cfg["rl"])


def _get_loop_queue(lp: TrainingLoop) -> queue.Queue:
    return lp._queue


def test_training_loop_init_no_error(small_cfg: dict) -> None:
    """TrainingLoop constructs without error."""
    env = GridWorld(small_cfg["env"])
    agent = QLearningAgent(env.num_states, env.num_actions, small_cfg["rl"])
    TrainingLoop(env, agent, queue.Queue(), small_cfg["rl"])


def test_training_loop_start_creates_thread(loop: TrainingLoop) -> None:
    """After start(), _thread is not None."""
    loop.start()
    assert loop._thread is not None
    loop.stop()


def test_training_loop_start_thread_is_daemon(loop: TrainingLoop) -> None:
    """Thread is a daemon thread."""
    loop.start()
    assert loop._thread is not None
    assert loop._thread.daemon is True
    loop.stop()


def test_training_loop_start_thread_is_alive(small_cfg: dict) -> None:
    """Thread is alive shortly after start() (using many episodes so it doesn't finish instantly)."""
    rl = dict(small_cfg["rl"], episodes=10000)
    env = GridWorld(small_cfg["env"])
    agent = QLearningAgent(env.num_states, env.num_actions, rl)
    lp = TrainingLoop(env, agent, queue.Queue(), rl)
    lp.start()
    # Thread should still be running because episodes=10000
    assert lp._thread is not None and lp._thread.is_alive()
    lp.stop()


def test_training_loop_queue_receives_stats_within_timeout(
    loop: TrainingLoop,
) -> None:
    """At least one EpisodeStats posted to queue within 5s."""
    loop.start()
    q = _get_loop_queue(loop)
    item = q.get(timeout=5.0)
    assert item is not None
    loop.stop()


def test_training_loop_stats_is_episode_stats_instance(
    loop: TrainingLoop,
) -> None:
    """Item from queue is EpisodeStats."""
    loop.start()
    q = _get_loop_queue(loop)
    item = q.get(timeout=5.0)
    assert isinstance(item, EpisodeStats)
    loop.stop()


def test_training_loop_stop_terminates_thread(loop: TrainingLoop) -> None:
    """After stop(), thread not alive within 3s."""
    loop.start()
    loop.stop()
    assert loop._thread is None or not loop._thread.is_alive()


def test_training_loop_stop_before_start_no_error(loop: TrainingLoop) -> None:
    """Calling stop() without start() raises no exception."""
    loop.stop()


def test_training_loop_episode_counter_increases(loop: TrainingLoop) -> None:
    """Second stat has higher episode number than first."""
    loop.start()
    q = _get_loop_queue(loop)
    s1 = q.get(timeout=5.0)
    s2 = q.get(timeout=5.0)
    assert s2.episode > s1.episode
    loop.stop()


def test_training_loop_pause_is_idempotent(loop: TrainingLoop) -> None:
    """Calling pause() twice does not raise."""
    loop.start()
    loop.pause()
    loop.pause()
    loop.stop()


def test_training_loop_resume_without_pause_no_error(loop: TrainingLoop) -> None:
    """Calling resume() without prior pause() is safe."""
    loop.start()
    loop.resume()
    loop.stop()


def test_training_loop_default_vis_delay_is_zero(
    small_cfg: dict,
) -> None:
    """loop.vis_delay == 0.0 after construction."""
    env = GridWorld(small_cfg["env"])
    agent = QLearningAgent(env.num_states, env.num_actions, small_cfg["rl"])
    lp = TrainingLoop(env, agent, queue.Queue(), small_cfg["rl"])
    assert lp.vis_delay == 0.0


def test_training_loop_set_vis_delay_100ms_stores_01s(
    loop: TrainingLoop,
) -> None:
    """set_vis_delay(100) → loop.vis_delay == 0.1."""
    loop.set_vis_delay(100)
    assert abs(loop.vis_delay - 0.1) < 1e-9


def test_training_loop_set_vis_delay_zero_is_valid(loop: TrainingLoop) -> None:
    """set_vis_delay(0) does not raise."""
    loop.set_vis_delay(0)
    assert loop.vis_delay == 0.0


def test_training_loop_set_vis_delay_500ms_stores_05s(loop: TrainingLoop) -> None:
    """set_vis_delay(500) → loop.vis_delay == 0.5."""
    loop.set_vis_delay(500)
    assert abs(loop.vis_delay - 0.5) < 1e-9
