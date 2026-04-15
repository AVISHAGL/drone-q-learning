"""Tests for src.core.q_agent — QLearningAgent."""

import numpy as np
import pytest

from src.core.episode_stats import EpisodeStats
from src.core.q_agent import QLearningAgent


@pytest.fixture()
def cfg() -> dict:
    """Default RL config for tests."""
    return {
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "episodes": 100,
        "max_steps": 50,
    }


@pytest.fixture()
def agent(cfg: dict) -> QLearningAgent:
    """Fresh 10-state, 4-action agent."""
    return QLearningAgent(10, 4, cfg)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_q_agent_init_no_error(cfg: dict) -> None:
    """QLearningAgent constructs without error."""
    QLearningAgent(10, 4, cfg)


def test_q_agent_q_table_is_numpy_array(agent: QLearningAgent) -> None:
    """get_q_table() returns np.ndarray."""
    assert isinstance(agent.get_q_table(), np.ndarray)


def test_q_agent_q_table_shape_num_states_x_4(agent: QLearningAgent) -> None:
    """Q-Table shape is (10, 4)."""
    assert agent.get_q_table().shape == (10, 4)


def test_q_agent_q_table_initialized_to_zeros(agent: QLearningAgent) -> None:
    """All Q-Table values equal 0.0."""
    assert np.all(agent.get_q_table() == 0.0)


def test_q_agent_get_policy_returns_dict(agent: QLearningAgent) -> None:
    """get_policy() returns dict."""
    assert isinstance(agent.get_policy(), dict)


def test_q_agent_get_policy_has_all_states(agent: QLearningAgent) -> None:
    """Policy dict has num_states entries."""
    assert len(agent.get_policy()) == 10


def test_q_agent_get_policy_all_actions_valid(agent: QLearningAgent) -> None:
    """Every action value is in {0, 1, 2, 3}."""
    assert all(v in {0, 1, 2, 3} for v in agent.get_policy().values())


# ---------------------------------------------------------------------------
# Bellman Update
# ---------------------------------------------------------------------------


def test_q_agent_update_returns_float(agent: QLearningAgent) -> None:
    """update() returns float."""
    result = agent.update(0, 0, -1.0, 1)
    assert isinstance(result, float)


def test_q_agent_update_zero_table_terminal_step(cfg: dict) -> None:
    """On all-zero table: Q(s,a) == alpha * r (max Q(s')=0)."""
    a = QLearningAgent(10, 4, cfg)
    a.update(0, 0, 10.0, 1)
    expected = cfg["alpha"] * 10.0
    assert abs(a.get_q_table()[0, 0] - expected) < 1e-9


def test_q_agent_update_positive_reward_increases_q(agent: QLearningAgent) -> None:
    """Positive reward increases Q(s,a)."""
    agent.update(0, 0, 50.0, 1)
    assert agent.get_q_table()[0, 0] > 0


def test_q_agent_update_negative_reward_decreases_q(agent: QLearningAgent) -> None:
    """Negative reward on all-zero table keeps Q negative (or zero for small |r|)."""
    agent.update(0, 0, -10.0, 1)
    assert agent.get_q_table()[0, 0] < 0


def test_q_agent_update_exact_bellman_formula(cfg: dict) -> None:
    """Numeric assertion with known alpha/gamma/r/Q values."""
    a = QLearningAgent(10, 4, cfg)
    a.get_q_table()  # confirm zeros
    # Set Q(1, *) manually so max Q(s')=5
    q = a.get_q_table()
    q[1, 2] = 5.0
    a.set_q_table(q)
    a.update(0, 0, -1.0, 1)
    # Expected: 0 + 0.1 * (-1 + 0.9 * 5 - 0) = 0.1 * 3.5 = 0.35
    assert abs(a.get_q_table()[0, 0] - 0.35) < 1e-9


def test_q_agent_update_delta_q_equals_absolute_change(cfg: dict) -> None:
    """Returned Δq equals |new_q - old_q|."""
    a = QLearningAgent(10, 4, cfg)
    old = a.get_q_table()[0, 0]
    delta = a.update(0, 0, 5.0, 1)
    new = a.get_q_table()[0, 0]
    assert abs(delta - abs(new - old)) < 1e-9


def test_q_agent_update_only_changes_given_action(agent: QLearningAgent) -> None:
    """Other (s, a') entries are unchanged."""
    agent.update(0, 0, 5.0, 1)
    for a in range(1, 4):
        assert agent.get_q_table()[0, a] == 0.0


def test_q_agent_update_uses_max_next_q(cfg: dict) -> None:
    """Uses max Q(s', a') over all actions."""
    a = QLearningAgent(10, 4, cfg)
    q = a.get_q_table()
    q[1, 3] = 100.0
    a.set_q_table(q)
    a.update(0, 0, 0.0, 1)
    # Expected: 0.1 * (0 + 0.9 * 100) = 9.0
    assert abs(a.get_q_table()[0, 0] - 9.0) < 1e-9


def test_q_agent_update_alpha_zero_no_change(cfg: dict) -> None:
    """With alpha=0, Q-Table unchanged."""
    c = dict(cfg, alpha=0.0)
    a = QLearningAgent(10, 4, c)
    a.update(0, 0, 50.0, 1)
    assert a.get_q_table()[0, 0] == 0.0


def test_q_agent_update_alpha_one_full_update(cfg: dict) -> None:
    """With alpha=1: Q(s,a) = r + gamma * max_Q."""
    c = dict(cfg, alpha=1.0)
    a = QLearningAgent(10, 4, c)
    a.update(0, 0, 5.0, 1)
    assert abs(a.get_q_table()[0, 0] - 5.0) < 1e-9


def test_q_agent_update_gamma_zero_ignores_future(cfg: dict) -> None:
    """With gamma=0: Q(s,a) += alpha*(r - Q(s,a))."""
    c = dict(cfg, gamma=0.0, alpha=1.0)
    a = QLearningAgent(10, 4, c)
    q = a.get_q_table()
    q[1, 0] = 999.0
    a.set_q_table(q)
    a.update(0, 0, 3.0, 1)  # max Q(s')=999 ignored because gamma=0
    assert abs(a.get_q_table()[0, 0] - 3.0) < 1e-9


def test_q_agent_update_multiple_steps_accumulates(cfg: dict) -> None:
    """Q(s,a) after 5 updates is consistent with repeated Bellman formula."""
    a = QLearningAgent(10, 4, cfg)
    for _ in range(5):
        a.update(0, 0, -1.0, 1)
    # Q should be negative and non-zero
    assert a.get_q_table()[0, 0] < 0


# ---------------------------------------------------------------------------
# ε-Greedy Action Selection
# ---------------------------------------------------------------------------


def test_q_agent_select_action_returns_int(agent: QLearningAgent) -> None:
    """select_action(state) returns int."""
    assert isinstance(agent.select_action(0), int)


def test_q_agent_select_action_in_valid_range(agent: QLearningAgent) -> None:
    """Returned value in [0, num_actions)."""
    for _ in range(50):
        assert 0 <= agent.select_action(0) < 4


def test_q_agent_select_action_epsilon_one_is_random(cfg: dict) -> None:
    """With epsilon=1.0, 100 calls are not always the same action."""
    c = dict(cfg, epsilon=1.0)
    a = QLearningAgent(10, 4, c)
    actions = {a.select_action(0) for _ in range(100)}
    assert len(actions) > 1


def test_q_agent_select_action_epsilon_zero_is_greedy(cfg: dict) -> None:
    """With epsilon=0.0, always returns argmax Q(s,a)."""
    c = dict(cfg, epsilon=0.0)
    a = QLearningAgent(10, 4, c)
    q = a.get_q_table()
    q[0, 2] = 99.0
    a.set_q_table(q)
    for _ in range(20):
        assert a.select_action(0) == 2


def test_q_agent_select_action_greedy_picks_max_q(cfg: dict) -> None:
    """Manually set one Q high; greedy picks it."""
    c = dict(cfg, epsilon=0.0)
    a = QLearningAgent(10, 4, c)
    q = a.get_q_table()
    q[3, 1] = 50.0
    a.set_q_table(q)
    assert a.select_action(3) == 1


def test_q_agent_select_action_zero_q_table_valid_action(cfg: dict) -> None:
    """All-zero table still returns a valid action."""
    c = dict(cfg, epsilon=0.0)
    a = QLearningAgent(10, 4, c)
    assert 0 <= a.select_action(0) < 4


def test_q_agent_select_action_different_states_different_best(cfg: dict) -> None:
    """Each state returns its own argmax."""
    c = dict(cfg, epsilon=0.0)
    a = QLearningAgent(10, 4, c)
    q = np.zeros((10, 4))
    q[0, 0] = 1.0
    q[1, 3] = 1.0
    a.set_q_table(q)
    assert a.select_action(0) == 0
    assert a.select_action(1) == 3


# ---------------------------------------------------------------------------
# Epsilon Decay
# ---------------------------------------------------------------------------


def test_q_agent_end_episode_returns_episode_stats(agent: QLearningAgent) -> None:
    """end_episode() returns EpisodeStats."""
    assert isinstance(agent.end_episode(), EpisodeStats)


def test_q_agent_end_episode_epsilon_decreases(agent: QLearningAgent) -> None:
    """Epsilon after end_episode() < epsilon before."""
    eps_before = agent._epsilon
    agent.end_episode()
    assert agent._epsilon < eps_before


def test_q_agent_end_episode_epsilon_decay_multiplied(cfg: dict) -> None:
    """new_eps == max(old_eps * decay, eps_min)."""
    c = dict(cfg, epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.01)
    a = QLearningAgent(10, 4, c)
    a.end_episode()
    assert abs(a._epsilon - max(0.5 * 0.9, 0.01)) < 1e-9


def test_q_agent_end_episode_epsilon_not_below_min(cfg: dict) -> None:
    """Epsilon never goes below epsilon_min."""
    c = dict(cfg, epsilon=0.001, epsilon_decay=0.01, epsilon_min=0.01)
    a = QLearningAgent(10, 4, c)
    a.end_episode()
    assert a._epsilon >= 0.01


def test_q_agent_end_episode_increments_episode_counter(
    agent: QLearningAgent,
) -> None:
    """Episode number increments by 1."""
    stats = agent.end_episode()
    assert stats.episode == 1


def test_q_agent_end_episode_stats_steps_reflects_counted_steps(
    agent: QLearningAgent,
) -> None:
    """stats.steps matches steps counted."""
    agent.update(0, 0, -1.0, 1)
    agent.update(1, 2, -1.0, 2)
    stats = agent.end_episode()
    assert stats.steps == 2


def test_q_agent_end_episode_stats_total_reward_reflects_cumulative(
    agent: QLearningAgent,
) -> None:
    """stats.total_reward matches sum of rewards."""
    agent.update(0, 0, -3.0, 1)
    agent.update(1, 0, -1.0, 2)
    stats = agent.end_episode()
    assert abs(stats.total_reward - (-4.0)) < 1e-9


def test_q_agent_end_episode_stats_max_delta_q_is_max_of_episode(
    cfg: dict,
) -> None:
    """stats.max_delta_q equals max Δq of the episode."""
    a = QLearningAgent(10, 4, cfg)
    d1 = a.update(0, 0, 50.0, 1)
    d2 = a.update(1, 1, -1.0, 2)
    stats = a.end_episode()
    assert abs(stats.max_delta_q - max(d1, d2)) < 1e-9


def test_q_agent_end_episode_resets_per_episode_accumulators(
    agent: QLearningAgent,
) -> None:
    """After end_episode, steps=0, reward=0.0."""
    agent.update(0, 0, -1.0, 1)
    agent.end_episode()
    assert agent._steps == 0
    assert agent._total_reward == 0.0


def test_q_agent_end_episode_reached_target_true_when_signaled(
    agent: QLearningAgent,
) -> None:
    """reached_target reflects signal."""
    stats = agent.end_episode(reached_target=True)
    assert stats.reached_target is True


def test_q_agent_end_episode_reached_target_false_by_default(
    agent: QLearningAgent,
) -> None:
    """Without signal, reached_target=False."""
    stats = agent.end_episode()
    assert stats.reached_target is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_q_agent_reset_q_table_all_zeros(agent: QLearningAgent) -> None:
    """After reset(), all Q values are 0.0."""
    agent.update(0, 0, 50.0, 1)
    agent.reset()
    assert np.all(agent.get_q_table() == 0.0)


def test_q_agent_reset_episode_counter_to_zero(agent: QLearningAgent) -> None:
    """After reset(), episode counter is 0."""
    agent.end_episode()
    agent.reset()
    assert agent._episode == 0


def test_q_agent_reset_epsilon_to_initial_value(cfg: dict) -> None:
    """After reset(), epsilon equals config epsilon."""
    a = QLearningAgent(10, 4, cfg)
    for _ in range(5):
        a.end_episode()
    a.reset()
    assert abs(a._epsilon - cfg["epsilon"]) < 1e-9


def test_q_agent_reset_after_training_clears_q_table(agent: QLearningAgent) -> None:
    """Q-Table previously non-zero is zeroed after reset."""
    for _ in range(10):
        agent.update(0, 0, 50.0, 1)
    assert agent.get_q_table()[0, 0] != 0.0
    agent.reset()
    assert agent.get_q_table()[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Hyperparameter Update
# ---------------------------------------------------------------------------


def test_q_agent_update_params_alpha_changes(agent: QLearningAgent) -> None:
    """update_params(alpha=0.5) changes alpha to 0.5."""
    agent.update_params(alpha=0.5)
    assert agent._alpha == 0.5


def test_q_agent_update_params_gamma_changes(agent: QLearningAgent) -> None:
    """update_params(gamma=0.8) changes gamma to 0.8."""
    agent.update_params(gamma=0.8)
    assert agent._gamma == 0.8


def test_q_agent_update_params_epsilon_changes(agent: QLearningAgent) -> None:
    """update_params(epsilon=0.3) changes epsilon to 0.3."""
    agent.update_params(epsilon=0.3)
    assert agent._epsilon == 0.3


def test_q_agent_update_params_unknown_key_raises_value_error(
    agent: QLearningAgent,
) -> None:
    """Unknown key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        agent.update_params(learning_rate=0.1)


def test_q_agent_update_params_multiple_keys_at_once(agent: QLearningAgent) -> None:
    """Can update alpha and gamma simultaneously."""
    agent.update_params(alpha=0.5, gamma=0.8)
    assert agent._alpha == 0.5
    assert agent._gamma == 0.8


def test_q_agent_update_params_effect_on_next_update(cfg: dict) -> None:
    """Changed alpha affects next Bellman update."""
    a = QLearningAgent(10, 4, cfg)
    a.update_params(alpha=1.0)
    a.update(0, 0, 5.0, 1)
    assert abs(a.get_q_table()[0, 0] - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Q-Table Set/Get
# ---------------------------------------------------------------------------


def test_q_agent_set_q_table_replaces_internal_table(agent: QLearningAgent) -> None:
    """get_q_table() returns new table after set_q_table()."""
    q = np.ones((10, 4))
    agent.set_q_table(q)
    assert np.allclose(agent.get_q_table(), q)


def test_q_agent_set_q_table_wrong_shape_raises_value_error(
    agent: QLearningAgent,
) -> None:
    """Wrong shape raises ValueError."""
    with pytest.raises(ValueError):
        agent.set_q_table(np.zeros((5, 4)))


def test_q_agent_set_q_table_stores_independent_copy(agent: QLearningAgent) -> None:
    """Modifying external array after set_q_table does not affect agent."""
    q = np.ones((10, 4))
    agent.set_q_table(q)
    q[:] = 99.0
    assert agent.get_q_table()[0, 0] == 1.0


def test_q_agent_get_q_table_returns_copy(agent: QLearningAgent) -> None:
    """Modifying returned array does not affect agent."""
    q = agent.get_q_table()
    q[0, 0] = 999.0
    assert agent.get_q_table()[0, 0] == 0.0
