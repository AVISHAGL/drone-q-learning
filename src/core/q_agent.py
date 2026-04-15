"""Tabular Q-Learning agent with ε-greedy action selection."""

import numpy as np

from src.core.episode_stats import EpisodeStats

__all__ = ["QLearningAgent"]


class QLearningAgent:
    """Off-policy tabular Q-Learning agent.

    Implements the Bellman update rule:
        Q(s,a) ← Q(s,a) + α·[r + γ·max_{a'}Q(s',a') − Q(s,a)]

    Args:
        num_states: Number of discrete states in the environment.
        num_actions: Number of discrete actions (4 for the drone).
        config: Dictionary loaded from config/rl.json.  Must contain keys
            'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'epsilon_min',
            'episodes', and 'max_steps'.
    """

    def __init__(self, num_states: int, num_actions: int, config: dict) -> None:
        """Initialise the agent from an RL config dict."""
        self._ns = num_states
        self._na = num_actions
        self._alpha: float = float(config["alpha"])
        self._gamma: float = float(config["gamma"])
        self._epsilon: float = float(config["epsilon"])
        self._epsilon_decay: float = float(config["epsilon_decay"])
        self._epsilon_min: float = float(config["epsilon_min"])
        self._init_epsilon: float = self._epsilon
        self._q: np.ndarray = np.zeros((num_states, num_actions))
        self._episode: int = 0
        self._reset_accumulators()


    def select_action(self, state: int) -> int:
        """ε-Greedy: random with prob ε, else argmax Q(s,a)."""
        if np.random.random() < self._epsilon:
            return int(np.random.randint(0, self._na))
        return int(np.argmax(self._q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> float:
        """Apply Bellman update Q(s,a) ← Q+α[r+γ·max Q(s')-Q] and return |Δq|."""
        old_q = self._q[state, action]
        td_target = reward + self._gamma * float(np.max(self._q[next_state]))
        new_q = old_q + self._alpha * (td_target - old_q)
        self._q[state, action] = new_q
        delta = abs(new_q - old_q)
        self._total_reward += reward
        self._steps += 1
        if delta > self._max_delta_q:
            self._max_delta_q = delta
        return delta

    def end_episode(self, reached_target: bool = False) -> EpisodeStats:
        """Finalise the episode: decay ε and return statistics.

        Args:
            reached_target: Whether the drone reached the TARGET cell.

        Returns:
            EpisodeStats for the completed episode.
        """
        self._episode += 1
        stats = EpisodeStats(
            episode=self._episode,
            steps=self._steps,
            total_reward=self._total_reward,
            max_delta_q=self._max_delta_q,
            reached_target=reached_target,
        )
        self._epsilon = max(
            self._epsilon * self._epsilon_decay, self._epsilon_min
        )
        self._reset_accumulators()
        return stats


    def get_policy(self) -> dict[int, int]:
        """Return the greedy policy as a state→action mapping.

        Returns:
            Dict mapping every state to its argmax action.
        """
        return {s: int(np.argmax(self._q[s])) for s in range(self._ns)}

    def get_q_table(self) -> np.ndarray:
        """Return a copy of the Q-Table.

        Returns:
            NumPy array of shape (num_states, num_actions).
        """
        return self._q.copy()

    def set_q_table(self, q: np.ndarray) -> None:
        """Replace the internal Q-Table with a copy of the given array.

        Args:
            q: New Q-Table.  Must have shape (num_states, num_actions).

        Raises:
            ValueError: If shape does not match.
        """
        if q.shape != (self._ns, self._na):
            msg = f"Q-Table shape {q.shape} != expected {(self._ns, self._na)}"
            raise ValueError(msg)
        self._q = q.copy()

    def reset(self) -> None:
        """Reset Q-Table, episode counter, epsilon, and accumulators."""
        self._q = np.zeros((self._ns, self._na))
        self._episode = 0
        self._epsilon = self._init_epsilon
        self._reset_accumulators()

    def update_params(self, **kwargs: float) -> None:
        """Update hyperparameters at runtime.

        Raises:
            ValueError: If an unknown key is provided.
        """
        allowed = {"alpha", "gamma", "epsilon", "epsilon_decay", "epsilon_min"}
        for key, val in kwargs.items():
            if key not in allowed:
                msg = f"Unknown hyperparameter: {key!r}"
                raise ValueError(msg)
            setattr(self, f"_{key}", float(val))


    def _reset_accumulators(self) -> None:
        self._total_reward: float = 0.0
        self._steps: int = 0
        self._max_delta_q: float = 0.0
