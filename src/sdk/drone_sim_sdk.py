"""DroneSimSDK — single façade entry point for all simulation logic."""

import queue
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.core.cell_type import CellType
from src.core.config_loader import load_env_config, load_rl_config
from src.core.episode_stats import EpisodeStats
from src.core.grid_world import GridWorld
from src.core.q_agent import QLearningAgent
from src.sdk.persistence import load_grid, load_q_table, save_grid, save_q_table
from src.sdk.training_loop import TrainingLoop

__all__ = ["DroneSimSDK"]

class DroneSimSDK:
    """Façade exposing all simulation capabilities to the GUI layer.

    The GUI must only import from this class — never from src.core directly.
    All business logic (RL, environment, persistence) is encapsulated here.
    """

    def __init__(self) -> None:
        """Load configs, build environment and agent, prepare queue."""
        self._env_cfg = load_env_config()
        self._rl_cfg = load_rl_config()
        self._env = GridWorld(self._env_cfg)
        self._agent = QLearningAgent(
            self._env.num_states, self._env.num_actions, self._rl_cfg
        )
        self._queue: queue.Queue = queue.Queue()
        self._loop: TrainingLoop | None = None
        self._default_grid = deepcopy(self._env.get_grid())

    def start_training(self) -> None:
        """Begin (or continue) the Q-Learning training loop."""
        if self._loop is None:
            self._loop = TrainingLoop(
                self._env, self._agent, self._queue, self._rl_cfg
            )
        self._loop.start()

    def pause(self) -> None:
        """Pause training without resetting the Q-Table."""
        if self._loop is not None:
            self._loop.pause()

    def resume(self) -> None:
        """Resume a paused training loop."""
        if self._loop is not None:
            self._loop.resume()

    def stop(self) -> None:
        """Stop training; Q-Table is preserved."""
        if self._loop is not None:
            self._loop.stop()
            self._loop = None

    def reset(self) -> None:
        """Clear Q-Table, episode counter, and stats queue."""
        self.stop()
        self._agent.reset()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def evaluate(self) -> tuple[list[int], EpisodeStats]:
        """Run one greedy episode (ε=0) without modifying the Q-Table."""
        saved_eps, saved_q = self._agent._epsilon, self._agent.get_q_table()
        self._agent._epsilon = 0.0
        state = self._env.reset()
        path = [state]
        for _ in range(int(self._rl_cfg["max_steps"])):
            action = self._agent.select_action(state)
            next_state, reward, done = self._env.step(state, action)
            self._agent._total_reward += reward
            self._agent._steps += 1
            state = next_state
            path.append(state)
            if done:
                break
        r, c = self._env.state_to_pos(state)
        reached = self._env.get_grid()[r][c] is CellType.TARGET
        stats = self._agent.end_episode(reached_target=reached)
        self._agent.set_q_table(saved_q)
        self._agent._epsilon = saved_eps
        self._agent._episode -= 1
        return path, stats

    def get_policy(self) -> dict[int, int]:
        """Return greedy policy mapping state → best action."""
        return self._agent.get_policy()

    def get_q_table(self) -> np.ndarray:
        """Return a copy of the current Q-Table."""
        return self._agent.get_q_table()

    def get_stats_queue(self) -> queue.Queue:
        """Return the stats queue polled by the GUI."""
        return self._queue

    def set_hyperparams(self, **kwargs: float) -> None:
        """Update agent hyperparameters at runtime."""
        self._agent.update_params(**kwargs)

    def set_vis_delay(self, ms: int) -> None:
        """Set animation delay (ms) in the training loop."""
        if self._loop is not None:
            self._loop.set_vis_delay(ms)

    def update_grid(self, row: int, col: int, cell_type: CellType) -> None:
        """Change a cell type in the live environment."""
        self._env.set_cell(row, col, cell_type)

    def load_default_grid(self) -> None:
        """Restore the grid to the layout loaded at startup."""
        for r, row in enumerate(self._default_grid):
            for c, cell in enumerate(row):
                self._env.set_cell(r, c, cell)

    def get_grid(self) -> list[list[CellType]]:
        """Return a deep copy of the current grid."""
        return self._env.get_grid()

    def save_q_table(self, path: str | Path) -> None:
        """Save the current Q-Table to a .npy file."""
        save_q_table(self._agent.get_q_table(), path)

    def load_q_table(self, path: str | Path) -> None:
        """Load a Q-Table from a .npy file and install it in the agent."""
        q = load_q_table(path)
        self._agent.set_q_table(q)

    def save_grid_to_file(self, path: str | Path) -> None:
        """Save the current grid layout to a JSON file."""
        save_grid(self._env.get_grid(), path)

    def load_grid_from_file(self, path: str | Path) -> None:
        """Load a grid layout from a JSON file."""
        grid = load_grid(path)
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                self._env.set_cell(r, c, cell)
