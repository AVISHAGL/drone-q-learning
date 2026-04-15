"""DroneSimSDK — single façade entry point for all simulation logic."""

import queue
from pathlib import Path

import numpy as np

from src.core.cell_type import CellType
from src.core.config_loader import load_env_config, load_rl_config
from src.core.episode_stats import EpisodeStats
from src.core.grid_world import GridWorld
from src.core.q_agent import QLearningAgent
from src.sdk.grid_sdk import GridSDK
from src.sdk.persistence import load_q_table, save_q_table
from src.sdk.training_loop import TrainingLoop

__all__ = ["DroneSimSDK"]


class DroneSimSDK:
    """Façade exposing all simulation capabilities to the GUI layer.

    Grid operations are delegated to GridSDK. GUI must import only from here.
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
        self._grid = GridSDK(self._env)

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
        """Clear Q-Table, episode counter, visited states, and stats queue."""
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

    def get_visited_states(self) -> frozenset[int]:
        """Return states visited by the agent since last reset."""
        return self._agent.get_visited_states()

    def set_hyperparams(self, **kwargs: float) -> None:
        """Update agent hyperparameters at runtime."""
        self._agent.update_params(**kwargs)

    def set_vis_delay(self, ms: int) -> None:
        """Set per-step animation delay in milliseconds (0 = fastest)."""
        if self._loop is not None:
            self._loop.set_vis_delay(ms)

    def get_vis_delay(self) -> int:
        """Return current per-step animation delay in milliseconds."""
        if self._loop is not None:
            return int(self._loop.vis_delay * 1000)
        return int(self._rl_cfg.get("vis_delay_ms", 0))

    def update_grid(self, row: int, col: int, cell_type: CellType) -> None:
        """Change a cell type in the live environment."""
        self._grid.update_grid(row, col, cell_type)

    def get_grid(self) -> list[list[CellType]]:
        """Return a deep copy of the current grid."""
        return self._grid.get_grid()

    def load_default_grid(self) -> None:
        """Restore the grid to the layout loaded at startup."""
        self._grid.load_default_grid()

    def save_q_table(self, path: str | Path) -> None:
        """Save the current Q-Table to a .npy file."""
        save_q_table(self._agent.get_q_table(), path)

    def load_q_table(self, path: str | Path) -> None:
        """Load a Q-Table from a .npy file and install it in the agent."""
        q = load_q_table(path)
        self._agent.set_q_table(q)

