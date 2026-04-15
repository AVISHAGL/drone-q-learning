"""Background training loop running Q-Learning in a daemon thread."""

import queue
import threading
import time

from src.core.episode_stats import StepUpdate
from src.core.grid_world import GridWorld
from src.core.q_agent import QLearningAgent

__all__ = ["TrainingLoop"]


class TrainingLoop:
    """Runs the Q-Learning episode loop in a daemon background thread.

    Communication with the main (GUI) thread happens exclusively via
    a queue.Queue: an EpisodeStats object is posted after every episode.
    Pause/stop control uses threading.Event flags.

    Args:
        env: GridWorld environment.
        agent: QLearningAgent to train.
        stats_queue: Queue for posting EpisodeStats to the main thread.
        config: RL config dict (used for 'episodes' and 'max_steps').
    """

    def __init__(
        self,
        env: GridWorld,
        agent: QLearningAgent,
        stats_queue: queue.Queue,
        config: dict,
    ) -> None:
        """Initialise loop without starting the thread."""
        self._env = env
        self._agent = agent
        self._queue = stats_queue
        self._episodes = int(config["episodes"])
        self._max_steps = int(config["max_steps"])
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.vis_delay: float = float(config.get("vis_delay_ms", 0)) / 1000.0

    def start(self) -> None:
        """Start the training thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Signal the training loop to pause after the current episode."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume a paused training loop."""
        self._pause_event.clear()

    def stop(self) -> None:
        """Signal the training loop to stop; waits up to 3 seconds."""
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def set_vis_delay(self, ms: int) -> None:
        """Set the per-step visualisation delay.

        Args:
            ms: Delay in milliseconds (0 = no delay).
        """
        self.vis_delay = ms / 1000.0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run(self) -> None:
        for _ in range(self._episodes):
            if self._stop_event.is_set():
                break
            while self._pause_event.is_set():
                if self._stop_event.is_set():
                    return
                threading.Event().wait(0.05)

            state = self._env.reset()
            trail: list[int] = [state]
            reached = False
            for _ in range(self._max_steps):
                if self._stop_event.is_set():
                    return
                action = self._agent.select_action(state)
                next_state, reward, done = self._env.step(state, action)
                self._agent.update(state, action, reward, next_state)
                state = next_state
                trail.append(state)
                self._queue.put(StepUpdate(state=state, trail=trail.copy()))
                if self.vis_delay > 0:
                    time.sleep(self.vis_delay)
                if done:
                    reached = True
                    break

            stats = self._agent.end_episode(reached_target=reached)
            self._queue.put(stats)
