"""Dataclass for per-episode training statistics and per-step updates."""

from dataclasses import dataclass, field

__all__ = ["EpisodeStats", "StepUpdate"]


@dataclass
class EpisodeStats:
    """Statistics collected at the end of each training episode.

    Attributes:
        episode: Sequential episode number (1-based).
        steps: Number of steps taken in the episode.
        total_reward: Cumulative reward received during the episode.
        max_delta_q: Maximum absolute Bellman update magnitude in the episode.
        reached_target: True if the drone reached the TARGET cell.
    """

    episode: int
    steps: int
    total_reward: float
    max_delta_q: float
    reached_target: bool


@dataclass
class StepUpdate:
    """Emitted by TrainingLoop after each step when vis_delay > 0.

    Allows the GUI to render the drone's position and trail in real time
    without waiting for the end of an episode.

    Attributes:
        state: Current drone grid-state index after the step.
        trail: Full path from episode start to current position (inclusive).
    """

    state: int
    trail: list[int] = field(default_factory=list)
