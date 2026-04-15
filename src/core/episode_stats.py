"""Dataclass for per-episode training statistics."""

from dataclasses import dataclass

__all__ = ["EpisodeStats"]


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
