"""Exploration schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSchedule:
    """Linearly interpolate from ``start`` to ``end`` over ``duration`` steps, then hold.

    epsilon(t) = end + (start - end) * max(0, 1 - t / duration)

    The schedule is indexed by *environment steps* for both agents, so the exploration
    budget of Q-Learning and DQN can be compared on the same x-axis.
    """

    start: float
    end: float
    duration: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.end <= 1.0 and 0.0 <= self.start <= 1.0):
            raise ValueError("start and end must lie in [0, 1]")
        if self.duration <= 0:
            raise ValueError("duration must be a positive number of steps")

    def __call__(self, step: int) -> float:
        fraction = min(max(step, 0) / self.duration, 1.0)
        return self.start + fraction * (self.end - self.start)
