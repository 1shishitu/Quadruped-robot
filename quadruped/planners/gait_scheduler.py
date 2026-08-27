"""Gait phase scheduler and contact state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GaitScheduler:
    """
    步态相位调度器.

    phi = (frequency * t + phase_offset) mod 1
    phi < swing_ratio → swing (contact=False)
    phi >= swing_ratio → stance (contact=True)
    """

    frequency: float = 2.0
    swing_ratio: float = 0.5
    phase_offset: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.phase_offset is None:
            self.phase_offset = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}

    @property
    def period(self) -> float:
        return 1.0 / self.frequency

    @property
    def swing_duration(self) -> float:
        return self.period * self.swing_ratio

    @property
    def stance_duration(self) -> float:
        return self.period * (1.0 - self.swing_ratio)

    def phase(self, leg: str, t: float) -> float:
        """Leg phase in [0, 1)."""
        return (self.frequency * t + self.phase_offset[leg]) % 1.0

    def is_stance(self, leg: str, t: float) -> bool:
        return self.phase(leg, t) >= self.swing_ratio

    def is_swing(self, leg: str, t: float) -> bool:
        return not self.is_stance(leg, t)

    def contact_state(self, t: float) -> dict[str, bool]:
        """Return {leg: in_stance} for all legs."""
        legs = self.phase_offset.keys()
        return {leg: self.is_stance(leg, t) for leg in legs}

    @classmethod
    def from_config(cls, cfg: dict) -> GaitScheduler:
        g = cfg["gait"]
        return cls(
            frequency=g["frequency"],
            swing_ratio=g["swing_ratio"],
            phase_offset=g["phase_offset"],
        )
