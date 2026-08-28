"""FL-only vertical lift / hold / lower cycle (other legs fixed stance)."""

from __future__ import annotations

import numpy as np

from quadruped.types import FootPlannerDebug, FootRef, FootRefs, LegFootDebug
from quadruped.utils.raibert import rpy_to_rotation


def _smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


class FlLiftPlanner:
    """One swing leg: stance → lift (vertical) → hold → lower → repeat."""

    def __init__(
        self,
        *,
        swing_leg: str = "FL",
        frequency: float = 0.25,
        clearance: float = 0.05,
        stance_ratio: float = 0.35,
        lift_ratio: float = 0.20,
        hold_ratio: float = 0.25,
        lower_ratio: float = 0.20,
        leg_names: tuple[str, ...] = ("FL", "FR", "RL", "RR"),
    ) -> None:
        self.swing_leg = swing_leg
        self.frequency = float(frequency)
        self.clearance = float(clearance)
        self.leg_names = leg_names
        ratios = np.array([stance_ratio, lift_ratio, hold_ratio, lower_ratio], dtype=float)
        ratios = ratios / ratios.sum()
        edges = np.concatenate([[0.0], np.cumsum(ratios)])
        self._edges = edges
        self._labels = ("stance", "lift", "hold", "lower")
        self._stance_body: dict[str, np.ndarray] = {leg: np.zeros(3) for leg in leg_names}
        self.debug = FootPlannerDebug()

    @property
    def period(self) -> float:
        return 1.0 / self.frequency

    def set_stance_positions(
        self,
        positions: dict[str, np.ndarray],
        *,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
    ) -> None:
        rot = rpy_to_rotation(base_rpy)
        bp = np.asarray(base_pos, dtype=float).reshape(3)
        for leg, p in positions.items():
            pw = np.asarray(p, dtype=float).reshape(3)
            self._stance_body[leg] = rot.T @ (pw - bp)

    def _anchor_world(
        self, leg: str, base_pos: np.ndarray, base_rpy: np.ndarray
    ) -> np.ndarray:
        rot = rpy_to_rotation(base_rpy)
        bp = np.asarray(base_pos, dtype=float).reshape(3)
        return bp + rot @ self._stance_body[leg]

    def _segment(self, phi: float) -> tuple[str, float]:
        for i, label in enumerate(self._labels):
            if phi < self._edges[i + 1] - 1e-12:
                lo, hi = self._edges[i], self._edges[i + 1]
                local = (phi - lo) / max(hi - lo, 1e-9)
                return label, float(local)
        return "stance", 0.0

    def update(
        self,
        t: float,
        *,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
    ) -> FootRefs:
        phi = (self.frequency * t) % 1.0
        seg, local = self._segment(phi)
        refs: dict[str, FootRef] = {}
        debug_legs: dict[str, LegFootDebug] = {}

        for leg in self.leg_names:
            anchor = self._anchor_world(leg, base_pos, base_rpy)
            if leg != self.swing_leg:
                refs[leg] = FootRef(
                    position=anchor.copy(),
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    contact=True,
                )
                debug_legs[leg] = LegFootDebug(
                    leg=leg,
                    phase=phi,
                    in_stance=True,
                    stance_anchor=anchor.copy(),
                    raibert_target=anchor.copy(),
                    swing_start=anchor.copy(),
                    ref_position=anchor.copy(),
                    ref_contact=True,
                )
                continue

            z0 = float(anchor[2])
            pos = anchor.copy()
            vel = np.zeros(3)
            contact = True

            if seg == "stance":
                contact = True
            elif seg == "lift":
                contact = False
                s = _smoothstep(local)
                pos[2] = z0 + self.clearance * s
            elif seg == "hold":
                contact = False
                pos[2] = z0 + self.clearance
            elif seg == "lower":
                contact = False
                s = _smoothstep(local)
                pos[2] = z0 + self.clearance * (1.0 - s)

            refs[leg] = FootRef(
                position=pos,
                velocity=vel,
                acceleration=np.zeros(3),
                contact=contact,
            )
            debug_legs[leg] = LegFootDebug(
                leg=leg,
                phase=phi,
                in_stance=contact,
                stance_anchor=anchor.copy(),
                raibert_target=anchor.copy(),
                swing_start=anchor.copy(),
                ref_position=pos.copy(),
                ref_contact=contact,
            )

        self.debug = FootPlannerDebug(legs=debug_legs, swing_paths={})
        return FootRefs(**refs)

    @classmethod
    def from_config(cls, gait_cfg: dict, robot_cfg: dict | None = None) -> FlLiftPlanner:
        g = gait_cfg["gait"]
        lift = g.get("lift", {})
        legs = tuple((robot_cfg or {}).get("leg_names", ["FL", "FR", "RL", "RR"]))
        return cls(
            swing_leg=str(g.get("swing_leg", "FL")),
            frequency=float(g.get("frequency", 0.25)),
            clearance=float(lift.get("clearance", 0.05)),
            stance_ratio=float(lift.get("stance_ratio", 0.35)),
            lift_ratio=float(lift.get("lift_ratio", 0.20)),
            hold_ratio=float(lift.get("hold_ratio", 0.25)),
            lower_ratio=float(lift.get("lower_ratio", 0.20)),
            leg_names=legs,
        )
