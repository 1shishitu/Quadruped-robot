"""Foot-end trajectory planner with quintic swing phase."""

from __future__ import annotations

import numpy as np

from quadruped.planners.gait_scheduler import GaitScheduler
from quadruped.types import FootPlannerDebug, FootRef, FootRefs, LegFootDebug
from quadruped.utils.quintic import QuinticPolynomial
from quadruped.utils.raibert import in_place_touchdown, raibert_foot_placement, rpy_to_rotation, yaw_rotation


class FootPlanner:
    """
    足端轨迹规划器.

    - in_place: 原地踏步，touchdown = stance 锚点（地面 z）
    - raibert:  行走，touchdown = stance + body 系速度预览
    """

    def __init__(
        self,
        gait: GaitScheduler,
        clearance: float = 0.08,
        raibert_kv: float = 0.04,
        raibert_kp: float = 0.0,
        *,
        placement: str = "in_place",
        stance_height: float = 0.0,
        default_stance_positions: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.gait = gait
        self.clearance = clearance
        self.raibert_kv = raibert_kv
        self.raibert_kp = raibert_kp
        self.placement = str(placement)
        self.stance_height = float(stance_height)
        self._stance: dict[str, np.ndarray] = {
            leg: np.zeros(3) for leg in gait.phase_offset
        }
        self._stance_body: dict[str, np.ndarray] = {
            leg: np.zeros(3) for leg in gait.phase_offset
        }
        self._use_body_stance = self.placement == "in_place"
        self._touchdown: dict[str, np.ndarray] = {
            leg: np.zeros(3) for leg in gait.phase_offset
        }
        self._was_swing: dict[str, bool] = {leg: False for leg in gait.phase_offset}
        if default_stance_positions:
            for leg, p in default_stance_positions.items():
                self._stance[leg] = np.asarray(p, dtype=float).copy()
                self._touchdown[leg] = self._anchor_stance(p)
        self.debug = FootPlannerDebug()

    def _anchor_stance(self, foot_pos: np.ndarray) -> np.ndarray:
        """Keep captured world-frame foot pose; do not project to z=0 (Go1 stand z≈0.18 m)."""
        p = np.asarray(foot_pos, dtype=float).copy()
        p[2] = float(p[2] + self.stance_height)
        return p

    def sample_swing_path(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        *,
        n: int = 12,
    ) -> np.ndarray:
        if n < 2:
            n = 2
        t_swing = self.gait.swing_duration
        if t_swing <= 0.0:
            return np.stack([np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float)])

        qx, qy, qz = QuinticPolynomial.vector3(
            p_start,
            np.zeros(3),
            np.zeros(3),
            p_end,
            np.zeros(3),
            np.zeros(3),
            t_swing,
        )
        samples = []
        for i in range(n):
            tau = t_swing * i / (n - 1)
            p = np.array([
                qx.position(tau),
                qy.position(tau),
                qz.position(tau),
            ], dtype=float)
            p[2] += self.clearance * np.sin(np.pi * tau / t_swing)
            samples.append(p)
        return np.stack(samples)

    def _world_stance_anchor(
        self,
        leg: str,
        base_pos: np.ndarray,
        *,
        base_rpy: np.ndarray | None = None,
        yaw: float = 0.0,
    ) -> np.ndarray:
        rot = rpy_to_rotation(base_rpy) if base_rpy is not None else yaw_rotation(yaw)
        bp = np.asarray(base_pos, dtype=float).reshape(3)
        return bp + rot @ self._stance_body[leg]

    def set_stance_positions(
        self,
        positions: dict[str, np.ndarray],
        *,
        base_pos: np.ndarray | None = None,
        base_rpy: np.ndarray | None = None,
        yaw: float = 0.0,
    ) -> None:
        rot = None
        bp = None
        if base_pos is not None:
            bp = np.asarray(base_pos, dtype=float).reshape(3)
            rot = (
                rpy_to_rotation(base_rpy)
                if base_rpy is not None
                else yaw_rotation(yaw)
            )
        for leg, p in positions.items():
            grounded = self._anchor_stance(p)
            self._stance[leg] = grounded.copy()
            self._touchdown[leg] = grounded.copy()
            if rot is not None and bp is not None:
                self._stance_body[leg] = rot.T @ (grounded - bp)
        self._was_swing = {leg: False for leg in self.gait.phase_offset}

    def _plan_touchdown(
        self,
        p_start: np.ndarray,
        v_body: np.ndarray,
        v_cmd: np.ndarray,
        *,
        yaw: float,
    ) -> np.ndarray:
        if self.placement == "in_place":
            return in_place_touchdown(p_start)
        return raibert_foot_placement(
            p_start,
            v_body,
            v_cmd,
            self.gait.period,
            yaw=yaw,
            kv=self.raibert_kv,
            kp=self.raibert_kp,
        )

    def update(
        self,
        t: float,
        v_body: np.ndarray,
        v_cmd: np.ndarray,
        hip_positions: dict[str, np.ndarray],
        *,
        yaw: float = 0.0,
        base_pos: np.ndarray | None = None,
        base_rpy: np.ndarray | None = None,
        foot_pos: dict[str, np.ndarray] | None = None,
    ) -> FootRefs:
        del hip_positions
        refs = {}
        debug_legs: dict[str, LegFootDebug] = {}
        swing_paths: dict[str, np.ndarray] = {}
        for leg in self.gait.phase_offset:
            ref, dbg = self._leg_ref(
                t,
                leg,
                v_body,
                v_cmd,
                yaw=yaw,
                base_pos=base_pos,
                base_rpy=base_rpy,
                foot_pos=foot_pos,
            )
            refs[leg] = ref
            debug_legs[leg] = dbg
            if not dbg.in_stance:
                swing_paths[leg] = self.sample_swing_path(
                    dbg.swing_start,
                    dbg.raibert_target,
                )
        self.debug = FootPlannerDebug(legs=debug_legs, swing_paths=swing_paths)
        return FootRefs(**refs)

    def _leg_ref(
        self,
        t: float,
        leg: str,
        v_body: np.ndarray,
        v_cmd: np.ndarray,
        *,
        yaw: float,
        base_pos: np.ndarray | None = None,
        base_rpy: np.ndarray | None = None,
        foot_pos: dict[str, np.ndarray] | None = None,
    ) -> tuple[FootRef, LegFootDebug]:
        phi = self.gait.phase(leg, t)
        in_stance = phi >= self.gait.swing_ratio
        was_swing = self._was_swing[leg]
        p_start = self._stance[leg].copy()

        if not in_stance and not was_swing:
            if foot_pos is not None and leg in foot_pos:
                p_start = np.asarray(foot_pos[leg], dtype=float).copy()
                self._stance[leg] = p_start.copy()
            if (
                self.placement == "in_place"
                and self._use_body_stance
                and base_pos is not None
            ):
                self._touchdown[leg] = self._world_stance_anchor(
                    leg, base_pos, base_rpy=base_rpy, yaw=yaw
                )
            else:
                self._touchdown[leg] = self._plan_touchdown(
                    p_start, v_body, v_cmd, yaw=yaw
                )

        p_end = self._touchdown[leg].copy()

        if in_stance and was_swing:
            self._stance[leg] = p_end.copy()

        self._was_swing[leg] = not in_stance

        if in_stance:
            if (
                self.placement == "in_place"
                and self._use_body_stance
                and base_pos is not None
            ):
                p = self._world_stance_anchor(
                    leg, base_pos, base_rpy=base_rpy, yaw=yaw
                )
            else:
                p = self._stance[leg].copy()
            dbg = LegFootDebug(
                leg=leg,
                phase=float(phi),
                in_stance=True,
                stance_anchor=p.copy(),
                raibert_target=p_end.copy(),
                swing_start=p.copy(),
                ref_position=p.copy(),
                ref_contact=True,
            )
            return FootRef(
                position=p,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                contact=True,
            ), dbg

        t_swing = self.gait.swing_duration
        t_in_swing = phi / self.gait.swing_ratio * t_swing if self.gait.swing_ratio > 0 else 0.0

        qx, qy, qz = QuinticPolynomial.vector3(
            p_start,
            np.zeros(3),
            np.zeros(3),
            p_end,
            np.zeros(3),
            np.zeros(3),
            t_swing,
        )
        position = np.array([
            qx.position(t_in_swing),
            qy.position(t_in_swing),
            qz.position(t_in_swing),
        ])
        velocity = np.array([
            qx.velocity(t_in_swing),
            qy.velocity(t_in_swing),
            qz.velocity(t_in_swing),
        ])
        acceleration = np.array([
            qx.acceleration(t_in_swing),
            qy.acceleration(t_in_swing),
            qz.acceleration(t_in_swing),
        ])
        position[2] += self.clearance * np.sin(np.pi * t_in_swing / t_swing) if t_swing > 0 else 0.0

        dbg = LegFootDebug(
            leg=leg,
            phase=float(phi),
            in_stance=False,
            stance_anchor=p_start.copy(),
            raibert_target=p_end.copy(),
            swing_start=p_start.copy(),
            ref_position=position.copy(),
            ref_contact=False,
        )
        return FootRef(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            contact=False,
        ), dbg

    @classmethod
    def from_config(cls, gait_cfg: dict, gait: GaitScheduler) -> FootPlanner:
        g = gait_cfg["gait"]
        foot = g.get("foot", {})
        raibert = g.get("raibert", {})
        return cls(
            gait,
            clearance=float(foot.get("clearance", 0.08)),
            raibert_kv=float(raibert.get("kv", 0.04)),
            raibert_kp=float(raibert.get("kp", 0.0)),
            placement=str(foot.get("placement", "in_place")),
            stance_height=float(foot.get("stance_height", 0.0)),
        )
