"""Shared data types for planners and controllers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Vec3 = np.ndarray  # shape (3,)


@dataclass
class TrunkRef:
    """躯干 / 质心参考轨迹."""

    p_com: Vec3
    v_com: Vec3
    a_com: Vec3
    rpy: Vec3
    omega: Vec3


@dataclass
class FootRef:
    """单足参考轨迹."""

    position: Vec3
    velocity: Vec3
    acceleration: Vec3
    contact: bool


@dataclass
class FootRefs:
    """四条腿足端参考."""

    FL: FootRef
    FR: FootRef
    RL: FootRef
    RR: FootRef

    def by_leg(self, leg: str) -> FootRef:
        return getattr(self, leg)

    def legs(self) -> tuple[str, ...]:
        return ("FL", "FR", "RL", "RR")


@dataclass
class LegFootDebug:
    """FootPlanner 调试量（Raibert + quintic）."""

    leg: str
    phase: float
    in_stance: bool
    stance_anchor: Vec3
    raibert_target: Vec3
    swing_start: Vec3
    ref_position: Vec3
    ref_contact: bool


@dataclass
class FootPlannerDebug:
    """最近一次 FootPlanner.update 的可视化快照."""

    legs: dict[str, LegFootDebug] = field(default_factory=dict)
    swing_paths: dict[str, Vec3] = field(default_factory=dict)


@dataclass
class JointCommand:
    """关节层输出."""

    q_des: np.ndarray
    dq_des: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray = field(default_factory=lambda: np.array([]))
    kd: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class RobotState:
    """仿真 / 实机反馈状态（占位）."""

    t: float
    q: np.ndarray
    dq: np.ndarray
    base_pos: Vec3
    base_vel: Vec3
    base_rpy: Vec3
    base_omega: Vec3
    contact: dict[str, bool] = field(default_factory=dict)
